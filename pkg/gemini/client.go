// Package gemini is a reusable client for Google's Gemini web backend
// (the BardChatUi batchexecute endpoints). It handles cookie/SAPISIDHASH auth,
// dynamic build-label and model-catalog resolution, cookie rotation, image
// uploads, and blocking or streaming text generation.
//
// A typical consumer builds a Config, calls New, optionally resolves the build
// label / auth / model list at startup, and then calls Generate or
// GenerateStream:
//
//	c, err := gemini.New(gemini.Config{CookieFile: "cookie.txt"}, logger)
//	c.CheckAuth(ctx)
//	c.ResolveModels(ctx)
//	m, _ := c.ResolveModelOrDefault("gemini-3.5-flash")
//	text, err := c.Generate(ctx, m.Params("Hello"))
package gemini

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/n0madic/go-gemini-web2api/pkg/util"
)

// geminiBaseURL is the public Gemini host serving the BardChatUi batchexecute endpoint.
const geminiBaseURL = "https://gemini.google.com"

// userAgent mirrors a desktop Chrome UA, as the reference does.
const userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

// codeArtifactRe strips internal code-execution artifacts that Gemini embeds in
// streamed text (e.g. ```python?code_reference&code_event_index=0 ... ```).
var codeArtifactRe = regexp.MustCompile("(?s)```(?:python|javascript|text)\\?code_(?:reference|stdout)&code_event_index=\\d+\\n.*?```\\n?")

// cardContentRe strips googleusercontent card-content reference URLs.
var cardContentRe = regexp.MustCompile(`http://googleusercontent\.com/card_content/\d+\n?`)

// ErrEmptyResponse indicates Gemini returned a 200 with no extractable text,
// typically because the prompt exceeded the web endpoint's size limit. Generate
// returns it (unwrapped) so callers can detect this case with errors.Is.
var ErrEmptyResponse = errors.New("empty response from Gemini (prompt may exceed the web endpoint size limit)")

// buildLabelRe extracts the build label (bl) from WIZ_global_data on the Gemini page.
var buildLabelRe = regexp.MustCompile(`"cfb2h":"(boq_[^"]+)"`)

// snlm0eRe captures the SNlM0e (XSRF/"at") token, which the Gemini page embeds
// only for a signed-in session: its presence proves the cookie authenticates, and
// its value is required as the "at" form field on authenticated StreamGenerate calls.
var snlm0eRe = regexp.MustCompile(`"SNlM0e":"([^"]+)"`)

// buildLabelTTL is how long an auto-resolved build label is reused before refresh.
const buildLabelTTL = 30 * time.Minute

// Session represents an individual account or anonymous session with its own
// cookie, SAPISID, and XSRF at-token.
type Session struct {
	mu         sync.Mutex
	ID         int
	CookieStr  string
	SAPISID    string
	AtToken    string
	AuthStatus string
	FileBacked bool
	FilePath   string
}

// Client talks to the Gemini web backend. It is safe for concurrent use.
type Client struct {
	cfg  Config
	http *http.Client
	log  *slog.Logger

	// cookie file cache, keyed by file mtime to avoid re-reading every request.
	cookieMu     sync.Mutex
	cookieStr    string
	cookieSAPI   string
	cookieMtime  time.Time
	cookieLoaded bool

	// Concurrency limiter channel semaphore
	sem chan struct{}

	// Multi-session pool for round-robin concurrent load balancing
	sessMu   sync.Mutex
	sessions []*Session
	sessIdx  uint64

	// Build label, optionally auto-resolved from the Gemini page.
	blMu         sync.Mutex
	bl           string
	blAuto       bool
	blAt         time.Time
	blRefreshing bool

	// XSRF "at" token (SNlM0e), scraped from the signed-in page. Required on
	// authenticated StreamGenerate requests; anonymous requests omit it.
	atMu    sync.Mutex
	atToken string

	// Model registry, populated from the listModels RPC (or the static fallback
	// set). modelOrder preserves a stable listing order.
	modelMu          sync.Mutex
	models           map[string]*AvailableModel
	modelOrder       []string
	modelsSource     string // "dynamic" | "fallback" | "static"
	modelsAt         time.Time
	modelsRefreshing bool
}

// New builds a Client with an HTTP client honouring the configured timeout,
// proxy, connection pooling, and concurrency limit. Any Config field left at
// its zero value is filled with the matching Default* constant (see
// Config.applyDefaults), so gemini.Config{} — optionally with just a cookie —
// yields a working client. A nil logger defaults to slog.Default().
func New(cfg Config, logger *slog.Logger) (*Client, error) {
	cfg.applyDefaults()
	if logger == nil {
		logger = slog.Default()
	}

	proxyFunc := http.ProxyFromEnvironment
	if cfg.Proxy != "" {
		pu, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL: %w", err)
		}
		proxyFunc = http.ProxyURL(pu)
	}

	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	transport := &http.Transport{
		Proxy:                 proxyFunc,
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          500,
		MaxIdleConnsPerHost:   200, // high-concurrency connection pool
		MaxConnsPerHost:       500,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	var sem chan struct{}
	if cfg.MaxConcurrency > 0 {
		sem = make(chan struct{}, cfg.MaxConcurrency)
	}

	c := &Client{
		cfg: cfg,
		http: &http.Client{
			Timeout:   time.Duration(cfg.RequestTimeout) * time.Second,
			Transport: transport,
		},
		log:    logger,
		bl:     cfg.GeminiBL,
		blAuto: cfg.GeminiBLAuto,
		sem:    sem,
	}
	c.initSessions()
	return c, nil
}

// GenParams carries everything needed for a single generation request.
type GenParams struct {
	Prompt string
	Images []InputImage    // images to attach (uploaded just before generation)
	Model  *AvailableModel // selected model (drives the selection headers)
}

// InputImage is an image to attach to a generation request. Exactly one of Data
// (inline bytes already decoded from base64) or URL (a remote http(s) image to
// fetch at upload time) is populated.
type InputImage struct {
	Data     []byte
	URL      string
	Filename string
}

// buildBody constructs the urlencoded request body for StreamGenerate. The Gemini
// backend expects f.req to be [null, "<inner-json-string>"] where the inner payload
// is a sparse 69-element array with specific indices populated. Attached images
// (already uploaded) are referenced at inner[0][3]. The model is selected via HTTP
// headers (see AvailableModel.headers), not the payload; inner[17] is a constant.
//
// It also returns the per-request UUID (inner[59]); the caller must echo it in the
// initSessions populates the session pool from Config.Cookies, Config.CookieFile,
// Config.Cookie, or an anonymous fallback session.
func (c *Client) initSessions() {
	c.sessMu.Lock()
	defer c.sessMu.Unlock()

	var sessions []*Session

	// 1. Explicit multi-cookie list
	if len(c.cfg.Cookies) > 0 {
		for i, ck := range c.cfg.Cookies {
			ck = strings.TrimSpace(ck)
			if ck == "" {
				continue
			}
			sapi := cookieValue(ck, cookieSAPISID)
			sessions = append(sessions, &Session{
				ID:        i,
				CookieStr: ck,
				SAPISID:   sapi,
			})
		}
	}

	// 2. Cookie file (supports multi-line cookies for account pooling)
	if len(sessions) == 0 && c.cfg.CookieFile != "" {
		if data, err := os.ReadFile(c.cfg.CookieFile); err == nil {
			lines := strings.Split(string(data), "\n")
			idx := 0
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "#") {
					continue
				}
				sapi := cookieValue(line, cookieSAPISID)
				sessions = append(sessions, &Session{
					ID:         idx,
					CookieStr:  line,
					SAPISID:    sapi,
					FileBacked: true,
					FilePath:   c.cfg.CookieFile,
				})
				idx++
			}
		}
	}

	// 3. Single inline cookie
	if len(sessions) == 0 && c.cfg.Cookie != "" {
		sapi := cookieValue(c.cfg.Cookie, cookieSAPISID)
		sessions = append(sessions, &Session{
			ID:        0,
			CookieStr: c.cfg.Cookie,
			SAPISID:   sapi,
		})
	}

	// 4. Anonymous fallback
	if len(sessions) == 0 {
		sessions = append(sessions, &Session{
			ID:         0,
			AuthStatus: "anonymous",
		})
	}

	c.sessions = sessions
}

// nextSession returns the next session from the pool using round-robin.
func (c *Client) nextSession() *Session {
	c.sessMu.Lock()
	defer c.sessMu.Unlock()
	if len(c.sessions) == 0 {
		return &Session{AuthStatus: "anonymous"}
	}
	idx := atomic.AddUint64(&c.sessIdx, 1) % uint64(len(c.sessions))
	return c.sessions[idx]
}

// primarySession returns the first session (or default anonymous session).
func (c *Client) primarySession() *Session {
	c.sessMu.Lock()
	defer c.sessMu.Unlock()
	if len(c.sessions) == 0 {
		return &Session{AuthStatus: "anonymous"}
	}
	return c.sessions[0]
}

// SessionCount returns the number of active sessions in the pool.
func (c *Client) SessionCount() int {
	c.sessMu.Lock()
	defer c.sessMu.Unlock()
	return len(c.sessions)
}

// acquireConcurrency claims a slot under the concurrency limiter, or blocks
// until one is available or ctx is cancelled.
func (c *Client) acquireConcurrency(ctx context.Context) (func(), error) {
	if c.sem == nil {
		return func() {}, nil
	}
	select {
	case c.sem <- struct{}{}:
		return func() { <-c.sem }, nil
	case <-ctx.Done():
		return nil, fmt.Errorf("concurrency limit (%d) reached: %w", c.cfg.MaxConcurrency, ctx.Err())
	}
}

// ActiveRequests returns the number of currently active in-flight upstream requests.
func (c *Client) ActiveRequests() int {
	if c.sem == nil {
		return 0
	}
	return len(c.sem)
}

// MaxConcurrency returns the configured max concurrency (or 0 if unlimited).
func (c *Client) MaxConcurrency() int {
	return c.cfg.MaxConcurrency
}

// buildBody constructs the urlencoded request body for StreamGenerate.
func (c *Client) buildBody(p GenParams, images []uploadedImage) (body, reqUUID string, err error) {
	return c.buildBodyWithSession(p, images, c.primarySession())
}

// buildBodyWithSession constructs the urlencoded request body using the given session's at token.
func (c *Client) buildBodyWithSession(p GenParams, images []uploadedImage, sess *Session) (body, reqUUID string, err error) {
	reqUUID = strings.ToUpper(uuid4())
	inner := make([]any, 69)
	inner[0] = []any{p.Prompt, 0, nil, imageRefsField(images), nil, nil, 0}
	inner[1] = []any{"en"}
	inner[2] = []any{"", "", "", nil, nil, nil, nil, nil, nil, ""}
	inner[6] = []any{1}
	inner[7] = 1
	inner[10] = 1
	inner[11] = 0
	inner[17] = []any{[]any{0}} // constant (reasoning level is not exposed)
	inner[18] = 0
	inner[27] = 1
	inner[30] = []any{4}
	inner[41] = []any{1}
	inner[53] = 0
	inner[59] = reqUUID
	inner[61] = []any{} // must serialize as [] not null
	inner[68] = 2

	innerJSON, err := util.MarshalNoEscape(inner)
	if err != nil {
		return "", "", err
	}
	outerJSON, err := util.MarshalNoEscape([]any{nil, string(innerJSON)})
	if err != nil {
		return "", "", err
	}

	vals := url.Values{}
	vals.Set("f.req", string(outerJSON))
	at := ""
	if sess != nil {
		sess.mu.Lock()
		at = sess.AtToken
		sess.mu.Unlock()
	}
	if at == "" {
		at = c.currentAt()
	}
	if at != "" {
		vals.Set("at", at)
	}
	return vals.Encode(), reqUUID, nil
}

// streamURL builds the StreamGenerate URL with a per-request id.
func (c *Client) streamURL() string {
	reqid := time.Now().Unix() % 1000000
	prefix := c.accountPrefix()
	return fmt.Sprintf(
		"%s%s/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"+
			"?bl=%s&hl=en&_reqid=%d&rt=c",
		geminiBaseURL, prefix, url.QueryEscape(c.CurrentBL()), reqid)
}

// CurrentBL returns the active build label, kicking off a background refresh when
// auto-resolution is enabled and the cached value has gone stale.
func (c *Client) CurrentBL() string {
	c.blMu.Lock()
	bl := c.bl
	stale := c.blAuto && !c.blRefreshing && (c.blAt.IsZero() || time.Since(c.blAt) > buildLabelTTL)
	if stale {
		c.blRefreshing = true
	}
	c.blMu.Unlock()
	if stale {
		go c.fetchAndStoreBL(context.Background())
	}
	return bl
}

// ResolveBuildLabel synchronously resolves the build label (used at startup). It
// is a no-op when auto-resolution is disabled.
func (c *Client) ResolveBuildLabel(ctx context.Context) {
	if !c.blAuto {
		return
	}
	c.blMu.Lock()
	if c.blRefreshing {
		c.blMu.Unlock()
		return
	}
	c.blRefreshing = true
	c.blMu.Unlock()
	c.fetchAndStoreBL(ctx)
}

// fetchAndStoreBL fetches the build label and stores it, recording the attempt
// time even on failure so refreshes are not retried too aggressively.
func (c *Client) fetchAndStoreBL(ctx context.Context) {
	bl := c.fetchBuildLabel(ctx)
	c.blMu.Lock()
	if bl != "" {
		c.bl = bl
	}
	c.blAt = time.Now()
	c.blRefreshing = false
	c.blMu.Unlock()
	if bl != "" {
		c.log.Info("resolved gemini build label", "bl", bl)
	} else {
		c.log.Warn("could not resolve gemini build label; using fallback", "fallback", c.cfg.GeminiBL)
	}
}

// fetchAppPage GETs the Gemini /app page (with the cookie when present); it is the
// shared source for scraping the build label and the SNlM0e XSRF token. The body is
// capped at 4MB.
func (c *Client) fetchAppPage(ctx context.Context) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, geminiBaseURL+c.accountPrefix()+"/app", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if cookie, _ := c.loadCookie(); cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(io.LimitReader(resp.Body, 4<<20))
}

// fetchBuildLabel scrapes the current build label from the Gemini app page.
func (c *Client) fetchBuildLabel(ctx context.Context) string {
	html, err := c.fetchAppPage(ctx)
	if err != nil {
		c.log.Warn("build label fetch failed", "err", err)
		return ""
	}
	if m := buildLabelRe.FindSubmatch(html); m != nil {
		return string(m[1])
	}
	return ""
}

// accountPrefix returns the /u/N path segment for non-default Google accounts.
func (c *Client) accountPrefix() string {
	if c.cfg.AuthUser == "" {
		return ""
	}
	return "/u/" + c.cfg.AuthUser
}

// CheckAuth reports whether the configured cookies establish signed-in Gemini
// sessions, detected via the SNlM0e token (issued only to authenticated sessions).
// It returns a human-readable status indicating how many sessions authenticated.
func (c *Client) CheckAuth(ctx context.Context) string {
	c.sessMu.Lock()
	sessions := append([]*Session(nil), c.sessions...)
	c.sessMu.Unlock()

	cookie, _ := c.loadCookie()
	if len(sessions) == 0 || (len(sessions) == 1 && cookie == "") {
		return "anonymous"
	}

	authenticatedCount := 0
	for _, sess := range sessions {
		sess.mu.Lock()
		cStr := sess.CookieStr
		sess.mu.Unlock()
		if cStr == "" {
			cStr = cookie
		}
		if cStr == "" {
			continue
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, geminiBaseURL+c.accountPrefix()+"/app", nil)
		if err != nil {
			continue
		}
		req.Header.Set("User-Agent", userAgent)
		req.Header.Set("Cookie", cStr)
		resp, err := c.http.Do(req)
		if err != nil {
			continue
		}
		html, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()

		if m := snlm0eRe.FindSubmatch(html); m != nil {
			tok := string(m[1])
			sess.mu.Lock()
			sess.AtToken = tok
			sess.AuthStatus = "authenticated"
			sess.mu.Unlock()
			authenticatedCount++
		} else {
			sess.mu.Lock()
			sess.AuthStatus = "not authenticated"
			sess.mu.Unlock()
		}
	}

	if authenticatedCount > 0 {
		c.setAt(sessions[0].AtToken)
		c.log.Info("gemini cookie authenticated", "authenticated_sessions", authenticatedCount, "total_sessions", len(sessions))
		if len(sessions) > 1 {
			return fmt.Sprintf("authenticated (%d/%d sessions active)", authenticatedCount, len(sessions))
		}
		return "authenticated"
	}

	c.log.Warn("gemini cookie is NOT authenticating — requests run anonymously",
		"hint", "the session needs __Secure-1PSIDTS (it expires; re-export the cookie)")
	return "NOT authenticated"
}

// currentAt returns the XSRF "at" token to send with StreamGenerate: the SNlM0e
// value scraped from the signed-in page.
func (c *Client) currentAt() string {
	sess := c.primarySession()
	sess.mu.Lock()
	defer sess.mu.Unlock()
	if sess.AtToken != "" {
		return sess.AtToken
	}
	c.atMu.Lock()
	defer c.atMu.Unlock()
	return c.atToken
}

// setAt caches the scraped SNlM0e XSRF token across the client and primary session.
func (c *Client) setAt(token string) {
	c.atMu.Lock()
	c.atToken = token
	c.atMu.Unlock()
	sess := c.primarySession()
	sess.mu.Lock()
	sess.AtToken = token
	sess.mu.Unlock()
}

// ImagesSupported reports whether image attachments can be used. They require an
// authenticated session (a non-empty XSRF "at" token); anonymous sessions — and
// sessions whose cookie no longer authenticates — cannot attach images.
func (c *Client) ImagesSupported() bool {
	return c.currentAt() != ""
}

// refreshAtToken re-scrapes the SNlM0e XSRF token from the signed-in page and
// caches it. Called after a cookie rotation so the token stays in sync with the
// refreshed session.
func (c *Client) refreshAtToken(ctx context.Context) {
	c.sessMu.Lock()
	sessions := append([]*Session(nil), c.sessions...)
	c.sessMu.Unlock()

	for _, sess := range sessions {
		sess.mu.Lock()
		ck := sess.CookieStr
		sess.mu.Unlock()
		if ck == "" {
			ck, _ = c.loadCookie()
		}
		if ck == "" {
			continue
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, geminiBaseURL+c.accountPrefix()+"/app", nil)
		if err != nil {
			continue
		}
		req.Header.Set("User-Agent", userAgent)
		req.Header.Set("Cookie", ck)
		resp, err := c.http.Do(req)
		if err != nil {
			continue
		}
		html, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if m := snlm0eRe.FindSubmatch(html); m != nil {
			sess.mu.Lock()
			sess.AtToken = string(m[1])
			sess.mu.Unlock()
		}
	}
	if len(sessions) > 0 {
		c.setAt(sessions[0].AtToken)
	}
}

// newRequest assembles a POST request with all required headers and auth.
func (c *Client) newRequest(ctx context.Context, body string, model *AvailableModel, reqUUID string) (*http.Request, error) {
	return c.newRequestWithSession(ctx, body, model, reqUUID, c.primarySession())
}

// newRequestWithSession assembles a POST request using the designated session credentials.
func (c *Client) newRequestWithSession(ctx context.Context, body string, model *AvailableModel, reqUUID string, sess *Session) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.streamURL(), strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	prefix := c.accountPrefix()
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", geminiBaseURL)
	req.Header.Set("Referer", geminiBaseURL+prefix+"/app")
	req.Header.Set("X-Same-Domain", "1")
	req.Header.Set("User-Agent", userAgent)
	if prefix != "" {
		req.Header.Set("X-Goog-AuthUser", c.cfg.AuthUser)
	}
	if model != nil && model.ModelID != "" {
		for k, v := range model.headers(reqUUID) {
			req.Header.Set(k, v)
		}
	}

	var cookie, sapisid string
	if sess != nil {
		sess.mu.Lock()
		cookie = sess.CookieStr
		sapisid = sess.SAPISID
		sess.mu.Unlock()
	}
	if cookie == "" {
		cookie, sapisid = c.loadCookie()
	}
	if cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	if sapisid != "" {
		req.Header.Set("Authorization", makeSapisidHash(sapisid))
	}
	return req, nil
}

// Generate sends a prompt and returns the final response text (blocking, with retry
// and concurrency throttling).
func (c *Client) Generate(ctx context.Context, p GenParams) (string, error) {
	release, err := c.acquireConcurrency(ctx)
	if err != nil {
		return "", err
	}
	defer release()

	sess := c.nextSession()
	images := c.uploadImages(ctx, p.Images)
	body, reqUUID, err := c.buildBodyWithSession(p, images, sess)
	if err != nil {
		return "", err
	}

	var lastErr error
	for attempt := 0; attempt < c.cfg.RetryAttempts; attempt++ {
		req, err := c.newRequestWithSession(ctx, body, p.Model, reqUUID, sess)
		if err != nil {
			return "", err
		}
		resp, err := c.http.Do(req)
		if err == nil {
			raw, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			switch {
			case readErr != nil:
				err = readErr
			default:
				if text := extractResponseText(string(raw)); text != "" {
					return text, nil
				}
				err = ErrEmptyResponse
			}
		}
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		lastErr = err
		if attempt < c.cfg.RetryAttempts-1 {
			c.log.Warn("retrying gemini request",
				"attempt", attempt+1, "max", c.cfg.RetryAttempts, "err", err)
			backoff := time.Duration(c.cfg.RetryDelaySec) * time.Second * (1 << attempt)
			if backoff > 30*time.Second {
				backoff = 30 * time.Second
			}
			jitter := time.Duration(time.Now().UnixNano()%300) * time.Millisecond
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(backoff + jitter):
			}
		}
	}
	// ErrEmptyResponse is already self-descriptive; don't double-wrap it.
	if errors.Is(lastErr, ErrEmptyResponse) {
		return "", ErrEmptyResponse
	}
	return "", fmt.Errorf("gemini request failed: %w", lastErr)
}

// GenerateStream sends a prompt and invokes emit for each incremental text delta.
func (c *Client) GenerateStream(ctx context.Context, p GenParams, emit func(delta string)) error {
	release, err := c.acquireConcurrency(ctx)
	if err != nil {
		return err
	}
	defer release()

	sess := c.nextSession()
	images := c.uploadImages(ctx, p.Images)
	body, reqUUID, err := c.buildBodyWithSession(p, images, sess)
	if err != nil {
		return err
	}
	req, err := c.newRequestWithSession(ctx, body, p.Model, reqUUID, sess)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("gemini request failed: %w", err)
	}
	defer resp.Body.Close()

	reader := bufio.NewReader(resp.Body)
	prevText := ""
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, readErr := reader.ReadString('\n')
		if line != "" {
			if text, ok := latestText(line); ok && len(text) > len(prevText) {
				delta := cleanGeminiText(runeDelta(prevText, text))
				prevText = text
				if delta != "" {
					emit(delta)
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return nil
			}
			return readErr
		}
	}
}

// runeDelta returns the part of cur that follows the longest rune-aligned prefix
// it shares with prev. Gemini streams cumulative snapshots, so cur is normally a
// byte-prefix extension of prev and the delta is simply its tail; but if a
// snapshot revises earlier text (so prev is not a byte-prefix of cur), slicing at
// len(prev) could land in the middle of a multi-byte rune. Diffing on rune
// boundaries keeps every emitted delta valid UTF-8.
func runeDelta(prev, cur string) string {
	if strings.HasPrefix(cur, prev) {
		// Fast path: prev ends on a rune boundary within cur, so the tail is safe.
		return cur[len(prev):]
	}
	i := 0
	for i < len(prev) && i < len(cur) {
		rp, size := utf8.DecodeRuneInString(prev[i:])
		rc, _ := utf8.DecodeRuneInString(cur[i:])
		if rp != rc {
			break
		}
		i += size
	}
	return cur[i:]
}

// extractResponseText parses a full StreamGenerate response and returns the final
// text. Gemini streams cumulative snapshots, so the longest candidate is the
// complete answer (more robust than picking the last non-empty one).
func extractResponseText(raw string) string {
	longest := ""
	for _, line := range strings.Split(raw, "\n") {
		for _, t := range parseWrbLine(line) {
			if len(t) > len(longest) {
				longest = t
			}
		}
	}
	return cleanGeminiText(longest)
}

// latestText returns the longest text candidate from a single streamed line.
func latestText(line string) (string, bool) {
	texts := parseWrbLine(line)
	best := ""
	for _, t := range texts {
		if len(t) > len(best) {
			best = t
		}
	}
	return best, best != ""
}

// parseWrbLine extracts text strings from a batchexecute "wrb.fr" response line.
// The line is JSON like [["wrb.fr", _, "<inner-json>", ...], ...]; the inner JSON
// nests the model text at inner2[4][*][1][*].
func parseWrbLine(line string) []string {
	if !strings.Contains(line, `"wrb.fr"`) || len(line) < 200 {
		return nil
	}
	var arr []any
	if err := json.Unmarshal([]byte(line), &arr); err != nil {
		return nil
	}
	first, ok := indexSlice(arr, 0)
	if !ok {
		return nil
	}
	innerStr, ok := indexString(first, 2)
	if !ok || len(innerStr) < 50 {
		return nil
	}
	var inner2 any
	if err := json.Unmarshal([]byte(innerStr), &inner2); err != nil {
		return nil
	}
	inner2Slice, ok := inner2.([]any)
	if !ok || len(inner2Slice) <= 4 {
		return nil
	}
	parts, ok := inner2Slice[4].([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, part := range parts {
		ps, ok := part.([]any)
		if !ok || len(ps) <= 1 {
			continue
		}
		texts, ok := ps[1].([]any)
		if !ok {
			continue
		}
		for _, t := range texts {
			if s, ok := t.(string); ok && s != "" {
				out = append(out, s)
			}
		}
	}
	return out
}

// indexSlice returns arr[i] as a []any if possible.
func indexSlice(arr []any, i int) ([]any, bool) {
	if i < 0 || i >= len(arr) {
		return nil, false
	}
	s, ok := arr[i].([]any)
	return s, ok
}

// indexString returns arr[i] as a string if possible.
func indexString(arr []any, i int) (string, bool) {
	if i < 0 || i >= len(arr) {
		return "", false
	}
	s, ok := arr[i].(string)
	return s, ok
}

// cleanGeminiText removes code-execution and card-content artifacts and trims
// surrounding whitespace.
func cleanGeminiText(text string) string {
	text = codeArtifactRe.ReplaceAllString(text, "")
	text = cardContentRe.ReplaceAllString(text, "")
	return strings.TrimSpace(text)
}

// ─── Cookie / SAPISIDHASH auth ───────────────────────────────────────────────

// loadCookie resolves the cookie string and its SAPISID (inline Cookie takes
// precedence over the cookie file). The SAPISID is parsed from the cookie string
// itself. An inline cookie is rotated in memory, so a rotated value (cached in
// cookieStr) is preferred over the original config value once available.
func (c *Client) loadCookie() (cookie, sapisid string) {
	if c.cfg.Cookie != "" {
		c.cookieMu.Lock()
		cached, cachedSapi := c.cookieStr, c.cookieSAPI
		c.cookieMu.Unlock()
		if cached != "" {
			return cached, cachedSapi
		}
		return c.cfg.Cookie, cookieValue(c.cfg.Cookie, cookieSAPISID)
	}
	if c.cfg.CookieFile == "" {
		return "", ""
	}
	return c.loadCookieFile()
}

// loadCookieFile reads the cookie file, caching the parsed result by mtime so
// the file is not re-read and re-parsed on every request.
func (c *Client) loadCookieFile() (cookie, sapisid string) {
	c.cookieMu.Lock()
	defer c.cookieMu.Unlock()

	info, err := os.Stat(c.cfg.CookieFile)
	if err != nil {
		c.log.Error("cookie stat failed", "path", c.cfg.CookieFile, "err", err)
		return c.cookieStr, c.cookieSAPI // fall back to last good value
	}
	if c.cookieLoaded && info.ModTime().Equal(c.cookieMtime) {
		return c.cookieStr, c.cookieSAPI
	}

	data, err := os.ReadFile(c.cfg.CookieFile)
	if err != nil {
		c.log.Error("cookie load failed", "path", c.cfg.CookieFile, "err", err)
		return c.cookieStr, c.cookieSAPI
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			cookie = line
			break
		}
	}
	if cookie != "" {
		sapisid = cookieValue(cookie, cookieSAPISID)
	}
	c.cookieStr, c.cookieSAPI = cookie, sapisid
	c.cookieMtime, c.cookieLoaded = info.ModTime(), true
	return cookie, sapisid
}

// makeSapisidHash builds the SAPISIDHASH Authorization header value.
func makeSapisidHash(sapisid string) string {
	ts := time.Now().Unix()
	sum := sha1.Sum(fmt.Appendf(nil, "%d %s %s", ts, sapisid, geminiBaseURL))
	return fmt.Sprintf("SAPISIDHASH %d_%x", ts, sum)
}
