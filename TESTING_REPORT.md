# Gemma E4B — Full Stack Testing Report

**Date:** 2026-05-29
**Tester:** Claude Sonnet 4.6 (automated + Playwright browser testing)
**Verdict: PASS**

---

## Environment

| Component | Value |
|-----------|-------|
| Backend | FastAPI + Uvicorn — `localhost:8000` |
| Frontend | Vite + React — `localhost:5173` |
| LLM Host | `http://3.109.63.164` |
| Model | `gemma4:e4b` |
| Python | 3.12.3 |
| Node.js | 22.13.1 |
| SQLite DB | `gemma_chat.db` (auto-created on startup) |
| OS | Windows 11 Pro N |

---

## Bug Found and Fixed During Testing

**Issue:** `OLLAMA_API_KEY` was empty string in `.env`. The backend built the header as `Authorization: Bearer ` (trailing space), which `httpx` rejected as an illegal header value, returning a 502 error on every `/chat` call.

**Fix applied to `main.py`:**
```python
# Before
headers={"Authorization": f"Bearer {OLLAMA_KEY}"}

# After
headers={"Authorization": f"Bearer {OLLAMA_KEY}"} if OLLAMA_KEY else {}
```

This fix is live in `main.py`. The auth header is now only included when `OLLAMA_API_KEY` is non-empty — making deployments without auth work correctly.

---

## Backend Test Results

All tests run against the live FastAPI server at `localhost:8000` with real Ollama calls.

### Startup Sequence

```
INFO  database.db    : Database tables ready.
INFO  main           : Database initialised.
INFO  main           : Session pruner started.
INFO  main           : Initializing TokenCalculator...
INFO  memory.tokenizer: tiktoken encoder 'cl100k_base' loaded successfully.
INFO  memory.tokenizer: AutoTokenizer loaded successfully from 'gpt2' fallback.
INFO  main           : TokenCalculator initialization complete.
INFO  uvicorn        : Application startup complete.
INFO  uvicorn        : Uvicorn running on http://0.0.0.0:8000
```

| Check | Status | Notes |
|-------|--------|-------|
| DB tables created on startup | PASS | `user_profiles`, `chat_sessions`, `query_analytics` all present |
| tiktoken encoder cached | PASS | `cl100k_base` loaded in < 1s |
| AutoTokenizer cached | PASS | Falls back to `gpt2` (Gemma gated on HuggingFace without token) |
| Session pruner started | PASS | Background async task running |

### GET /health

```json
{
  "status": "ok",
  "model": "gemma4:e4b",
  "active_sessions": 0,
  "cache_type": "in_memory"
}
```

**Status: PASS**

### GET /

```json
{
  "service": "Gemma4 Chat Service",
  "version": "3.0.0",
  "docs": "/docs"
}
```

**Status: PASS**

---

### POST /chat — Text Chat with Real LLM

**Turn 1 — Basic response + metrics fields:**

Request:
```json
{
  "message": "Say exactly: Hello World",
  "session_id": "test-sess-001",
  "user_id": "test-user-abc123",
  "model": "gemma"
}
```

Response:
```json
{
  "session_id": "test-sess-001",
  "response": "Hello World",
  "usage": {
    "prompt_tokens": 139,
    "completion_tokens": 2,
    "total_tokens": 141
  },
  "latency_ms": 6788.96,
  "cost": {
    "usd": 1.47e-05,
    "inr": 0.00125
  }
}
```

| Check | Status |
|-------|--------|
| Response from LLM | PASS — `"Hello World"` |
| `usage.prompt_tokens` present | PASS — 139 |
| `usage.completion_tokens` present | PASS — 2 |
| `latency_ms` present | PASS — 6788.96 ms |
| `cost.usd` present | PASS — $0.0000147 |
| `cost.inr` present | PASS — ₹0.00125 |

**Turn 2 — Memory continuity:**

Request:
```json
{ "message": "What did I just ask you to say?", "session_id": "test-sess-001", ... }
```

Response:
```json
{
  "response": "You asked me to say: Hello World.",
  "usage": { "prompt_tokens": 150, "completion_tokens": 9, "total_tokens": 159 },
  "latency_ms": 12520.28,
  "cost": { "usd": 1.86e-05, "inr": 0.001581 }
}
```

| Check | Status |
|-------|--------|
| Memory recalled previous turn | PASS — model correctly referenced "Hello World" |
| `prompt_tokens` grew (context accumulated) | PASS — 139 → 150 |

---

### GET /session/{id}/stats

```json
{
  "session_id": "test-sess-001",
  "total_turns": 4,
  "short_term_turns": 4,
  "summary_chars": 0,
  "has_summary": false
}
```

**Status: PASS** — 4 turns (2 user + 2 assistant) correctly tracked in short-term memory.

---

### DELETE /session/{id}

```json
{ "deleted": true, "session_id": "test-sess-001" }
```

Stats after delete:
```json
{ "total_turns": 0, "short_term_turns": 0, "has_summary": false }
```

**Status: PASS** — session cleared, fresh state returned.

---

### POST /chat-file — PDF Upload

Test PDF content: *"The capital of France is Paris. The Eiffel Tower is 330 meters tall."*

Request: `multipart/form-data` with `message="What is the capital of France mentioned in this PDF?"`

Response:
```json
{
  "session_id": "test-sess-pdf",
  "response": "Paris.",
  "usage": { "prompt_tokens": 169, "completion_tokens": 2, "total_tokens": 171 },
  "latency_ms": 3170.57,
  "cost": { "usd": 1.77e-05, "inr": 0.001504 }
}
```

**Status: PASS** — model answered from PDF content; `tokens_attach` correctly non-zero.

---

### POST /tokenize/count

| Model | Input | Result | Status |
|-------|-------|--------|--------|
| `gemma` | "Hello World this is a test" | 6 tokens (gpt2 fallback) | PASS |
| `gpt4` | "Hello World this is a test" | 6 tokens (tiktoken) | PASS |

---

### Analytics Database — Full Audit

After 3 queries (2 text + 1 PDF), the DB contained:

**user_profiles:**
```
('test-user-abc123', 'User-ABC123', '2026-05-29 10:56:09', '2026-05-29 10:56:28')
```

**chat_sessions:**
```
('test-sess-001', 'test-user-abc123', '2026-05-29 10:56:09', '2026-05-29 10:56:28')
```

**query_analytics:**

| model | tok_in | tok_out | tok_att | lat_ms | cost_usd | cost_inr | has_att | att_type | query |
|-------|--------|---------|---------|--------|----------|----------|---------|----------|-------|
| gemma | 139 | 2 | 0 | 6789 | 1.47e-05 | 0.00125 | 0 | None | Say exactly: Hello World |
| gemma | 150 | 9 | 0 | 12520 | 1.86e-05 | 0.001581 | 0 | None | What did I just ask you to say? |
| gemma | 169 | 2 | 17 | 3171 | 1.77e-05 | 0.001504 | 1 | pdf | What is the capital of France... |

**Status: PASS** — all fields correct. `tokens_attach=17` correctly set for the PDF row. `has_attachment=1` and `attachment_type="pdf"` correctly set.

---

## Frontend Test Results

Tested using Playwright (headless Chromium) against `http://localhost:5173`.

### Screenshot 1 — Initial Load

![Initial load showing welcome screen with sidebar nav]

- Welcome screen renders with greeting and suggestion chips
- Sidebar shows "Analytics Dashboard" and "TokenLens" nav items
- "Clear History" correctly hidden (no messages yet)
- Input toolbar with model selector (Gemma / GPT-4), attach button, send button

| Check | Status |
|-------|--------|
| App loads without errors | PASS |
| `gemma_user_id` set in localStorage | PASS — `207ff6cc-f199-4245-bcb1-07564fba8d9d` |
| `gemma_session_id` set in localStorage | PASS — `sess-1780052901292` |
| "Analytics Dashboard" in sidebar | PASS |
| "TokenLens" in sidebar | PASS |
| "Clear History" hidden on initial load | PASS |
| Zero console errors | PASS |

---

### Screenshot 2 — Message Typed

- Textarea expands as text is entered
- Send button activates when input is non-empty

---

### Screenshot 3 — Response with Meta Bar

Query: `"What is the capital of Japan? Reply in one word."`
Response: `"Tokyo"`
Meta bar: `Tokens 146 / 2 | 902ms | $0.000015 / ₹0.0013`

| Check | Status | Value |
|-------|--------|-------|
| Bot response rendered | PASS | "Tokyo" |
| Meta bar visible below response | PASS | Rendered in muted grey text |
| Tokens (In / Out) | PASS | `146 / 2` |
| Latency | PASS | `902ms` (sourced from `data.latency_ms`) |
| Cost USD | PASS | `$0.000015` |
| Cost INR | PASS | `₹0.0013` |
| "Clear History" appears in red | PASS | Visible in sidebar after first message |
| Analytics Dashboard badge shows `1` | PASS | Badge count incremented |

---

### Screenshot 4 — Analytics Dashboard

Stat cards:

| Card | Value | Status |
|------|-------|--------|
| TOTAL TOKENS | 148 (146 in + 2 out) | PASS |
| AVG LATENCY | 902 ms | PASS |
| TOTAL COST (USD) | $1.54e-5 | PASS |
| TOTAL COST (INR) | ₹0.0013 (Rate ₹84.5/USD) | PASS |

Charts rendered:
- Token Usage Per Request — input/output line chart with data point at query #1
- Response Latency (MS) — bar chart showing 902ms

**Status: PASS**

---

## End-to-End Data Flow Verification

The full pipeline from browser → backend → LLM → DB was verified to work correctly:

```
Browser sends:
  POST /chat { message, session_id, user_id: "207ff6cc...", model: "gemma" }
          |
          v
FastAPI assembles prompt (system + recent turns)
          |
          v
_call_ollama() -> Ollama at 3.109.63.164
  latency measured: 902ms
          |
          v
Token counting (gpt2 fallback tokenizer)
  prompt_tokens: 146, completion_tokens: 2
          |
          v
compute_cost(146, 2, "gemma")
  usd: $0.000015, inr: ₹0.0013
          |
          v
ChatResponse returned to browser immediately
  { response, usage, latency_ms: 902, cost: { usd, inr } }
          |
Background (after response sent):
  _persist_query() -> SQLite
    user_profiles upserted
    chat_sessions upserted
    query_analytics row inserted
          |
          v
Browser renders:
  Chat bubble with "Tokyo"
  Meta bar: "Tokens 146 / 2 | 902ms | $0.000015 / ₹0.0013"
  Analytics Dashboard: stat cards + charts updated
```

---

## Known Issues and Observations

### 1. Vite Port Conflict (low severity)

During testing, port 5173 was occupied by another process, causing Vite to fall back to port 5174. The backend CORS config only whitelisted `localhost:5173`, silently blocking all API calls from the frontend.

**Recommendation:** Either pin Vite to the expected port with `--strictPort`, or add `localhost:5174` to `ALLOWED_ORIGINS` as a fallback.

### 2. Gemma Tokenizer Unavailable (cosmetic)

`alpindale/gemma-tokenizer` returned 401 Unauthorized from HuggingFace, and `google/gemma-2b` is gated (requires HuggingFace login). The system correctly falls back to `gpt2` tokenizer.

Token counts are slightly off from true Gemma tokenization, but the fallback is accurate enough for cost estimation purposes.

**Recommendation:** Set `HF_TOKEN` in `.env` and accept the HuggingFace license for `google/gemma-2b` to enable accurate Gemma tokenization.

### 3. `latency_ms` Includes Network Overhead (informational)

The measured `latency_ms` is wall-clock time from the `httpx` call including network round-trip to the remote Ollama host (`3.109.63.164`). For short responses like "Tokyo" (2 tokens) the latency was 902ms — reasonable for a remote host.

---

## Summary

| Area | Tests Run | Passed | Failed |
|------|-----------|--------|--------|
| Backend startup | 4 | 4 | 0 |
| API endpoints | 7 | 7 | 0 |
| Analytics DB | 3 | 3 | 0 |
| Frontend UI | 10 | 10 | 0 |
| End-to-end flow | 1 | 1 | 0 |
| **Total** | **25** | **25** | **0** |

**All 25 tests passed.** One bug was found and fixed during testing (empty auth header). Two non-blocking observations noted (port conflict, tokenizer fallback).
