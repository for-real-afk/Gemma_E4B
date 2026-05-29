# Gemma E4B — Backend

A production-grade FastAPI backend that serves as the brain of the Gemma E4B chat application. It connects a Vite/React frontend to a locally-running Ollama LLM, wraps every LLM call with latency measurement and cost calculation, maintains a two-layer conversation memory per session, and persists per-query analytics to a SQLite database — all without requiring any user login.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Feature Breakdown](#feature-breakdown)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Memory System](#memory-system)
- [LLM Wrapper and Cost Engine](#llm-wrapper-and-cost-engine)
- [Token Counting](#token-counting)

---

## What It Does

Gemma E4B is a self-hosted AI chat service. On every user message it:

1. Loads the session's conversation memory (recent verbatim turns + compressed summary of older turns)
2. Assembles the full prompt and forwards it to Ollama running locally
3. Measures wall-clock latency for the LLM call
4. Counts prompt and completion tokens using cached tokenizers (tiktoken or HuggingFace AutoTokenizer)
5. Calculates cost in both **USD and INR** based on per-model pricing
6. Returns the response, token counts, latency, and cost to the frontend in one JSON payload
7. Writes analytics to a SQLite database in a background task (zero added response latency)
8. Compresses old turns into a summary in a background task (also zero latency to the user)

No user registration or login is required. Anonymous identity is a UUID the browser generates once and sends with every request.

---

## Architecture

```
Browser  (React / Vite : 5173)
        |
        |  POST /chat        { message, session_id, user_id, model }
        |  POST /chat-file   { file, mssg, session_id, user_id, model }
        v
+-----------------------------------------------+
|              FastAPI  (main.py : 8000)         |
|                                                |
|  1. Memory manager                             |
|     add_turn_and_get_prompt()                  |
|     -> assembles: system + summary + turns     |
|                                                |
|  2. _call_ollama()  -> Ollama :11434           |
|     - time.perf_counter() latency wrap         |
|     - logs latency at INFO level               |
|                                                |
|  3. Token counting                             |
|     cached TokenCalculator (app.state)         |
|     tiktoken (gpt4) or AutoTokenizer (gemma)   |
|                                                |
|  4. compute_cost()                             |
|     returns { usd, inr }                       |
|                                                |
|  5. Return ChatResponse immediately            |
|     { response, usage, latency_ms, cost }      |
|                                                |
|  Background tasks (after response is sent):    |
|  - summarize_in_background()  -> Ollama        |
|  - _persist_query()           -> SQLite        |
+-----------------------------------------------+
        |
        v
   Ollama  (localhost:11434)
   Model: gemma3:4b  (configurable via .env)
```

---

## Feature Breakdown

### Two-Layer Memory System

Each browser session gets its own `SessionMemory` object managed in `memory/`:

- **Short-term layer** — the last 5 full turns (10 messages: user + assistant pairs) kept verbatim. Always sent to the LLM. Authoritative.
- **Summary layer** — when the short-term window overflows, older turns are evicted and compressed into a running plain-text summary by the LLM itself via `summarize_in_background`. The summary is injected as a low-priority system block on every subsequent call.
- Sessions expire after 1 hour of inactivity. A background async task prunes stale sessions every 10 minutes.
- Default store: in-memory dict with a thread lock. Redis is supported via `CACHE_TYPE=redis`.

### LLM Call Wrapper with Latency Logging

All three LLM callers share one internal function:

```python
async def _call_ollama(payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    # httpx POST to Ollama /api/chat
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("LLM latency=%.0f ms  model=%s", latency_ms, ...)
    return parsed_result, latency_ms
```

`call_llm` (used as the summarizer callback) discards the latency value so its signature stays `(str) -> dict` — the summarizer module does not need to know about timing.

### Dynamic Cost Calculation (USD + INR)

```
Model pricing:
  gemma  ->  input $0.10 / 1M tokens,  output $0.40 / 1M tokens
  gpt4   ->  input $2.50 / 1M tokens,  output $10.00 / 1M tokens

USD_TO_INR  configurable via .env  (default 85.0)
```

`compute_cost(prompt_tokens, completion_tokens, model)` returns `{"usd": ..., "inr": ...}`. Both appear in every `ChatResponse` and are stored in the analytics DB.

### Anonymous Identity and Persistent Analytics

The frontend generates a UUID once (`crypto.randomUUID()`) stored in localStorage as `gemma_user_id`. The backend uses it to:

1. Upsert a row in `user_profiles` — create on first visit, update `last_seen` on every call
2. Upsert a row in `chat_sessions` — create on first message, update `last_active`
3. Insert a row in `query_analytics` with full per-query metrics (tokens, latency, cost, attachment info)

All three writes happen in `_persist_query()`, a sync function registered as a FastAPI `BackgroundTask`. It runs **after** the HTTP response is sent, adding zero latency to the user.

### Multimodal File Handling

| File type | Processing |
|-----------|------------|
| Image (PNG, JPEG, GIF, WebP) | One-shot vision call to Ollama to extract text semantics. Raw image bytes are never stored — only the text description enters session memory. |
| PDF | Text extracted via PyPDF2, first 6 000 chars injected into the user message as `[PDF Content]`. |
| Audio | Rejected with a 400 error and a pointer to a future `/voice/transcribe` endpoint. |
| Video | Rejected with a 400 error. |
| > 10 MB | Rejected with a 413 error. |

---

## Tech Stack

| Component | Library / Version |
|-----------|-------------------|
| Web framework | FastAPI 0.109+ |
| ASGI server | Uvicorn 0.27+ |
| LLM transport | httpx (async) |
| PDF extraction | PyPDF2 3.0 |
| BPE tokenizer | tiktoken (cl100k_base) |
| Gemma tokenizer | HuggingFace `transformers` AutoTokenizer (falls back to gpt2) |
| Analytics DB | SQLite via SQLAlchemy 2.0 |
| Session store | In-memory dict (Redis optional) |
| Config | python-dotenv |

---

## Project Structure

```
Gemma_E4B/
|
|-- main.py                  # App entry point: endpoints, LLM callers,
|                            # cost engine, analytics background task
|-- requirements.txt
|-- .env                     # Not committed (see Environment Variables)
|-- gemma_chat.db            # SQLite analytics DB — auto-created on startup
|-- uploads/                 # Temp directory for uploaded files
|
|-- database/                # Analytics persistence layer
|   |-- __init__.py          # Re-exports: init_db, upsert_user, upsert_session,
|   |                        #             log_query, SessionLocal
|   |-- models.py            # SQLAlchemy ORM: UserProfile, ChatSession,
|   |                        #                 QueryAnalytic
|   +-- db.py                # Engine, SessionLocal, CRUD helpers
|
+-- memory/                  # Two-layer session memory system
    |-- __init__.py          # Re-exports public API
    |-- manager.py           # Turn management, prompt assembly, eviction logic
    |-- models.py            # Turn, SessionMemory, Role dataclasses
    |-- store.py             # Thread-safe in-memory store + Redis backend
    |-- summarizer.py        # LLM-based summary compression with truncation fallback
    +-- tokenizer.py         # TokenCalculator (tiktoken + AutoTokenizer)
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running with a model pulled:

```bash
ollama pull gemma3:4b
ollama serve          # starts Ollama on localhost:11434
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/for-real-afk/Gemma_E4B.git
cd Gemma_E4B

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
# Copy the template below and fill in your values
```

---

## Environment Variables

Create `.env` in the project root. Never commit this file.

```env
# --- Required ---
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# --- Optional ---
# Leave blank if Ollama does not require authentication
OLLAMA_API_KEY=

# Comma-separated list of allowed frontend origins
# Defaults to http://localhost:5173 and http://127.0.0.1:5173 when empty
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Server port (default 8000)
PORT=8000

# INR exchange rate used for cost display (default 85.0)
USD_TO_INR=85.0

# Session cache backend: "in_memory" (default) or "redis"
CACHE_TYPE=in_memory
REDIS_URL=redis://localhost:6379/0
```

---

## Running the Server

```bash
python main.py
```

Or with Uvicorn directly (useful for development with auto-reload):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**What happens on startup:**

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

Interactive API docs: `http://localhost:8000/docs`

---

## API Reference

### POST /chat

Memory-aware text chat.

**Request body (JSON):**

```json
{
  "session_id": "sess-1748539200000",
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Explain transformers in one paragraph",
  "model": "gemma"
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `session_id` | yes | — | Browser tab identifier from localStorage |
| `user_id` | no | `"anonymous"` | Anonymous UUID from localStorage |
| `message` | yes | — | User's text message |
| `model` | no | `"gemma"` | `"gemma"` (SentencePiece) or `"gpt4"` (tiktoken) |

**Response (200):**

```json
{
  "session_id": "sess-1748539200000",
  "response": "Transformers are neural network architectures...",
  "usage": {
    "prompt_tokens": 312,
    "completion_tokens": 89,
    "total_tokens": 401
  },
  "latency_ms": 1842.5,
  "cost": {
    "usd": 0.00006748,
    "inr": 0.00573580
  }
}
```

> `prompt_tokens` is the full assembled context (system prompt + summary block + all recent turns), not just the current message. This is what the LLM actually processes and what determines billing cost.

---

### POST /chat-file

Memory-aware multimodal chat. Send as `multipart/form-data`.

| Form field | Required | Description |
|-----------|----------|-------------|
| `file` | yes | PDF or image file (max 10 MB) |
| `session_id` | yes | Browser tab identifier |
| `user_id` | no | Anonymous UUID, default `"anonymous"` |
| `message` | no | User's text message (optional for images) |
| `model` | no | `"gemma"` or `"gpt4"` |

Response shape is identical to `/chat`.

---

### GET /health

```json
{
  "status": "ok",
  "model": "gemma3:4b",
  "active_sessions": 3,
  "cache_type": "in_memory"
}
```

---

### DELETE /session/{session_id}

Clears a session's conversation memory from the store.

```json
{ "deleted": true, "session_id": "sess-..." }
```

---

### GET /session/{session_id}/stats

Inspect a session's memory state. Useful for debugging.

```json
{
  "session_id": "sess-...",
  "total_turns": 12,
  "short_term_turns": 10,
  "summary_chars": 843,
  "has_summary": true
}
```

---

### POST /tokenize/count

Count tokens in an arbitrary text block using the globally cached tokenizer.

**Query parameters:** `text` (string), `model_type` (`gemma` or `gpt4`)

```json
{ "token_count": 47, "model_type": "gemma", "tokenizer": "gemma" }
```

---

## Database Schema

`gemma_chat.db` is a SQLite file created automatically on server startup via `Base.metadata.create_all()`. It is safe to restart the server — tables are created only if they do not already exist.

### user_profiles

Represents one anonymous browser identity.

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | TEXT PK | UUID from browser `crypto.randomUUID()` |
| `display_name` | TEXT | Auto-generated e.g. `User-A3F9B2` (last 6 chars of UUID) |
| `created_at` | DATETIME | First request timestamp |
| `last_seen` | DATETIME | Updated on every request |

### chat_sessions

One row per browser session (localStorage `gemma_session_id`).

| Column | Type | Notes |
|--------|------|-------|
| `session_id` | TEXT PK | From localStorage |
| `user_id` | TEXT FK | References `user_profiles.user_id` |
| `created_at` | DATETIME | Session start |
| `last_active` | DATETIME | Updated on every message |

### query_analytics

One row per LLM call. The core analytics table.

| Column | Type | Notes |
|--------|------|-------|
| `query_id` | TEXT PK | UUID (server-generated) |
| `session_id` | TEXT FK | References `chat_sessions` |
| `user_id` | TEXT FK | References `user_profiles` |
| `model_used` | TEXT | `"gemma"` or `"gpt4"` |
| `tokens_in` | INTEGER | Full prompt context token count |
| `tokens_out` | INTEGER | Completion token count |
| `tokens_attach` | INTEGER | Tokens from uploaded file content |
| `latency_ms` | REAL | Wall-clock LLM call time in milliseconds |
| `cost_usd` | REAL | USD cost for this query |
| `cost_inr` | REAL | INR cost (= cost_usd × USD_TO_INR) |
| `timestamp` | DATETIME | Query time (UTC) |
| `query_text` | TEXT | First 500 chars of the user's message |
| `has_attachment` | BOOLEAN | Whether a file was uploaded |
| `attachment_type` | TEXT | `"image"`, `"pdf"`, or NULL |

---

## Memory System

```
Synchronous path (hot, no I/O waits):
  add_turn_and_get_prompt()
    1. Load session from store
    2. Append user turn to short_term list
    3. Evict overflow (short_term > 10 msgs) -> overflow list
    4. Save session back to store
    5. Assemble and return messages list

Async await (LLM call):
  _call_ollama() -> (response_text, latency_ms)

Synchronous:
  record_assistant_reply()
    Append assistant turn, save session

Background tasks (after HTTP response sent):
  summarize_in_background()
    If overflow non-empty: LLM compresses overflow + existing summary
    -> new_summary saved back to session

  _persist_query()
    upsert_user, upsert_session, log_query -> SQLite
```

**Assembled prompt structure (order is fixed):**

```
1. System prompt (always)
2. [SUMMARY MEMORY] block (if summary exists) — marked "lower priority"
3. Recent verbatim turns (user + assistant, up to 10 messages)
```

---

## LLM Wrapper and Cost Engine

```python
# Single internal wrapper used by all three public callers:
async def _call_ollama(payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = await client.post(OLLAMA_HOST + "/api/chat", json=payload,
                          headers={"Authorization": f"Bearer {KEY}"} if KEY else {})
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("LLM latency=%.0f ms  model=%s", latency_ms, payload["model"])
    return _parse_ollama_response(r.text), latency_ms

# Cost helper:
def compute_cost(prompt_tokens, completion_tokens, model) -> dict[str, float]:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gemma"])
    usd = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
    return {"usd": round(usd, 8), "inr": round(usd * USD_TO_INR, 6)}
```

---

## Token Counting

`TokenCalculator` is instantiated once during the FastAPI lifespan event, stored in `app.state.token_calculator`, and reused on every request — no tokenizer reload overhead per call.

| `model` param | Tokenizer | Fallback chain |
|---------------|-----------|----------------|
| `"gemma"` | `alpindale/gemma-tokenizer` | `google/gemma-2b` → `gpt2` → char estimate |
| `"gpt4"` | tiktoken `cl100k_base` | char estimate (len ÷ 4) |

`count_messages_tokens(messages, model)` sums token counts across the full assembled prompt list — system prompt, summary block, and all recent turns — accurately reflecting the actual context window charged by the LLM.
