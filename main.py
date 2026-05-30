"""
main.py  —  Gemma4 Chat Service  v3.0.0
────────────────────────────────────────
Changes from v2.2.0:
  - Two-layer memory system (short-term + LLM summary)
  - session_id added to all chat request bodies
  - Multimodal input (images) converted to text semantics before memory storage
  - Background session pruner registered via FastAPI lifespan
  - /session/{session_id} DELETE  endpoint for explicit session reset
  - /session/{session_id}/stats   endpoint for debugging
"""

import logging
import os
import time
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
import asyncio
import base64
import io
import json

import httpx
import PyPDF2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from memory import (
    add_turn_and_get_prompt,
    active_session_count,
    prune_stale_sessions,
    record_assistant_reply,
    get_session_stats,
    summarize_in_background,
)
from memory.store import delete as delete_session
from database import init_db, upsert_user, upsert_session, log_query, save_pdf_chunks, load_pdf_chunks, SessionLocal


# ── env ───────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_HOST or not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_HOST or OLLAMA_MODEL in .env")

# OpenAI — optional. Required only when the frontend selects the gpt4 model.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # override with gpt-4o, gpt-4, etc.

# Ollama embedding model for semantic PDF search.
# Pull on your server with: ollama pull nomic-embed-text
# Falls back to keyword matching automatically if the model is not available.
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

MAX_FILE_SIZE_MB = 3

# ── per-session PDF chunk store ───────────────────────────────────────────────
# Maps session_id → list of text chunks from the last uploaded PDF.
# Kept in memory (lost on server restart — acceptable for demo).
# Allows follow-up questions to re-run RAG without re-uploading the file.
_session_pdfs: dict[str, list[str]] = {}

# ── pricing ───────────────────────────────────────────────────────────────────
USD_TO_INR = float(os.getenv("USD_TO_INR", "85.0"))

# GPT-5 Nano pricing (model: gpt-5-nano)
# Input: $1.10/1M tokens  |  Output: $4.40/1M tokens
# Update these if OpenAI changes the rates.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemma":    {"input": 0.10  / 1_000_000, "output": 0.40  / 1_000_000},
    "gpt4":     {"input": 1.10  / 1_000_000, "output": 4.40  / 1_000_000},  # gpt-5-nano rates
    "gpt5nano": {"input": 1.10  / 1_000_000, "output": 4.40  / 1_000_000},
    "openai":   {"input": 1.10  / 1_000_000, "output": 4.40  / 1_000_000},
}


def compute_cost(prompt_tokens: int, completion_tokens: int, model: str) -> dict[str, float]:
    pricing = _MODEL_PRICING.get(model.lower(), _MODEL_PRICING["gemma"])
    usd = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
    return {"usd": round(usd, 8), "inr": round(usd * USD_TO_INR, 6)}


# ── lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. DB — fast, must complete before serving
    await asyncio.to_thread(init_db)
    logger.info("Database initialised.")

    # 2. Session pruner
    pruner = asyncio.create_task(prune_stale_sessions())
    logger.info("Session pruner started.")

    # 3. TokenCalculator — place the instance immediately so requests never
    #    hit an AttributeError, then load tokenizers in the background.
    #    Primary token counts come from Ollama's response fields; the
    #    tokenizer is only a fallback, so a short warm-up delay is fine.
    from memory.tokenizer import TokenCalculator
    calculator = TokenCalculator()
    app.state.token_calculator = calculator
    tokenizer_task = asyncio.create_task(asyncio.to_thread(calculator.initialize))
    tokenizer_task.add_done_callback(
        lambda t: logger.info("TokenCalculator ready.") if not t.exception()
                  else logger.warning("TokenCalculator init failed (char-estimate fallback active): %s", t.exception())
    )
    logger.info("TokenCalculator initializing in background — server starting now.")

    yield

    pruner.cancel()
    tokenizer_task.cancel()
    logger.info("Session pruner stopped.")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Gemma4 Chat Service", version="3.0.0", lifespan=lifespan)
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "path": file_path
    }

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not origins:
    origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Allows any Vercel preview/production URL without manual config updates
    # e.g. gemma-e4b-abc123-user.vercel.app AND gemma-e4b.vercel.app
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique conversation ID")
    user_id:    str = Field(default="anonymous", description="Persistent anonymous user ID from localStorage")
    message:    str = Field(..., min_length=1, description="User message text")
    model:      str = Field(default="gemma", description="Tokenizer model: 'gemma' (SentencePiece) or 'gpt4' (BPE)")


class Usage(BaseModel):
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int


class Cost(BaseModel):
    usd: float
    inr: float


class ChatResponse(BaseModel):
    session_id:  str
    response:    str
    usage:       Usage
    latency_ms:  float
    cost:        Cost
    search_mode: str = "none"   # "semantic" | "keyword" | "none"


# ── low-level LLM callers ─────────────────────────────────────────────────────

async def _call_ollama(payload: dict) -> tuple[dict, float]:
    """
    Internal: POST payload to Ollama, measure wall-clock latency, log it.
    Returns (parsed_result, latency_ms).
    """
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"} if OLLAMA_KEY else {},
            )
        except Exception as e:
            raise HTTPException(502, f"LLM connection failed: {e}")

        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("LLM latency=%.0f ms  model=%s", latency_ms, payload.get("model", "?"))
    return _parse_ollama_response(r.text), latency_ms


async def call_llm(prompt: str) -> dict:
    """
    Text-only call used by the summarizer (background task).
    Signature must stay (str) -> dict — latency is discarded here.
    """
    result, _ = await _call_ollama({
        "model":    OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    })
    return result


async def call_llm_with_messages(messages: list[dict]) -> tuple[dict, float]:
    """
    Multi-turn call — primary path for all memory-aware requests.
    Returns ({"response": "<text>"}, latency_ms).
    """
    return await _call_ollama({
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
    })


async def call_llm_with_image(prompt: str, image_b64: str) -> tuple[dict, float]:
    """
    Vision call — one-shot image-to-text semantics extraction.
    Returns ({"response": "<text>"}, latency_ms).
    """
    return await _call_ollama({
        "model":    OLLAMA_MODEL,
        "messages": [
            {
                "role":    "user",
                "content": prompt,
                "images":  [image_b64],   # base64, no data URI prefix
            }
        ],
        "stream": False,
    })


async def call_openai_with_messages(messages: list[dict]) -> tuple[dict, float]:
    """
    OpenAI Chat Completions API call.
    Returns ({"response", "prompt_tokens", "completion_tokens"}, latency_ms).
    Raises 503 if OPENAI_API_KEY is not set.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            503,
            "OpenAI API key not configured. Set OPENAI_API_KEY in your environment variables.",
        )

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": OPENAI_MODEL, "messages": messages},
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type":  "application/json",
                },
            )
        except Exception as e:
            raise HTTPException(502, f"OpenAI connection failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"OpenAI error: {r.text}")

    data    = r.json()
    content = data["choices"][0]["message"]["content"]
    usage   = data.get("usage", {})

    logger.info(
        "OpenAI latency=%.0f ms  model=%s  tokens=%d+%d",
        latency_ms, OPENAI_MODEL,
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
    )

    return {
        "response":          content,
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }, latency_ms


def _is_openai_model(model: str) -> bool:
    return model.lower().strip() in ("gpt4", "gpt-4", "gpt", "gpt5nano", "gpt-5-nano", "gpt5", "openai")


def _parse_ollama_response(raw: str) -> dict:
    try:
        lines  = [l for l in raw.strip().splitlines() if l.strip()]
        parsed = [json.loads(l) for l in lines]
        content = "".join(p.get("message", {}).get("content", "") for p in parsed)

        final             = parsed[-1] if parsed else {}
        prompt_eval_count = final.get("prompt_eval_count")
        eval_count        = final.get("eval_count")

        # Debug log — shows exactly what Ollama reports vs visible text length
        logger.info(
            "[TOKEN DEBUG] visible_chars=%d  prompt_eval_count=%s  eval_count=%s  objects_in_response=%d",
            len(content), prompt_eval_count, eval_count, len(parsed),
        )

        return {
            "response":          content,
            "prompt_tokens":     prompt_eval_count,
            "completion_tokens": eval_count,
        }
    except Exception:
        raise HTTPException(500, "Invalid response from LLM")


# ── PDF chunk DB helpers (sync — called via asyncio.to_thread) ───────────────

def _db_load_pdf_chunks(session_id: str) -> list[str] | None:
    db = SessionLocal()
    try:
        return load_pdf_chunks(db, session_id)
    except Exception as exc:
        logger.error("Failed to load PDF chunks from DB: %s", exc)
        return None
    finally:
        db.close()


# ── analytics background writer ───────────────────────────────────────────────

def _persist_query(
    user_id:         str,
    session_id:      str,
    model_used:      str,
    tokens_in:       int,
    tokens_out:      int,
    tokens_attach:   int,
    latency_ms:      float,
    cost_usd:        float,
    cost_inr:        float,
    query_text:      str | None,
    has_attachment:  bool,
    attachment_type: str | None,
    pdf_chunks:      list[str] | None = None,
) -> None:
    """
    Sync DB writer — FastAPI runs sync background tasks in a thread pool,
    so this never blocks the event loop.

    pdf_chunks is saved AFTER upsert_session so the session row exists.
    This is the correct order — attempting to save chunks before the
    session row is created causes a foreign-key/not-found error.
    """
    db = SessionLocal()
    try:
        upsert_user(db, user_id)
        upsert_session(db, session_id, user_id)
        if pdf_chunks:
            save_pdf_chunks(db, session_id, pdf_chunks)
        log_query(
            db,
            session_id      = session_id,
            user_id         = user_id,
            model_used      = model_used,
            tokens_in       = tokens_in,
            tokens_out      = tokens_out,
            tokens_attach   = tokens_attach,
            latency_ms      = latency_ms,
            cost_usd        = cost_usd,
            cost_inr        = cost_inr,
            query_text      = query_text,
            has_attachment  = has_attachment,
            attachment_type = attachment_type,
        )
    except Exception as exc:
        logger.error("DB persist_query failed: %s", exc)
        db.rollback()
    finally:
        db.close()


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Memory-aware text chat.

    Flow:
      1. Append user turn → assemble prompt  (sync, instant)
      2. Call LLM with full message history
      3. Count tokens via routed tokenizer (gemma→SentencePiece, gpt4→BPE)
      4. Store assistant reply
      5. Summarize overflow in background    (after response is sent)
    """
    from memory.tokenizer import TokenCalculator
    from memory.pdf_rag import retrieve_relevant_chunks
    from memory.embeddings import embed_query, embed_texts
    from memory.vector_store import get_store

    # Re-run RAG for every follow-up question using the stored PDF from this session.
    # Prefer semantic vector search; fall back to keyword matching if embeddings
    # are not available (nomic-embed-text not pulled on Ollama server).
    pdf_context   = ""
    tokens_attach = 0
    search_mode   = "none"

    # If this session has no chunks in memory (e.g. server restarted),
    # try loading them from Supabase and rebuilding the in-memory store.
    if req.session_id not in _session_pdfs:
        db_chunks = await asyncio.to_thread(_db_load_pdf_chunks, req.session_id)
        if db_chunks:
            _session_pdfs[req.session_id] = db_chunks
            logger.info(
                "PDF chunks restored from DB for session %s (%d chunks)",
                req.session_id, len(db_chunks),
            )
            # Re-embed restored chunks so semantic search is available again
            embeddings = await embed_texts(db_chunks, OLLAMA_HOST, OLLAMA_EMBED_MODEL, OLLAMA_KEY or "")
            if embeddings:
                from memory.vector_store import VectorStore, set_store as _set_store
                store = VectorStore()
                store.add(db_chunks, embeddings)
                _set_store(req.session_id, store)

    vector_store = get_store(req.session_id)
    if vector_store and vector_store.ready:
        q_embedding = await embed_query(req.message, OLLAMA_HOST, OLLAMA_EMBED_MODEL, OLLAMA_KEY or "")
        if q_embedding:
            pdf_context = vector_store.search(q_embedding, max_tokens=2000)
            search_mode = "semantic"

    if not pdf_context and req.session_id in _session_pdfs:
        pdf_context = retrieve_relevant_chunks(
            _session_pdfs[req.session_id],
            query      = req.message,
            max_tokens = 2000,
        )
        if pdf_context:
            search_mode = "keyword"

    if search_mode != "none":
        logger.info(
            "[PDF RAG] session=%s  mode=%-8s  context=%d chars (~%d tokens)",
            req.session_id, search_mode, len(pdf_context), len(pdf_context) // 4,
        )
    else:
        logger.debug("[PDF RAG] session=%s  no PDF in session", req.session_id)

    user_message = (
        f"{req.message}\n\n[PDF Context from uploaded document]\n{pdf_context}"
        if pdf_context else req.message
    )

    messages, overflow, old_summary = add_turn_and_get_prompt(
        session_id   = req.session_id,
        user_message = user_message,
    )

    if _is_openai_model(req.model):
        if not OPENAI_API_KEY:
            logger.warning("OpenAI model '%s' requested but OPENAI_API_KEY not set — falling back to Gemma.", req.model)
            result, latency_ms = await call_llm_with_messages(messages)
        else:
            logger.info("Routing to OpenAI model=%s", OPENAI_MODEL)
            result, latency_ms = await call_openai_with_messages(messages)
    else:
        result, latency_ms = await call_llm_with_messages(messages)

    reply = result["response"]

    record_assistant_reply(req.session_id, reply)

    calculator        = app.state.token_calculator
    tokenizer_type    = TokenCalculator.route_tokenizer(req.model)
    prompt_tokens     = result.get("prompt_tokens")     or calculator.count_messages_tokens(messages, req.model)
    completion_tokens = result.get("completion_tokens") or calculator.count_tokens(reply, tokenizer_type)
    if pdf_context:
        tokens_attach = calculator.count_tokens(pdf_context, tokenizer_type)

    cost = compute_cost(prompt_tokens, completion_tokens, req.model)

    background_tasks.add_task(
        summarize_in_background,
        req.session_id, overflow, old_summary, call_llm,
    )
    background_tasks.add_task(
        _persist_query,
        req.user_id, req.session_id, req.model,
        prompt_tokens, completion_tokens, tokens_attach,
        latency_ms, cost["usd"], cost["inr"],
        req.message[:500], bool(pdf_context), "pdf" if pdf_context else None,
    )

    return ChatResponse(
        session_id  = req.session_id,
        response    = reply,
        latency_ms  = round(latency_ms, 2),
        search_mode = search_mode,
        usage       = Usage(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
        ),
        cost = Cost(**cost),
    )


@app.post("/chat-file", response_model=ChatResponse)
async def chat_file(
    background_tasks: BackgroundTasks,
    session_id: str        = Form(...),
    user_id:    str        = Form("anonymous"),
    message:    str        = Form(""),
    model:      str        = Form("gemma"),
    file:       UploadFile = File(...),
):
    """
    Memory-aware multimodal chat (image or PDF).

    Multimodal contract:
      - Images  → one-shot vision call → text description stored in memory
      - PDFs    → text extracted by PyPDF2 → injected as context in user message
      - Audio / Video → rejected with a clear error
    """
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE_MB} MB)")

    content_type = file.content_type or ""

    latency_ms      = 0.0
    attach_text     = ""
    attachment_type = None

    try:
        # ── IMAGE ─────────────────────────────────────────────────────────────
        if content_type.startswith("image"):
            image_b64       = base64.b64encode(file_bytes).decode("utf-8")
            attachment_type = "image"

            # Step 1: one-shot vision pass to extract semantics
            vision_prompt = (
                message.strip()
                or "Describe this image in detail, capturing all key facts, "
                   "decisions, and structural information."
            )
            vision_result, _ = await call_llm_with_image(vision_prompt, image_b64)
            description       = vision_result["response"]
            attach_text       = description   # track for token counting

            # Step 2: store description as media_description (image itself is discarded)
            #   ❌  "User shared an image"
            #   ✅  actual extracted content
            messages, overflow, old_summary = add_turn_and_get_prompt(
                session_id        = session_id,
                user_message      = message.strip() or "(image uploaded)",
                media_description = description,
            )

            # Step 3: get conversational reply using the full memory context
            result, latency_ms = await call_llm_with_messages(messages)
            reply               = result["response"]

        # ── PDF ───────────────────────────────────────────────────────────────
        elif content_type == "application/pdf":
            from memory.pdf_rag import chunk_text, retrieve_relevant_chunks

            attachment_type = "pdf"
            reader          = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            extracted       = "".join(page.extract_text() or "" for page in reader.pages)

            if not extracted.strip():
                raise HTTPException(422, "Could not extract text from PDF.")

            # Chunk the full PDF and retrieve only the sections most relevant
            # to the user's query, capped at 3000 tokens of context.
            chunks      = chunk_text(extracted)
            attach_text = retrieve_relevant_chunks(
                chunks,
                query     = message or extracted[:200],
                max_tokens = 3000,
            )

            # Store in memory for fast access this session
            _session_pdfs[session_id] = chunks
            # chunks are saved to DB inside _persist_query (after upsert_session)

            # Try to embed all chunks for semantic search
            from memory.embeddings import embed_texts
            from memory.vector_store import VectorStore, set_store

            embeddings = await embed_texts(chunks, OLLAMA_HOST, OLLAMA_EMBED_MODEL, OLLAMA_KEY or "")
            if embeddings:
                store = VectorStore()
                store.add(chunks, embeddings)
                set_store(session_id, store)
                logger.info(
                    "PDF: %d chunks embedded (semantic search active) for session %s",
                    len(chunks), session_id,
                )
            else:
                logger.info(
                    "PDF: %d chunks stored (keyword fallback, embed model unavailable) for session %s",
                    len(chunks), session_id,
                )

            logger.info(
                "PDF RAG: retrieved %d chars (~%d tokens) for initial query",
                len(attach_text), len(attach_text) // 4,
            )

            # Inject retrieved sections into the user message
            combined_message = (
                f"{message}\n\n[PDF Content]\n{attach_text}".strip()
                if message.strip()
                else f"[PDF Content]\n{attach_text}"
            )

            messages, overflow, old_summary = add_turn_and_get_prompt(
                session_id   = session_id,
                user_message = combined_message,
            )

            if _is_openai_model(model):
                result, latency_ms = await call_openai_with_messages(messages)
            else:
                result, latency_ms = await call_llm_with_messages(messages)
            reply = result["response"]

        # ── AUDIO ─────────────────────────────────────────────────────────────
        elif content_type.startswith("audio"):
            raise HTTPException(
                400,
                "Audio transcription not supported. Use /voice/transcribe endpoint.",
            )

        # ── VIDEO ─────────────────────────────────────────────────────────────
        elif content_type.startswith("video"):
            raise HTTPException(400, "Video processing not supported yet.")

        else:
            raise HTTPException(400, f"Unsupported file type: {content_type}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"File processing failed: {e}")

    record_assistant_reply(session_id, reply)

    from memory.tokenizer import TokenCalculator
    calculator        = app.state.token_calculator
    tokenizer_type    = TokenCalculator.route_tokenizer(model)
    prompt_tokens     = result.get("prompt_tokens")     or calculator.count_messages_tokens(messages, model)
    completion_tokens = result.get("completion_tokens") or calculator.count_tokens(reply, tokenizer_type)
    tokens_attach     = calculator.count_tokens(attach_text, tokenizer_type) if attach_text else 0
    cost              = compute_cost(prompt_tokens, completion_tokens, model)

    background_tasks.add_task(
        summarize_in_background,
        session_id, overflow, old_summary, call_llm,
    )
    background_tasks.add_task(
        _persist_query,
        user_id, session_id, model,
        prompt_tokens, completion_tokens, tokens_attach,
        latency_ms, cost["usd"], cost["inr"],
        message[:500], True, attachment_type,
        _session_pdfs.get(session_id),   # saved to DB after upsert_session
    )

    return ChatResponse(
        session_id  = session_id,
        response    = reply,
        latency_ms  = round(latency_ms, 2),
        search_mode = "keyword",   # first upload always uses retrieved text directly
        usage       = Usage(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
        ),
        cost = Cost(**cost),
    )


# ── session management endpoints ──────────────────────────────────────────────

@app.delete("/session/{session_id}", summary="Clear a session's memory")
def clear_session(session_id: str):
    delete_session(session_id)
    _session_pdfs.pop(session_id, None)
    from memory.vector_store import delete_store
    delete_store(session_id)
    return {"deleted": True, "session_id": session_id}


@app.get("/session/{session_id}/stats", summary="Inspect session memory state")
def session_stats(session_id: str):
    return get_session_stats(session_id)


# ── health / root ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from memory.store import CACHE_TYPE
    return {
        "status":           "ok",
        "model":            OLLAMA_MODEL,
        "active_sessions":  active_session_count(),
        "cache_type":       CACHE_TYPE,
    }


@app.post("/tokenize/count", summary="Count tokens in a text block")
def count_tokens(text: str, model_type: str = "gemma"):
    """
    Utility endpoint — count tokens using the globally cached TokenCalculator.
    `model_type` accepts UI values ('gemma', 'gpt4') or internal aliases ('openai', 'tiktoken').
    """
    from memory.tokenizer import TokenCalculator

    if not hasattr(app.state, "token_calculator"):
        raise HTTPException(status_code=503, detail="TokenCalculator not initialized yet")

    routed = TokenCalculator.route_tokenizer(model_type)
    count  = app.state.token_calculator.count_tokens(text, routed)
    return {"token_count": count, "model_type": model_type, "tokenizer": routed}


@app.get("/")
def root():
    return {"service": "Gemma4 Chat Service", "version": "3.0.0", "docs": "/docs"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "path": file_path
    }
# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)