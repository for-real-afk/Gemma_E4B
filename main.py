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
from database import init_db, upsert_user, upsert_session, log_query, SessionLocal


# ── env ───────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_HOST or not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_HOST or OLLAMA_MODEL in .env")

MAX_FILE_SIZE_MB = 10

# ── pricing ───────────────────────────────────────────────────────────────────
USD_TO_INR = float(os.getenv("USD_TO_INR", "85.0"))

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemma": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gpt4":  {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
}


def compute_cost(prompt_tokens: int, completion_tokens: int, model: str) -> dict[str, float]:
    pricing = _MODEL_PRICING.get(model.lower(), _MODEL_PRICING["gemma"])
    usd = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
    return {"usd": round(usd, 8), "inr": round(usd * USD_TO_INR, 6)}


# ── lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background pruner on startup; initialize tokenizers; cancel cleanly on shutdown."""
    await asyncio.to_thread(init_db)
    logger.info("Database initialised.")

    pruner = asyncio.create_task(prune_stale_sessions())
    logger.info("Session pruner started.")
    
    # Initialize and cache TokenCalculator globally in app.state
    from memory.tokenizer import TokenCalculator
    calculator = TokenCalculator()
    logger.info("Initializing TokenCalculator (loading/caching tokenizers)...")
    await asyncio.to_thread(calculator.initialize)
    app.state.token_calculator = calculator
    logger.info("TokenCalculator initialization complete.")
    
    yield
    pruner.cancel()
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
    session_id: str
    response:   str
    usage:      Usage
    latency_ms: float
    cost:       Cost


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


def _parse_ollama_response(raw: str) -> dict:
    try:
        lines   = [l for l in raw.strip().splitlines() if l.strip()]
        parsed  = [json.loads(l) for l in lines]
        content = "".join(p.get("message", {}).get("content", "") for p in parsed)
        return {"response": content}
    except Exception:
        raise HTTPException(500, "Invalid response from LLM")


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
) -> None:
    """
    Sync DB writer — FastAPI runs sync background tasks in a thread pool,
    so this never blocks the event loop.
    """
    db = SessionLocal()
    try:
        upsert_user(db, user_id)
        upsert_session(db, session_id, user_id)
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

    messages, overflow, old_summary = add_turn_and_get_prompt(
        session_id   = req.session_id,
        user_message = req.message,
    )

    result, latency_ms = await call_llm_with_messages(messages)
    reply              = result["response"]

    record_assistant_reply(req.session_id, reply)

    calculator        = app.state.token_calculator
    tokenizer_type    = TokenCalculator.route_tokenizer(req.model)
    prompt_tokens     = calculator.count_messages_tokens(messages, req.model)
    completion_tokens = calculator.count_tokens(reply, tokenizer_type)

    cost = compute_cost(prompt_tokens, completion_tokens, req.model)

    background_tasks.add_task(
        summarize_in_background,
        req.session_id, overflow, old_summary, call_llm,
    )
    background_tasks.add_task(
        _persist_query,
        req.user_id, req.session_id, req.model,
        prompt_tokens, completion_tokens, 0,
        latency_ms, cost["usd"], cost["inr"],
        req.message[:500], False, None,
    )

    return ChatResponse(
        session_id = req.session_id,
        response   = reply,
        latency_ms = round(latency_ms, 2),
        usage      = Usage(
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
            attachment_type = "pdf"
            reader          = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            extracted       = "".join(page.extract_text() or "" for page in reader.pages)

            if not extracted.strip():
                raise HTTPException(422, "Could not extract text from PDF.")

            attach_text = extracted[:6000]   # track for token counting

            # Inject PDF text directly into the user message; treat as text turn
            combined_message = (
                f"{message}\n\n[PDF Content]\n{attach_text}".strip()
                if message.strip()
                else f"[PDF Content]\n{attach_text}"
            )

            messages, overflow, old_summary = add_turn_and_get_prompt(
                session_id   = session_id,
                user_message = combined_message,
            )

            result, latency_ms = await call_llm_with_messages(messages)
            reply               = result["response"]

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
    prompt_tokens     = calculator.count_messages_tokens(messages, model)
    completion_tokens = calculator.count_tokens(reply, tokenizer_type)
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
    )

    return ChatResponse(
        session_id = session_id,
        response   = reply,
        latency_ms = round(latency_ms, 2),
        usage      = Usage(
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