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

from contextlib import asynccontextmanager
import asyncio
import base64
import io
import json
import logging
import os

import httpx
import PyPDF2
from dotenv import load_dotenv
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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── env ───────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_HOST or not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_HOST or OLLAMA_MODEL in .env")

MAX_FILE_SIZE_MB = 5


# ── lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background pruner on startup; cancel cleanly on shutdown."""
    pruner = asyncio.create_task(prune_stale_sessions())
    logger.info("Session pruner started.")
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
    session_id: str   = Field(..., min_length=1, description="Unique conversation ID")
    message:    str   = Field(..., min_length=1, description="User message text")


class ChatResponse(BaseModel):
    session_id: str
    response:   str


# ── low-level LLM callers ─────────────────────────────────────────────────────

async def call_llm(prompt: str) -> dict:
    """
    Text-only call (used by the summarizer and plain /chat fallback path).
    Returns {"response": "<text>"}.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                },
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
            )
        except Exception as e:
            raise HTTPException(502, f"LLM connection failed: {e}")

        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)

        return _parse_ollama_response(r.text)


async def call_llm_with_messages(messages: list[dict]) -> dict:
    """
    Multi-turn call — accepts the full assembled message list from the memory manager.
    This is the primary path for all memory-aware requests.
    Returns {"response": "<text>"}.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": messages,
                    "stream":   False,
                },
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
            )
        except Exception as e:
            raise HTTPException(502, f"LLM connection failed: {e}")

        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)

        return _parse_ollama_response(r.text)


async def call_llm_with_image(prompt: str, image_b64: str) -> dict:
    """
    Vision call — used ONCE per image to extract text semantics.
    The raw image is never stored; only the returned description enters memory.
    Returns {"response": "<text>"}.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role":    "user",
                            "content": prompt,
                            "images":  [image_b64],   # base64, no data URI prefix
                        }
                    ],
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
            )
        except Exception as e:
            raise HTTPException(502, f"LLM connection failed: {e}")

        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)

        return _parse_ollama_response(r.text)


def _parse_ollama_response(raw: str) -> dict:
    try:
        lines   = [l for l in raw.strip().splitlines() if l.strip()]
        parsed  = [json.loads(l) for l in lines]
        content = "".join(p.get("message", {}).get("content", "") for p in parsed)
        return {"response": content}
    except Exception:
        raise HTTPException(500, "Invalid response from LLM")


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Memory-aware text chat.

    Flow:
      1. Append user turn → assemble prompt  (sync, instant)
      2. Call LLM with full message history
      3. Store assistant reply
      4. Summarize overflow in background    (after response is sent)
    """
    messages, overflow, old_summary = add_turn_and_get_prompt(
        session_id   = req.session_id,
        user_message = req.message,
    )

    result = await call_llm_with_messages(messages)
    reply  = result["response"]

    record_assistant_reply(req.session_id, reply)

    # Schedule summarization — runs after HTTP response is delivered
    background_tasks.add_task(
        summarize_in_background,
        req.session_id, overflow, old_summary, call_llm,
    )

    return ChatResponse(session_id=req.session_id, response=reply)


@app.post("/chat-file", response_model=ChatResponse)
async def chat_file(
    background_tasks: BackgroundTasks,
    session_id: str        = Form(...),
    message:    str        = Form(""),
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

    try:
        # ── IMAGE ─────────────────────────────────────────────────────────────
        if content_type.startswith("image"):
            image_b64 = base64.b64encode(file_bytes).decode("utf-8")

            # Step 1: one-shot vision pass to extract semantics
            vision_prompt = (
                message.strip()
                or "Describe this image in detail, capturing all key facts, "
                   "decisions, and structural information."
            )
            vision_result = await call_llm_with_image(vision_prompt, image_b64)
            description   = vision_result["response"]

            # Step 2: store description as media_description (image itself is discarded)
            #   ❌  "User shared an image"
            #   ✅  actual extracted content
            messages, overflow, old_summary = add_turn_and_get_prompt(
                session_id        = session_id,
                user_message      = message.strip() or "(image uploaded)",
                media_description = description,
            )

            # Step 3: get conversational reply using the full memory context
            result = await call_llm_with_messages(messages)
            reply  = result["response"]

        # ── PDF ───────────────────────────────────────────────────────────────
        elif content_type == "application/pdf":
            reader    = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            extracted = "".join(page.extract_text() or "" for page in reader.pages)

            if not extracted.strip():
                raise HTTPException(422, "Could not extract text from PDF.")

            # Inject PDF text directly into the user message; treat as text turn
            combined_message = (
                f"{message}\n\n[PDF Content]\n{extracted[:6000]}".strip()
                if message.strip()
                else f"[PDF Content]\n{extracted[:6000]}"
            )

            messages, overflow, old_summary = add_turn_and_get_prompt(
                session_id   = session_id,
                user_message = combined_message,
            )

            result = await call_llm_with_messages(messages)
            reply  = result["response"]

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

    # Schedule summarization — runs after HTTP response is delivered
    background_tasks.add_task(
        summarize_in_background,
        session_id, overflow, old_summary, call_llm,
    )

    return ChatResponse(session_id=session_id, response=reply)


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
    return {
        "status":           "ok",
        "model":            OLLAMA_MODEL,
        "active_sessions":  active_session_count(),
    }


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