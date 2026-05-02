from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import httpx, os, json, io, base64
from dotenv import load_dotenv
import PyPDF2

# CORS
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_HOST or not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_HOST or OLLAMA_MODEL in .env")

app = FastAPI(title="Gemma4 Chat Service", version="2.2.0")

# ---------- CORS CONFIG ----------
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

if not origins or origins == [""]:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------


# ---------- CONFIG ----------
MAX_FILE_SIZE_MB = 5


# ---------- MODELS ----------
class ChatRequest(BaseModel):
    message: str


# ---------- LLM CALL (text only) ----------
async def call_llm(prompt: str):
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM connection failed: {str(e)}")

        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        try:
            lines = [l for l in r.text.strip().splitlines() if l.strip()]
            parsed = [json.loads(l) for l in lines]
            content = "".join(p.get("message", {}).get("content", "") for p in parsed)
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid response from LLM")

        return {"response": content}


# ---------- LLM CALL (image + text) ----------
async def call_llm_with_image(prompt: str, image_b64: str):
    """
    Send image directly to Gemma4 as base64.
    Gemma4 is multimodal — no OCR needed, it understands images natively.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],   # base64 string, no data URI prefix
                        }
                    ],
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM connection failed: {str(e)}")

        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        try:
            lines = [l for l in r.text.strip().splitlines() if l.strip()]
            parsed = [json.loads(l) for l in lines]
            content = "".join(p.get("message", {}).get("content", "") for p in parsed)
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid response from LLM")

        return {"response": content}


# ---------- TEXT CHAT ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    return await call_llm(req.message)


# ---------- FILE CHAT ----------
@app.post("/chat-file")
async def chat_file(file: UploadFile = File(...), message: str = Form("")):

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")

    try:
        # ── IMAGE → send directly to Gemma4 as base64 ──────────────────────
        if file.content_type and file.content_type.startswith("image"):
            image_b64 = base64.b64encode(file_bytes).decode("utf-8")
            prompt = message.strip() if message.strip() else "Describe this image in detail."
            return await call_llm_with_image(prompt, image_b64)

        # ── PDF → extract text → send to Gemma4 ────────────────────────────
        elif file.content_type == "application/pdf":
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            extracted = ""
            for page in reader.pages:
                extracted += page.extract_text() or ""

            if not extracted.strip():
                raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

            prompt = f"""User Message:\n{message}\n\nPDF Content:\n{extracted[:6000]}"""
            return await call_llm(prompt)

        # ── AUDIO ───────────────────────────────────────────────────────────
        elif file.content_type and file.content_type.startswith("audio"):
            raise HTTPException(
                status_code=400,
                detail="Audio transcription not supported. Use /voice/transcribe endpoint."
            )

        # ── VIDEO ───────────────────────────────────────────────────────────
        elif file.content_type and file.content_type.startswith("video"):
            raise HTTPException(
                status_code=400,
                detail="Video processing not supported yet."
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


# ---------- HEALTH ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL}


@app.get("/")
def root():
    return {"service": "Gemma4 Chat Service", "docs": "/docs"}


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)