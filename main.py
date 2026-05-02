from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import httpx, os, json, io
from dotenv import load_dotenv
from PIL import Image
import PyPDF2

# CORS
from fastapi.middleware.cors import CORSMiddleware

# Optional OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_HOST or not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_HOST or OLLAMA_MODEL in .env")

app = FastAPI(title="Gemma4 Chat Service", version="2.1.0")

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


# ---------- LLM CALL ----------
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
        raise HTTPException(status_code=413, detail="File too large")

    content = ""

    try:
        # IMAGE
        if file.content_type.startswith("image"):
            if not OCR_AVAILABLE:
                content = "[OCR not available - install pytesseract]"
            else:
                image = Image.open(io.BytesIO(file_bytes))
                content = pytesseract.image_to_string(image)

        # PDF
        elif file.content_type == "application/pdf":
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                content += page.extract_text() or ""

        # AUDIO
        elif file.content_type.startswith("audio"):
            content = "[Audio received - transcription not implemented]"

        # VIDEO
        elif file.content_type.startswith("video"):
            content = "[Video received - processing not implemented]"

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

    final_prompt = f"""
User Message:
{message}

Extracted Content:
{content}
"""

    return await call_llm(final_prompt)


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