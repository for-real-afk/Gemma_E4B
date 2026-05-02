# Gemma4 Chat Service

A FastAPI-based backend service for interacting with a Gemma LLM via an Ollama-compatible API.
Supports text chat and document-based interaction.

---

## 🚀 Features

* Chat endpoint (`/chat`)
* File-based chat (`/chat-file`)
* PDF text extraction
* Optional OCR for images
* Health check endpoint
* CORS-enabled for frontend integration

---

## 📦 Requirements

* Python 3.9+
* pip
* Virtual environment (recommended)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/sathvik1607/Gemma_E4B.git
cd Gemma_E4B
```

---

### 2. Create virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate   # Windows
# source myenv/bin/activate  # Linux/Mac
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory and define the following variables:

* `OLLAMA_HOST` → URL of your Ollama server
* `OLLAMA_MODEL` → model name to use
* `OLLAMA_API_KEY` → API authentication key
* `ALLOWED_ORIGINS` → frontend origin(s) allowed for CORS

⚠️ Do **not** commit your `.env` file to version control.

---

## ▶️ Run the Server

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## 📄 API Documentation

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Endpoints

### 1. Chat

**POST** `/chat`

```json
{
  "message": "Hello"
}
```

---

### 2. Chat with File

**POST** `/chat-file`

Form data:

* `file`: upload file (image/pdf)
* `message`: optional text

---

### 3. Health Check

**GET** `/health`

---

## 🌐 Frontend Integration

Example:

```js
fetch(`${import.meta.env.VITE_API_URL}/chat`)
```

Set in frontend `.env`:

```
VITE_API_URL=<your-backend-url>
```

---

## 🛡️ CORS Configuration

Set allowed frontend origins using:

```
ALLOWED_ORIGINS=<your-frontend-url>
```

---

## 🧠 Notes

* OCR requires `pytesseract` to be installed
* File size limit: 5MB
* Ensure Ollama service is reachable from your backend

---

## 📌 Future Improvements

* Add authentication & API protection
* Improve file processing robustness
* Add monitoring and logging


---

## 👨‍💻 Author

Sathvik
