# RAG Chatbot — MVP (built from your BRD)

[![CI](https://github.com/Raheel32/Simple_RAG_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Raheel32/Simple_RAG_Chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A free/local RAG (Retrieval-Augmented Generation) chatbot: upload documents,
ask questions, get answers grounded in those documents with page-level
source citations. No paid API keys needed.

## How it maps to the BRD

| BRD requirement | File |
|---|---|
| FR-01 Document Upload | `main.py` (`/api/documents/upload`) |
| FR-02 Document Processing (extract/chunk/embed) | `ingestion.py` |
| FR-03 Knowledge Base (metadata) | `database.py` |
| FR-04 User Question | `static/index.html`, `main.py` (`/api/ask`) |
| FR-05 Relevant Information Retrieval | `retrieval.py` |
| FR-06 AI Answer Generation | `llm.py` |
| FR-07 Source Reference | `main.py` (`sources` in the `/api/ask` response) |
| FR-08 Unknown Question Handling | `llm.py` (`NO_ANSWER_PHRASE`) |

**Stack used** (all free/local, per your requirement):
- Backend: FastAPI
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`) — runs on your CPU
- Vector DB: Chroma (local, file-based — no server to run)
- LLM: Ollama (local model, e.g. `llama3.2`)
- Metadata DB: PostgreSQL

## 1. Prerequisites

- Python 3.10–3.12 (3.13+ can have issues with some ML packages — if you're
  on 3.14 like your other project, consider a separate virtual environment
  with 3.11 for this one)
- [Ollama](https://ollama.com) installed

## 2. Install Ollama and pull a model

```bash
# after installing Ollama from ollama.com:
ollama pull llama3.2
```

Ollama runs a local server automatically on `http://localhost:11434`. If it's
not running, start it with `ollama serve`.

## 3. Set up PostgreSQL

You need a running PostgreSQL server with a `rag_chatbot` database.

```bash
# Using createdb (comes with PostgreSQL):
createdb -U postgres rag_chatbot
```

If that prompts for a password and you're not sure what it is, you can also
create it via `psql`:

```bash
psql -U postgres
# then at the psql prompt:
CREATE DATABASE rag_chatbot;
\q
```

By default the app connects as user `postgres` on `localhost:5432` with
password `postgres`. If your setup differs, set these before running the
app (PowerShell):

```powershell
$env:POSTGRES_PASSWORD = "your_actual_password"
```

(Same idea for `POSTGRES_USER`, `POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_DB` if any of those differ too — see `config.py`.)

The `documents` table itself is created automatically the first time you
run the app — no manual schema setup needed beyond creating the database.

## 4. Set up the Python project

```bash
cd rag_chatbot
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

The first run will download the embedding model (~90MB) automatically — this
only happens once.

## 5. Run the server

```bash
uvicorn main:app --reload
```

Open your browser at **http://127.0.0.1:8000** — you'll see a simple chat UI
where you can:
1. Upload a PDF/DOCX/TXT file (left sidebar)
2. Wait a few seconds for it to be processed
3. Ask a question about it in the chat box

## 5b. (Optional) Run the Streamlit frontend instead

There's also a Streamlit UI (`streamlit_app.py`) that talks to the same
FastAPI backend over HTTP — it's a separate frontend, not a replacement,
so **both servers need to be running at once**:

```bash
# Terminal 1 — the backend (same as step 5 above)
uvicorn main:app --reload

# Terminal 2 — the Streamlit frontend
streamlit run streamlit_app.py
```

Streamlit opens automatically at **http://localhost:8501**. It has the same
features as the HTML UI (upload, chat, sources, New chat) — pick whichever
one you prefer, or run both side by side against the same backend and same
documents.

## 6. API endpoints (if you want to test with curl/Postman instead of the UI)

```bash
# Upload a document
curl -X POST -F "file=@Employee_Handbook.pdf" http://127.0.0.1:8000/api/documents/upload

# List uploaded documents
curl http://127.0.0.1:8000/api/documents

# Ask a question (session_id is optional — include it to enable follow-up
# questions like "what about the other size?" referencing earlier turns)
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the annual leave policy?", "session_id": "any-string-you-pick"}'

# Clear a conversation's history
curl -X DELETE http://127.0.0.1:8000/api/conversations/<session_id>

# Delete a document
curl -X DELETE http://127.0.0.1:8000/api/documents/<document_id>
```

## 7. How it works, step by step (matches BRD section 10)

1. **Upload**: file is saved to `uploaded_docs/`.
2. **Extraction**: `ingestion.py` reads text per page (PDF) or as a whole
   (DOCX/TXT) using PyMuPDF / python-docx.
3. **Chunking**: each page's text is split line-by-line into ~500-character
   groups of whole lines (never cutting a line in half), so a product row
   or sentence never gets separated from itself across two chunks.
4. **Embedding**: each chunk is converted into a vector using
   `sentence-transformers` — this vector captures the *meaning* of the text.
5. **Storage**: chunk text + vector + metadata (filename, page number) go
   into Chroma, a local vector database.
6. **Question**: when you ask something, if there's conversation history
   in your session, a quick LLM call first rewrites follow-ups like "what
   about the ghee version?" into a standalone query ("what is the price
   of Ikhlas Ghee?") — skipped entirely on the first message in a chat.
7. **Embedding**: that (possibly rewritten) query is embedded the same
   way as the document chunks.
8. **Retrieval**: Chroma finds the chunks whose vectors are closest in
   meaning to the query (semantic search, not just keyword matching).
9. **Generation**: those chunks — plus your original question and recent
   conversation turns — are sent to the local LLM (via Ollama) with an
   instruction to answer *only* from that context, and to say it doesn't
   know if the answer isn't there (FR-08 — this is what prevents
   hallucination).
10. **Answer + Source**: the answer and the filename/page it came from are
    returned to the chat UI, and the turn is saved to this session's
    history for future follow-ups.

## 8. Things to try next (from the BRD's "Future Enhancements")

- Add simple login for the Admin role vs. Normal User role
- Add a "confidence" indicator based on the retrieval distance score
  (already returned by `retrieval.py`, just not shown in the UI yet)

## 9. Common issues

| Problem | Fix |
|---|---|
| `Could not reach Ollama` error | Run `ollama serve`, and make sure `ollama pull llama3.2` completed |
| Postgres connection error on startup | Confirm the server is running and the `rag_chatbot` database exists; check `POSTGRES_PASSWORD` matches your actual password |
| Upload says "No extractable text found" | The PDF is likely scanned images, not real text — you'd need OCR (out of scope for this MVP) |
| Slow first request | The embedding model and Ollama model both need to "warm up" on first use — subsequent requests are faster |

## 10. Deploying online (Railway backend + Streamlit Cloud frontend)

This gives you a public link to share, like your Hospital Management
System project. Two pieces get deployed separately:

- **Backend** (`main.py` + Postgres + Chroma) → Railway
- **Frontend** (`streamlit_app.py`) → Streamlit Community Cloud

They talk to each other over the internet, so **deploy the backend
first** — you'll need its URL for the frontend's config.

### 10a. Get a free Groq API key

The deployed backend uses Groq instead of Ollama (running a full local
LLM isn't practical on typical free cloud hosting — see the note in
`config.py`).

1. Go to https://console.groq.com and sign up (free)
2. Create an API key
3. Keep it handy for the Railway step below

### 10b. Deploy the backend to Railway

1. Go to https://railway.app, sign up, and create a **New Project**
2. Choose **Deploy from GitHub repo** → select `Simple_RAG_Chatbot`
   (you'll need to connect your GitHub account if you haven't)
3. Railway will detect Python and start building automatically using
   `requirements.txt` and the `Procfile`
4. Click **+ New** → **Database** → **Add PostgreSQL** in the same
   project. Railway automatically creates a `DATABASE_URL` variable and
   makes it available to your backend service — no manual connection
   string needed.
5. Click **+ New** → **Volume**, mount it at `/data`. This is what makes
   your uploaded documents and vector database survive redeploys
   (without it, Railway's filesystem resets every deploy).
6. On your backend service, go to **Variables** and add:
   ```
   DATA_DIR=/data
   LLM_PROVIDER=groq
   GROQ_API_KEY=<your key from 10a>
   ```
7. Go to **Settings → Networking → Generate Domain** to get a public
   URL like `https://your-app.up.railway.app`. That's your backend's
   public address — copy it, you'll need it next.
8. Visit `<your-railway-url>/` in a browser — you should see the same
   HTML chat UI you've been testing locally, now live on the internet.

### 10c. Deploy the frontend to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **New app** → select the `Simple_RAG_Chatbot` repo, branch
   `main`, and set the main file path to `streamlit_app.py`
3. Before deploying, click **Advanced settings → Secrets** and add:
   ```toml
   API_BASE_URL = "https://your-app.up.railway.app"
   ```
   (use the exact URL from step 10b.7 — no trailing slash)
4. Click **Deploy**. After the build finishes, you'll get a public link
   like `https://your-app.streamlit.app` — that's what you share.

### 10d. Notes

- Both platforms redeploy automatically when I push changes to GitHub
  for you — you'll just need to refresh the link.
- Local dev keeps working exactly as before (Ollama, SQLite→Postgres
  locally, etc.) — `LLM_PROVIDER` defaults to `"ollama"`, so nothing
  changes unless you explicitly set it to `"groq"`.
- If the Streamlit app shows a "can't reach the backend" error, double
  check the `API_BASE_URL` secret matches your Railway domain exactly.
