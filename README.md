# RAG Chatbot — MVP (built from your BRD)

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
- Metadata DB: SQLite (single file — swap for PostgreSQL later if you need multi-user)

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

## 3. Set up the Python project

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

## 4. Run the server

```bash
uvicorn main:app --reload
```

Open your browser at **http://127.0.0.1:8000** — you'll see a simple chat UI
where you can:
1. Upload a PDF/DOCX/TXT file (left sidebar)
2. Wait a few seconds for it to be processed
3. Ask a question about it in the chat box

## 5. API endpoints (if you want to test with curl/Postman instead of the UI)

```bash
# Upload a document
curl -X POST -F "file=@Employee_Handbook.pdf" http://127.0.0.1:8000/api/documents/upload

# List uploaded documents
curl http://127.0.0.1:8000/api/documents

# Ask a question
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the annual leave policy?"}'

# Delete a document
curl -X DELETE http://127.0.0.1:8000/api/documents/<document_id>
```

## 6. How it works, step by step (matches BRD section 10)

1. **Upload**: file is saved to `uploaded_docs/`.
2. **Extraction**: `ingestion.py` reads text per page (PDF) or as a whole
   (DOCX/TXT) using PyMuPDF / python-docx.
3. **Chunking**: each page's text is split into ~800-character overlapping
   chunks so we don't lose context at boundaries.
4. **Embedding**: each chunk is converted into a vector using
   `sentence-transformers` — this vector captures the *meaning* of the text.
5. **Storage**: chunk text + vector + metadata (filename, page number) go
   into Chroma, a local vector database.
6. **Question**: when you ask something, your question is embedded the same
   way.
7. **Retrieval**: Chroma finds the chunks whose vectors are closest in
   meaning to your question (semantic search, not just keyword matching).
8. **Generation**: those chunks are sent to the local LLM (via Ollama) with
   an instruction to answer *only* from that context, and to say it doesn't
   know if the answer isn't there (FR-08 — this is what prevents
   hallucination).
9. **Answer + Source**: the answer and the filename/page it came from are
   returned to the chat UI.

## 7. Things to try next (from the BRD's "Future Enhancements")

- Swap SQLite → PostgreSQL if multiple people will manage documents at once
- Add conversation history / memory across turns
- Add simple login for the Admin role vs. Normal User role
- Add a "confidence" indicator based on the retrieval distance score
  (already returned by `retrieval.py`, just not shown in the UI yet)

## 8. Common issues

| Problem | Fix |
|---|---|
| `Could not reach Ollama` error | Run `ollama serve`, and make sure `ollama pull llama3.2` completed |
| Upload says "No extractable text found" | The PDF is likely scanned images, not real text — you'd need OCR (out of scope for this MVP) |
| Slow first request | The embedding model and Ollama model both need to "warm up" on first use — subsequent requests are faster |
