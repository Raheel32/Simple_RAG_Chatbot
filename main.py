"""
main.py
-------
FastAPI app that exposes the chatbot as a small web API + serves a
simple HTML chat page. This is the entry point of the project.

Run it with:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import database
import ingestion
import retrieval
import llm
from config import UPLOAD_DIR

app = FastAPI(title="RAG Chatbot", version="1.0")

# Create the SQLite metadata table on startup (safe to call every time)
database.init_db()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class AskRequest(BaseModel):
    question: str


# --------------------------------------------------------------------------
# FR-01 / FR-02 / FR-03: Document Upload, Processing, Knowledge Base
# --------------------------------------------------------------------------
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    saved_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingestion.ingest_document(saved_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"status": "success", **result}


@app.get("/api/documents")
def list_documents():
    return database.list_documents()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    ingestion.delete_document_from_store(document_id)
    return {"status": "deleted", "document_id": document_id}


# --------------------------------------------------------------------------
# FR-04 / FR-05 / FR-06 / FR-07 / FR-08: Ask a question, get an answer
# --------------------------------------------------------------------------
@app.post("/api/ask")
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks = retrieval.retrieve_relevant_chunks(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {e}")

    try:
        answer = llm.generate_answer(question, chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # FR-07: Source Reference (deduplicated, in order of relevance)
    seen = set()
    sources = []
    for c in chunks:
        key = (c["filename"], c["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": c["filename"], "page_number": c["page_number"]})

    return {"answer": answer, "sources": sources}


# --------------------------------------------------------------------------
# Simple built-in web UI for manual testing
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
