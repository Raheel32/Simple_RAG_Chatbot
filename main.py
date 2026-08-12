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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
import ingestion
import retrieval
import llm
from config import UPLOAD_DIR

app = FastAPI(title="RAG Chatbot", version="1.0")

# Allows a frontend hosted on a different domain (e.g. a Streamlit Cloud
# app) to call this API. Locked down to GET/POST/DELETE since that's all
# this API exposes; origins left open since there's no user auth/cookies
# here to protect — tighten to your specific frontend URL if you add any.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Create the PostgreSQL tables on startup (safe to call every time)
database.init_db()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None  # groups turns into a conversation; omit for a one-off question


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

    session_id = payload.session_id
    has_documents = bool(database.list_documents())

    # With no documents uploaded at all, there's nothing to search —
    # respond warmly and invite an upload instead of running the message
    # through strict document-grounded retrieval (which would correctly,
    # but unhelpfully, refuse almost everything).
    if not has_documents:
        answer = llm.greeting_reply(has_documents=False)
        if session_id:
            database.add_message(session_id, "user", question)
            database.add_message(session_id, "assistant", answer)
        return {"answer": answer, "sources": [], "session_id": session_id}

    # Greetings/small talk are handled in code, not by the LLM — see
    # llm.is_greeting for why. Skips retrieval and the LLM call entirely:
    # faster, and immune to the model repeating a canned reply for
    # everything else.
    if llm.is_greeting(question):
        answer = llm.greeting_reply(has_documents=True)
        if session_id:
            database.add_message(session_id, "user", question)
            database.add_message(session_id, "assistant", answer)
        return {"answer": answer, "sources": [], "session_id": session_id}

    # Pull recent history for this session (empty list if no session_id,
    # or if this is the first message in a new one) — needed BEFORE
    # retrieval now, since it feeds query rewriting.
    history = database.get_recent_messages(session_id) if session_id else []

    # Rewrite follow-ups like "what about the ghee version?" into a
    # standalone query ("what is the price of Ikhlas Ghee?") so semantic
    # search can actually match the right chunks. No-op (and no extra
    # LLM call) when there's no history yet.
    search_query = llm.rewrite_query(question, history)

    try:
        chunks = retrieval.retrieve_relevant_chunks(search_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {e}")

    try:
        answer = llm.generate_answer(question, chunks, history)
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

    # Store this turn so the NEXT question in this session can reference it.
    if session_id:
        database.add_message(session_id, "user", question)
        database.add_message(session_id, "assistant", answer)

    return {"answer": answer, "sources": sources, "session_id": session_id}


@app.delete("/api/conversations/{session_id}")
def clear_conversation(session_id: str):
    database.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# --------------------------------------------------------------------------
# Simple built-in web UI for manual testing
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
