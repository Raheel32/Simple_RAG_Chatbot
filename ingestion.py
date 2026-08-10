"""
ingestion.py
------------
This file implements FR-01, FR-02 and FR-03 from the BRD:
    Document Upload -> Text Extraction -> Chunking -> Embeddings -> Vector DB

Flow for one uploaded file:
    1. Save the raw file to disk (uploaded_docs/)
    2. Extract raw text (PDF / DOCX / TXT all supported)
    3. Split the text into overlapping chunks
    4. Turn each chunk into an embedding (a list of numbers that
       represents its *meaning*)
    5. Store the chunk text + embedding + metadata (filename, page
       number, chunk index) in the Chroma vector database

Later, when a user asks a question, retrieval.py compares the
question's embedding against these stored chunk embeddings to find
the most semantically similar ones (FR-05).
"""

import os
import uuid
import pymupdf as fitz  # PyMuPDF — 'fitz' import name is deprecated, but keeping the alias so the rest of this file's fitz.* calls don't need renaming
import docx
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    UPLOAD_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)
import database

# --- Load the embedding model and vector DB once, at import time ----------
# (Loading a model is slow-ish, so we do it once and reuse it for every
# request rather than reloading it every time.)
print("Loading embedding model (first run downloads it, please wait)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# --- Step 1: Text extraction (supports PDF, DOCX, TXT) ---------------------
def extract_text_with_pages(file_path: str, ext: str):
    """
    Returns a list of (page_number, text) tuples.
    For TXT/DOCX (no real "pages"), everything is treated as page 1.
    """
    ext = ext.lower()

    if ext == ".pdf":
        pages = []
        pdf = fitz.open(file_path)
        for i, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((i, text))
        pdf.close()
        return pages

    elif ext == ".docx":
        document = docx.Document(file_path)
        full_text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        return [(1, full_text)] if full_text.strip() else []

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [(1, text)] if text.strip() else []

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# --- Step 2: Chunking -------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Simple sliding-window chunker. Splits `text` into pieces of
    `chunk_size` characters, each overlapping the previous one by
    `overlap` characters so we don't lose context at chunk boundaries.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# --- Step 3+4+5: full pipeline for one uploaded file ------------------------
def ingest_document(file_path: str, original_filename: str) -> dict:
    """
    Runs the full FR-01 -> FR-03 pipeline for a single file and stores
    the result in Chroma + the metadata SQLite table.
    Returns a small summary dict for the API response.
    """
    ext = os.path.splitext(original_filename)[1].lower()
    doc_id = str(uuid.uuid4())

    pages = extract_text_with_pages(file_path, ext)
    if not pages:
        raise ValueError("No extractable text found in this document.")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for page_number, page_text in pages:
        page_chunks = chunk_text(page_text)
        for idx, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "document_id": doc_id,
                    "filename": original_filename,
                    "page_number": page_number,
                    "chunk_index": idx,
                }
            )
            all_ids.append(f"{doc_id}_{page_number}_{idx}")

    if not all_chunks:
        raise ValueError("Document produced no usable text chunks.")

    # Embeddings: convert every chunk to a vector in one batch call (fast)
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False).tolist()

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    database.add_document(doc_id=doc_id, filename=original_filename, num_chunks=len(all_chunks))

    return {
        "document_id": doc_id,
        "filename": original_filename,
        "pages_processed": len(pages),
        "chunks_created": len(all_chunks),
    }


def delete_document_from_store(doc_id: str):
    """Remove all chunks belonging to one document from Chroma + SQLite."""
    collection.delete(where={"document_id": doc_id})
    database.delete_document(doc_id)
