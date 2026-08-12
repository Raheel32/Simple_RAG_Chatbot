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
import pandas as pd
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


# --- Step 1: Text extraction (PDF, DOCX, TXT, CSV, XLSX, XLS) --------------
def _dataframe_to_lines(df: pd.DataFrame) -> str:
    """
    Converts a spreadsheet's rows into readable "Column: value" lines —
    one row per line, matching the line-aware chunker's assumption that
    each line is one self-contained record (same approach that worked
    well for the tabular price list PDF).
    """
    lines = []
    columns = [str(c) for c in df.columns]
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def extract_text_with_pages(file_path: str, ext: str):
    """
    Returns a list of (page_number, text) tuples.
    For TXT/DOCX/CSV (no real "pages"), everything is treated as page 1.
    For XLSX with multiple sheets, each sheet is treated as its own "page"
    so sources can point to which sheet an answer came from.
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

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        text = _dataframe_to_lines(df)
        return [(1, text)] if text.strip() else []

    elif ext in (".xlsx", ".xls"):
        # Some POS/inventory/web-export tools produce files with a .xls
        # extension that are actually HTML tables underneath, not real
        # binary or OOXML spreadsheets. Sniff the first bytes to catch
        # this before handing it to the binary Excel parser, which would
        # otherwise fail with a cryptic "unsupported format" error.
        with open(file_path, "rb") as f:
            head = f.read(512)
        sniff = head.lstrip(b"\xef\xbb\xbf").lstrip().lower()  # strip BOM + whitespace
        if sniff.startswith((b"<html", b"<!doctype", b"<table")):
            tables = pd.read_html(file_path)
            pages = []
            for i, df in enumerate(tables, start=1):
                text = _dataframe_to_lines(df)
                if text.strip():
                    pages.append((i, text))
            return pages

        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        sheets = pd.read_excel(file_path, sheet_name=None, engine=engine)  # dict of {sheet_name: DataFrame}
        pages = []
        for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
            text = _dataframe_to_lines(df)
            if text.strip():
                pages.append((i, f"[Sheet: {sheet_name}]\n{text}"))
        return pages

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# --- Step 2: Chunking -------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Line-aware chunker. Groups whole lines together up to ~chunk_size
    characters, WITHOUT ever splitting a line in half.

    This matters a lot for line-per-record data (like a price list where
    each product is one line: "IKHLAS OIL 500 ML   255   238.75   0").
    A naive fixed-character window can cut a line in half, separating a
    product name from its price across two different chunks — which
    breaks retrieval for exactly that kind of document. Grouping by
    whole lines keeps every record intact in at least one chunk.

    A handful of overlapping lines are carried into the next chunk so
    a record near a chunk boundary still has surrounding context.
    """
    text = text.strip()
    if not text:
        return []

    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    chunks = []
    current_lines = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the newline joining it back
        if current_lines and current_len + line_len > chunk_size:
            chunks.append("\n".join(current_lines))
            # carry the last few lines forward as overlap, budgeted by
            # character count rather than a fixed line count
            overlap_lines = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                overlap_len += len(prev_line) + 1
                if overlap_len > overlap:
                    break
                overlap_lines.insert(0, prev_line)
            current_lines = overlap_lines
            current_len = sum(len(l) + 1 for l in current_lines)

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

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
