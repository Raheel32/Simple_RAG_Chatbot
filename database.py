"""
database.py
-----------
Tiny SQLite wrapper that stores metadata ABOUT each uploaded document
(name, upload date, number of chunks, etc.) — this is separate from
the vector database, which stores the actual embeddings.

Why SQLite and not PostgreSQL (like the BRD's stack table suggests)?
For an MVP, SQLite needs zero setup (it's a single file) and is
functionally identical for our purposes. If you later need to run
this for real with multiple concurrent admins, swapping in
PostgreSQL only means changing this file — nothing else in the
project depends on SQLite specifically.
"""

import sqlite3
from datetime import datetime
from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the documents table if it doesn't already exist."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            num_chunks INTEGER NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def add_document(doc_id: str, filename: str, num_chunks: int, status: str = "processed"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO documents (id, filename, upload_date, num_chunks, status) VALUES (?, ?, ?, ?, ?)",
        (doc_id, filename, datetime.utcnow().isoformat(), num_chunks, status),
    )
    conn.commit()
    conn.close()


def list_documents():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_document(doc_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
