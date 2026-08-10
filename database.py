"""
database.py
-----------
PostgreSQL wrapper that stores metadata ABOUT each uploaded document
(name, upload date, number of chunks, etc.) — this is separate from
the vector database, which stores the actual embeddings.

Requires a running PostgreSQL server with a database already created:
    createdb rag_chatbot
(or via psql / pgAdmin — see the README for exact steps).

Connection settings come from config.py, which reads them from
environment variables so you're never hardcoding a password in code
that might get committed to a repo.
"""

import psycopg2
import psycopg2.extras
from datetime import datetime
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD


def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def init_db():
    """Create the documents table if it doesn't already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP NOT NULL,
            num_chunks INTEGER NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def add_document(doc_id: str, filename: str, num_chunks: int, status: str = "processed"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (id, filename, upload_date, num_chunks, status) VALUES (%s, %s, %s, %s, %s)",
        (doc_id, filename, datetime.utcnow(), num_chunks, status),
    )
    conn.commit()
    cur.close()
    conn.close()


def list_documents():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM documents ORDER BY upload_date DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # RealDictRow -> plain dict, and convert the datetime to a string so
    # it's JSON-serializable when FastAPI returns it in an API response.
    result = []
    for row in rows:
        d = dict(row)
        d["upload_date"] = d["upload_date"].isoformat()
        result.append(d)
    return result


def delete_document(doc_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()
    cur.close()
    conn.close()
