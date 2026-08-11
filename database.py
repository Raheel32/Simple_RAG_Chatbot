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
from config import (
    DATABASE_URL,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)


def get_connection():
    if DATABASE_URL:
        # Railway (and most cloud providers) hand you one connection
        # string that already encodes host/port/db/user/password.
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def init_db():
    """Create the documents and messages tables if they don't already exist."""
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, created_at)"
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


# --- Conversation history -------------------------------------------------
# Each browser tab gets its own session_id (generated client-side). Turns
# are stored per session so follow-up questions can reference earlier ones
# ("what about the ghee version?").

def add_message(session_id: str, role: str, content: str):
    """role is 'user' or 'assistant'."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
        (session_id, role, content, datetime.utcnow()),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_recent_messages(session_id: str, limit: int = 6):
    """
    Returns the last `limit` messages for a session, oldest first, ready
    to drop straight into the LLM prompt. `limit` counts individual
    messages (so limit=6 is roughly the last 3 question/answer pairs) —
    kept small on purpose since a longer history means a longer prompt,
    which means slower generation, especially on CPU.
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT role, content FROM (
            SELECT role, content, created_at FROM messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        ) recent
        ORDER BY created_at ASC
        """,
        (session_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(row) for row in rows]


def clear_session(session_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
    conn.commit()
    cur.close()
    conn.close()
