"""
config.py
---------
All the "knobs" for the project live here, in one place, so you don't
have to hunt through every file when you want to change something
(e.g. switch the LLM model, or change how big each text chunk is).
"""

import os

# --- Folders -----------------------------------------------------------
# DATA_DIR lets you point storage at a mounted volume in production (e.g.
# a Railway Volume at /data) instead of the project folder, so uploaded
# docs and the vector DB survive redeploys. Defaults to this folder for
# local dev, where that doesn't matter.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_docs")   # original files saved here
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")        # vector database files

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- PostgreSQL metadata DB (FR-03) ---------------------------------------
# Railway (and most cloud Postgres providers) give you a single
# DATABASE_URL connection string via an environment variable — if that's
# set, we use it directly. Otherwise we fall back to the discrete
# POSTGRES_* variables for local dev.
# On Windows PowerShell, discrete vars are set like:
#   $env:POSTGRES_PASSWORD = "your_password"
DATABASE_URL = os.environ.get("DATABASE_URL")  # set automatically by Railway's Postgres plugin
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_chatbot")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

# --- Embeddings (free, runs locally, no API key needed) ----------------
# "all-MiniLM-L6-v2" is a small, fast, well-regarded sentence-embedding
# model. First run will download it (~90MB) and cache it locally.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ------------------------------------------------------------
# How big each text chunk is (in characters) and how much consecutive
# chunks overlap, so we don't cut a sentence in half and lose context.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Retrieval -----------------------------------------------------------
# How many chunks to pull back from the vector DB per question.
TOP_K = 5

# When the whole knowledge base has this many chunks or fewer, retrieval
# returns ALL of them instead of doing similarity search — see
# retrieval.py for why. 40 chunks at ~500 chars each is roughly a few
# pages of text, comfortably within any LLM's context window.
FULL_CONTEXT_CHUNK_THRESHOLD = 40

# --- LLM provider ----------------------------------------------------------
# Two options:
#   "ollama" (default) — free, runs on your own machine, needs no API key.
#     1. Install Ollama from https://ollama.com
#     2. Run:  ollama pull llama3.2
#     3. Leave `ollama serve` running (it usually starts automatically).
#   "groq" — free tier hosted API (https://console.groq.com), used when
#     deployed to Railway since a full LLM won't run well on typical
#     cloud hobby-tier RAM. Needs a GROQ_API_KEY environment variable.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# --- Vector DB collection name -------------------------------------------
COLLECTION_NAME = "company_documents"
