"""
config.py
---------
All the "knobs" for the project live here, in one place, so you don't
have to hunt through every file when you want to change something
(e.g. switch the LLM model, or change how big each text chunk is).
"""

import os

# --- Folders -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")   # original files saved here
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")        # vector database files
DB_PATH = os.path.join(BASE_DIR, "metadata.db")         # SQLite metadata (FR-03)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- Embeddings (free, runs locally, no API key needed) ----------------
# "all-MiniLM-L6-v2" is a small, fast, well-regarded sentence-embedding
# model. First run will download it (~90MB) and cache it locally.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ------------------------------------------------------------
# How big each text chunk is (in characters) and how much consecutive
# chunks overlap, so we don't cut a sentence in half and lose context.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval -----------------------------------------------------------
# How many chunks to pull back from the vector DB per question.
TOP_K = 4

# --- LLM (Ollama - free, runs on your own machine) ------------------------
# 1. Install Ollama from https://ollama.com
# 2. Run:  ollama pull llama3.2
# 3. Leave `ollama serve` running (it usually starts automatically).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# --- Vector DB collection name -------------------------------------------
COLLECTION_NAME = "company_documents"
