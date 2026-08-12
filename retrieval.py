"""
retrieval.py
------------
Implements FR-05: Relevant Information Retrieval.

Two modes, chosen automatically:
  - SMALL knowledge base (few chunks total, e.g. a short resume or memo):
    return EVERY chunk as context. Semantic search can miss broad
    questions like "what's in this document?" or "summarize this" since
    they don't closely match any single chunk's wording — but if the
    whole thing fits in the prompt anyway, there's no need to guess.
  - LARGER knowledge base (e.g. a 680-row price list): fall back to
    normal semantic search for the TOP_K most relevant chunks, since
    dumping everything into the prompt would be slow and unfocused.
"""

from config import TOP_K, FULL_CONTEXT_CHUNK_THRESHOLD
from ingestion import embedding_model, collection


def _all_chunks():
    """Returns every chunk currently stored, regardless of the question."""
    results = collection.get(include=["documents", "metadatas"])
    chunks = []
    for text, meta in zip(results.get("documents", []), results.get("metadatas", [])):
        chunks.append(
            {
                "text": text,
                "filename": meta.get("filename"),
                "page_number": meta.get("page_number"),
                "distance": None,  # not applicable — not a similarity search
            }
        )
    return chunks


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K):
    total_chunks = collection.count()

    if 0 < total_chunks <= FULL_CONTEXT_CHUNK_THRESHOLD:
        return _all_chunks()

    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "filename": meta.get("filename"),
                "page_number": meta.get("page_number"),
                "distance": distance,  # lower = more similar
            }
        )
    return chunks
