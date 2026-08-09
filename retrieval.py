"""
retrieval.py
------------
Implements FR-05: Relevant Information Retrieval.

Given a user's question, we:
    1. Turn the question into an embedding (same model used for chunks)
    2. Ask Chroma for the TOP_K most similar chunks (by meaning, not
       just matching keywords -> this is what "semantic search" means)
    3. Return the chunk text plus its source metadata, so the caller
       can both build the LLM prompt and show "Source: file.pdf, page 3"
"""

from config import TOP_K
from ingestion import embedding_model, collection


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K):
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
