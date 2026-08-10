"""
llm.py
------
Implements FR-06 (AI Answer Generation) and FR-08 (Unknown Question
Handling) by calling a locally-running Ollama model.

Why Ollama? It's free, runs entirely on your own machine (no API key,
no per-request cost), which matches your "free/local only" requirement.
Install: https://ollama.com, then `ollama pull llama3.2`.
"""

import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

NO_ANSWER_PHRASE = "Mujhe provided documents mein is question ka relevant answer nahi mila."

SYSTEM_PROMPT = f"""You are a company knowledge-base assistant.
Answer the user's question using ONLY the CONTEXT provided below.
Rules:
- Do not use any outside knowledge. Do not guess or make anything up.
- If the answer is not clearly contained in the CONTEXT, reply with
  exactly this sentence and nothing else: "{NO_ANSWER_PHRASE}"
- Keep answers concise and directly address the question.
"""


def build_prompt(question: str, chunks: list) -> str:
    if not chunks:
        context = "(no relevant context found)"
    else:
        context = "\n\n".join(
            f"[Source: {c['filename']}, Page {c['page_number']}]\n{c['text']}" for c in chunks
        )
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def generate_answer(question: str, chunks: list) -> str:
    """
    Calls Ollama's /api/generate endpoint with the retrieved context.
    If retrieval found nothing at all, we skip the LLM call entirely
    and return the "not found" fallback directly (also cheaper/faster).
    """
    if not chunks:
        return NO_ANSWER_PHRASE

    prompt = build_prompt(question, chunks)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,      # deterministic — good for factual lookups, not creative writing
                    "num_predict": 200,    # caps response length so it can't ramble on and slow things down
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip() or NO_ANSWER_PHRASE

    except requests.exceptions.ConnectionError:
        # This is the #1 error beginners hit: Ollama isn't running.
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{OLLAMA_BASE_URL}. Make sure Ollama is installed and running "
            f"(`ollama serve`), and that you've pulled the model with "
            f"`ollama pull {OLLAMA_MODEL}`."
        )
    except requests.exceptions.Timeout:
        # Connection succeeded but no response came back in time — usually
        # also means Ollama isn't actually running/listening on that port,
        # or the model is still loading for the first time.
        raise RuntimeError(
            f"Ollama at {OLLAMA_BASE_URL} did not respond in time. "
            f"Make sure Ollama is running (`ollama serve`) and that you've "
            f"pulled the model with `ollama pull {OLLAMA_MODEL}`. If this is "
            f"your first request, the model may still be loading — try again "
            f"in a minute."
        )
    except requests.exceptions.RequestException as e:
        # Catch-all for any other network-level failure so it never
        # surfaces as a raw, unhandled 500 to the user.
        raise RuntimeError(f"Error contacting Ollama: {e}")
