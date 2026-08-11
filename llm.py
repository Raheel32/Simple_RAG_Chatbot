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
- Recent conversation turns may be included below the context. Use them
  only to understand what the user is referring to (e.g. "it", "that
  one", "the other size") — the CONTEXT is still the only source for
  facts in your answer, never something said earlier in the conversation.
"""


REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone
questions using the conversation so far. Rules:
- If the question is already standalone (doesn't depend on earlier turns),
  return it EXACTLY as given, unchanged.
- Otherwise, rewrite it to include whatever context it's implicitly
  referring to (e.g. "it", "that one", "the other size").
- Output ONLY the rewritten question. No explanation, no quotes, no
  extra text.
"""


def _call_ollama(system_prompt: str, prompt: str, num_predict: int, timeout: int = 180) -> str:
    """Shared low-level call to Ollama's /api/generate, with consistent
    error handling used by both answer generation and query rewriting."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,      # deterministic — good for factual lookups, not creative writing
                    "num_predict": num_predict,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

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


def rewrite_query(question: str, history: list) -> str:
    """
    Rewrites a follow-up question into a standalone one, using recent
    conversation turns, so RETRIEVAL (not just the final answer) can
    semantically match on the full intent — e.g. "what about the ghee
    version?" -> "what is the price of Ikhlas Ghee?".

    This is a small, separate LLM call before retrieval even happens.
    It costs extra latency, so it's skipped entirely when there's no
    history to rewrite against (first message in a conversation).

    If this call fails for any reason (Ollama down, timeout, etc.), we
    fall back to the original question rather than failing the whole
    request — query rewriting is a nice-to-have, not essential.
    """
    if not history:
        return question

    turns = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history)
    prompt = f"Conversation so far:\n{turns}\n\nFollow-up question: {question}\n\nStandalone question:"

    try:
        rewritten = _call_ollama(REWRITE_SYSTEM_PROMPT, prompt, num_predict=60, timeout=30)
        return rewritten or question
    except RuntimeError:
        return question


def build_prompt(question: str, chunks: list, history: list = None) -> str:
    if not chunks:
        context = "(no relevant context found)"
    else:
        context = "\n\n".join(
            f"[Source: {c['filename']}, Page {c['page_number']}]\n{c['text']}" for c in chunks
        )

    history_block = ""
    if history:
        turns = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history)
        history_block = f"RECENT CONVERSATION (for reference only, not a source of facts):\n{turns}\n\n"

    return f"{history_block}CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def generate_answer(question: str, chunks: list, history: list = None) -> str:
    """
    Calls Ollama's /api/generate endpoint with the retrieved context and,
    optionally, recent conversation turns so follow-up questions like
    "what about the ghee version?" can be understood.
    If retrieval found nothing at all, we skip the LLM call entirely
    and return the "not found" fallback directly (also cheaper/faster).
    """
    if not chunks:
        return NO_ANSWER_PHRASE

    prompt = build_prompt(question, chunks, history)
    answer = _call_ollama(SYSTEM_PROMPT, prompt, num_predict=200, timeout=180)
    return answer or NO_ANSWER_PHRASE
