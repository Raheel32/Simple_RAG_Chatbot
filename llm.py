"""
llm.py
------
Implements FR-06 (AI Answer Generation) and FR-08 (Unknown Question
Handling) by calling either a locally-running Ollama model, or Groq's
hosted API — controlled by LLM_PROVIDER in config.py.

Why two providers? Ollama is free and needs no API key, but it requires
a real machine with real RAM to run on — great for local dev, a poor fit
for a typical free/hobby cloud server. Groq's API is also free (within
generous usage limits) and needs no local compute, so it's what the
deployed version uses instead. Same functions, same call sites — only
_call_llm's internals differ.
"""

import re
import requests
from config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL

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

# --- Greeting / small-talk detection ----------------------------------------
# Handled in CODE, not by asking the LLM to classify — a smaller/faster
# model (like the Groq one used in production) can anchor on its own
# earlier reply and repeat the same canned greeting for EVERY message,
# including real document questions. A plain regex is faster and 100%
# reliable for the narrow case it needs to catch: short greetings only.
_GREETING_PATTERN = re.compile(
    r"^\s*("
    r"hi+|hello+|hey+|hii+|salam|assalam.*|good\s*(morning|afternoon|evening)|"
    r"how\s*are\s*you|what'?s\s*up|"
    r"who\s*are\s*you|what\s*can\s*you\s*do|what\s*is\s*your\s*name|"
    r"thanks?|thank\s*you|bye|goodbye|ok(ay)?"
    r")[\s!.?]*$",
    re.IGNORECASE,
)


def is_greeting(question: str) -> bool:
    """True only for short, unambiguous greetings/small-talk — anything
    with real question content (even 'what is in the file?') should NOT
    match, so it goes through normal document-grounded retrieval."""
    return bool(_GREETING_PATTERN.match(question.strip()))


def greeting_reply(has_documents: bool) -> str:
    if has_documents:
        return (
            "Hi! I'm your document assistant — ask me anything about the "
            "documents you've uploaded and I'll answer based on their content."
        )
    return (
        "Hi! I'm your document assistant. Upload a PDF, DOCX, TXT, CSV, or "
        "Excel file and I can answer questions about it."
    )


REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone
questions using the conversation so far. Rules:
- If the question is already standalone (doesn't depend on earlier turns),
  return it EXACTLY as given, unchanged.
- Otherwise, rewrite it to include whatever context it's implicitly
  referring to (e.g. "it", "that one", "the other size").
- Output ONLY the rewritten question. No explanation, no quotes, no
  extra text.
"""


def _call_ollama(system_prompt: str, prompt: str, num_predict: int, timeout: int) -> str:
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


def _call_groq(system_prompt: str, prompt: str, num_predict: int, timeout: int) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and set it as an "
            "environment variable."
        )
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": num_predict,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API did not respond in time. Try again shortly.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"Groq API error: {detail or e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error contacting Groq: {e}")


def _call_llm(system_prompt: str, prompt: str, num_predict: int, timeout: int = 180) -> str:
    """Dispatches to whichever provider LLM_PROVIDER is set to."""
    if LLM_PROVIDER == "groq":
        return _call_groq(system_prompt, prompt, num_predict, timeout)
    return _call_ollama(system_prompt, prompt, num_predict, timeout)


def rewrite_query(question: str, history: list) -> str:
    """
    Rewrites a follow-up question into a standalone one, using recent
    conversation turns, so RETRIEVAL (not just the final answer) can
    semantically match on the full intent — e.g. "what about the ghee
    version?" -> "what is the price of Ikhlas Ghee?".

    This is a small, separate LLM call before retrieval even happens.
    It costs extra latency, so it's skipped entirely when there's no
    history to rewrite against (first message in a conversation).

    If this call fails for any reason (provider down, timeout, etc.), we
    fall back to the original question rather than failing the whole
    request — query rewriting is a nice-to-have, not essential.
    """
    if not history:
        return question

    turns = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history)
    prompt = f"Conversation so far:\n{turns}\n\nFollow-up question: {question}\n\nStandalone question:"

    try:
        rewritten = _call_llm(REWRITE_SYSTEM_PROMPT, prompt, num_predict=60, timeout=30)
        return rewritten or question
    except RuntimeError:
        return question


def build_prompt(question: str, chunks: list, history: list = None) -> str:
    if not chunks:
        context = "(no documents have been uploaded yet, or none matched this question)"
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
    Calls the configured LLM provider with the retrieved context and,
    optionally, recent conversation turns so follow-up questions like
    "what about the ghee version?" can be understood.

    We always call the LLM, even when chunks is empty (e.g. no documents
    uploaded yet) — the system prompt handles that case itself, replying
    naturally to conversational messages and only using the "not found"
    fallback for factual questions it genuinely can't answer. A hardcoded
    empty-chunks shortcut would incorrectly refuse even a plain "hi".
    """
    prompt = build_prompt(question, chunks, history)
    answer = _call_llm(SYSTEM_PROMPT, prompt, num_predict=200, timeout=180)
    return answer or NO_ANSWER_PHRASE
