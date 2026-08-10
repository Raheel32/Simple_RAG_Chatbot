"""
streamlit_app.py
-----------------
A Streamlit frontend for the RAG chatbot. This talks to the FastAPI
backend (main.py) over HTTP — it doesn't touch the database, vector
store, or LLM directly. That means BOTH servers need to be running:

    Terminal 1:  uvicorn main:app --reload
    Terminal 2:  streamlit run streamlit_app.py

Streamlit then opens automatically at http://localhost:8501, while the
FastAPI backend keeps running at http://127.0.0.1:8000 underneath it.
"""

import uuid
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# --- Session state ----------------------------------------------------------
# Streamlit re-runs this whole script on every interaction, so anything
# that needs to persist across reruns (chat history, the session id used
# for conversation memory) has to live in st.session_state.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str, "sources": list}


def api_get(path):
    return requests.get(f"{API_BASE_URL}{path}", timeout=10)


def api_post(path, json_body):
    return requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=180)


def api_delete(path):
    return requests.delete(f"{API_BASE_URL}{path}", timeout=10)


# --- Sidebar: knowledge base management (FR-01, FR-02, FR-03) --------------
with st.sidebar:
    st.header("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        # Streamlit re-runs on every interaction, so guard against
        # re-uploading the same file object on every rerun.
        upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_upload_key") != upload_key:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_BASE_URL}/api/documents/upload", files=files, timeout=120)
                    if resp.ok:
                        st.session_state.last_upload_key = upload_key
                        st.success(f"Added {uploaded_file.name}")
                    else:
                        st.error(resp.json().get("detail", "Upload failed"))
                except requests.exceptions.ConnectionError:
                    st.error(f"Can't reach the backend at {API_BASE_URL}. Is `uvicorn main:app --reload` running?")

    st.divider()

    try:
        docs_resp = api_get("/api/documents")
        docs = docs_resp.json() if docs_resp.ok else []
    except requests.exceptions.ConnectionError:
        docs = []
        st.error(f"Can't reach the backend at {API_BASE_URL}. Is `uvicorn main:app --reload` running?")

    if not docs:
        st.caption("No documents uploaded yet.")
    for doc in docs:
        col1, col2 = st.columns([5, 1])
        col1.text(doc["filename"])
        if col2.button("✕", key=f"del_{doc['id']}"):
            api_delete(f"/api/documents/{doc['id']}")
            st.rerun()

    st.divider()
    if st.button("🆕 New chat"):
        api_delete(f"/api/conversations/{st.session_state.session_id}")
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()


# --- Main area: chat (FR-04 through FR-08) ----------------------------------
st.title("RAG Chatbot")
st.caption("Answers are grounded only in the documents you upload.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("sources"):
            source_lines = ", ".join(f"{s['filename']} (p. {s['page_number']})" for s in turn["sources"])
            st.caption(f"Source: {source_lines}")

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = api_post(
                    "/api/ask",
                    {"question": question, "session_id": st.session_state.session_id},
                )
                if resp.ok:
                    data = resp.json()
                    st.write(data["answer"])
                    if data.get("sources"):
                        source_lines = ", ".join(
                            f"{s['filename']} (p. {s['page_number']})" for s in data["sources"]
                        )
                        st.caption(f"Source: {source_lines}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": data["answer"], "sources": data.get("sources")}
                    )
                else:
                    error_msg = f"Error: {resp.json().get('detail', 'Something went wrong')}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.ConnectionError:
                error_msg = f"Can't reach the backend at {API_BASE_URL}. Is `uvicorn main:app --reload` running?"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
