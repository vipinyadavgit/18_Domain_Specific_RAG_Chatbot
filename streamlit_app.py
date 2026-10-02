"""Simple web UI for the chatbot.

Run it with:   uv run streamlit run streamlit_app.py
"""

import streamlit as st
from openai import OpenAIError

from app.config import DATA_DIR, TOP_K, setup_logging
from app.ingest import SUPPORTED_TYPES, build_index
from app.rag_chain import generate_answer
from app.retriever import store_exists

setup_logging()

st.set_page_config(page_title="Automotive RAG Chatbot", page_icon="🚗")
st.title("🚗 Automotive RAG Chatbot")
st.caption("Ask about AUTOSAR, ISO 26262, cybersecurity, diagnostics or the vehicle user guide.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)

    st.header("Documents")
    files = [f.name for f in sorted(DATA_DIR.glob("*")) if f.suffix.lower() in SUPPORTED_TYPES]
    st.write("\n".join(f"- {name}" for name in files) or "No documents found in data/")

    if st.button("Re-build index from data/"):
        with st.spinner("Reading documents and creating embeddings..."):
            try:
                total = build_index()
                st.success(f"Done! {total} chunks stored.")
            except Exception as error:
                st.error(f"Ingestion failed: {error}")

    if st.button("Clear chat"):
        st.session_state.messages = []


def show_sources(sources):
    """Show the chunks used for an answer inside a collapsible box."""
    if not sources:
        return
    with st.expander("Sources"):
        for number, chunk in enumerate(sources, start=1):
            page = f", page {chunk['page']}" if chunk["page"] else ""
            st.markdown(f"**[{number}] {chunk['source']}{page}** (similarity {chunk['score']:.2f})")
            st.caption(chunk["text"])


if not store_exists():
    st.warning("No vector store yet. Click 'Re-build index from data/' in the sidebar.")
    st.stop()

# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

# ---------------- New question ----------------
question = st.chat_input("Ask a question...")
if question and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = generate_answer(question.strip(), top_k)
                answer, sources = result["answer"], result["sources"]
            except ValueError as error:
                answer, sources = f"Configuration problem: {error}", []
            except OpenAIError:
                answer, sources = "The OpenAI request failed. Check your API key, credit and internet.", []
            except Exception as error:
                answer, sources = f"Something went wrong: {error}", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
