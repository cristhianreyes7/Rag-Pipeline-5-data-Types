from __future__ import annotations

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import config
from rag import answer_question

COLLECTION_NAME = "final_rag_collection"

# Page config (no emoji)
st.set_page_config(
    page_title="SRH Berlin Campus — RAG Demo",
    layout="centered",
)

st.title("SRH Berlin Campus — RAG Demo")
st.caption("Query-only Retrieval-Augmented Generation (pre-built Chroma DB)")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    k = st.slider(
        "Top-K sources",
        min_value=1,
        max_value=8,
        value=config.TOP_K,
        step=1,
    )
    show_source_chars = st.slider(
        "Source preview length",
        min_value=200,
        max_value=1200,
        value=600,
        step=100,
    )
    st.divider()
    if st.button("Clear chat / reset"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


# Load vector DB once
vectordb = load_vectorstore()
st.info("Vector database loaded.")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Input box ---
user_query = st.chat_input("Ask a question about SRH Berlin campus / buildings...")

if user_query:
    q = user_query.strip()
    if not q:
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            result = answer_question(q, vectordb=vectordb, k=k)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        st.markdown(answer)

        # Sources
        st.markdown("### Sources")
        if not sources:
            st.info("No sources retrieved.")
        else:
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata or {}
                src = meta.get("source", "unknown")
                typ = meta.get("type", "n/a")

                with st.expander(f"[{i}] {src} — type: {typ}", expanded=(i == 1)):
                    st.code((doc.page_content or "")[:show_source_chars])

    # Store assistant answer in history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
