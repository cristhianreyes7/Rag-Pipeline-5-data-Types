# vectorstore.py
from __future__ import annotations

import os
import hashlib
from typing import List

# Reduce noisy telemetry logs (safe if ignored by some versions)
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["POSTHOG_DISABLED"] = "1"

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import config

# IMPORTANT: must match rag.py + ui.py
COLLECTION_NAME = "final_rag_collection"


def _safe_collection_count(vectordb: Chroma) -> int:
    """Best-effort collection count. Uses internal API; returns 0 if unavailable."""
    try:
        return int(vectordb._collection.count())  # internal, but fine for demo/dev
    except Exception:
        return 0


def _make_chunk_ids(chunks: List[Document]) -> List[str]:
    """
    Create stable IDs so re-adding the same chunks doesn't create duplicates.
    ID is a hash of (source + chunk_index + content).
    """
    ids: List[str] = []
    for d in chunks:
        src = (d.metadata or {}).get("source", "unknown")
        cix = (d.metadata or {}).get("chunk_index", "0")
        content = d.page_content or ""
        raw = f"{src}::chunk={cix}:::{content}".encode("utf-8", errors="ignore")
        ids.append(hashlib.sha256(raw).hexdigest())
    return ids


def build_or_load_chroma(
    chunks: List[Document],
    *,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> Chroma:
    """
    Load or build a persistent Chroma DB at config.CHROMA_DIR.

    - reset=True: delete and rebuild collection
    - reset=False: only add chunks if collection is empty
    """
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    if reset:
        try:
            vectordb.delete_collection()
        except Exception:
            pass

        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(config.CHROMA_DIR),
        )

    existing_count = _safe_collection_count(vectordb)

    # Only add if empty (keeps presentation-day fast and prevents accidental duplication)
    if chunks and existing_count == 0:
        ids = _make_chunk_ids(chunks)
        vectordb.add_documents(chunks, ids=ids)

    return vectordb


def _env_flag(name: str, default: str = "0") -> bool:
    """
    Read boolean flags from environment variables:
    1/true/yes/on => True
    0/false/no/off => False
    """
    v = (os.getenv(name, default) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    import ingest
    import chunking
    import images_ingest
    import audio_ingest
    import email_ingest
    import pdf_ingest  # ✅ NEW: PDF text-only ingestion

    # Control reset without editing code:
    # Windows PowerShell example:
    #   $env:BUILD_RESET="1"; python vectorstore.py
    # Demo day:
    #   $env:BUILD_RESET="0"; python vectorstore.py
    BUILD_RESET = _env_flag("BUILD_RESET", "1")

    print("🔹 Ingesting text documents (txt + html)...")
    text_docs = ingest.ingest_all_text_only()

    print("🔹 Ingesting PDFs (text-only, no OCR)...")
    pdf_docs = pdf_ingest.load_pdfs_text_only()

    print("🔹 Ingesting images (OpenAI Vision → text)...")
    image_docs = images_ingest.load_images()

    print("🔹 Ingesting audio (OpenAI Transcribe → text)...")
    audio_docs = audio_ingest.load_audio_documents()

    print("🔹 Ingesting emails (.eml → text)...")
    email_docs = email_ingest.load_emails()

    all_docs = text_docs + pdf_docs + image_docs + audio_docs + email_docs

    print(
        "Total docs: "
        f"{len(text_docs)} text + {len(pdf_docs)} pdf + {len(image_docs)} image + "
        f"{len(audio_docs)} audio + {len(email_docs)} email"
    )

    print("🔹 Chunking...")
    chunks = chunking.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    print("🔹 Loading/building Chroma vector store...")
    # Load once to show before-count (optional but nice)
    tmp_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        persist_directory=str(config.CHROMA_DIR),
    )
    before = _safe_collection_count(tmp_db)

    db = build_or_load_chroma(chunks, reset=BUILD_RESET)

    after = _safe_collection_count(db)

    print(f"✅ Vector DB path: {config.CHROMA_DIR}")
    print(f"✅ Collection: {COLLECTION_NAME}")
    print(f"✅ Reset used: {BUILD_RESET}")
    print(f"✅ Count before: {before} | Count after: {after}")

    # Quick retrieval test (PDF-focused, so you can verify PDFs are included)
    query = "What does the PDF say about the campus or buildings?"
    results = db.similarity_search(query, k=config.TOP_K)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    if results:
        print("Top result type:", results[0].metadata.get("type"))
        print("Top result source:", results[0].metadata.get("source"))
        print(results[0].page_content[:700])

