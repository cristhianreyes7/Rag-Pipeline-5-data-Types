# chunking.py
from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Standard chunking for ALL text-based content in this project.
    Output is still List[Document], but now each Document is a chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # add chunk_index per source (helps traceability + citations)
    counters = {}
    for d in chunks:
        src = d.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        d.metadata["chunk_index"] = counters[src]

    return chunks


if __name__ == "__main__":
    import ingest

    docs = ingest.ingest_all_text_only()
    chunks = split_documents(docs)

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    if chunks:
        print("Sample metadata:", chunks[0].metadata)
        print("Sample preview:", chunks[0].page_content[:400])
