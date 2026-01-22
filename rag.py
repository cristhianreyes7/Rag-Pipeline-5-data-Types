# rag.py
from __future__ import annotations

from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

import config


SYSTEM_PROMPT = """You are a Retrieval-Augmented Generation (RAG) assistant for a university demo.

RULES (must follow):
- Use ONLY the provided SOURCES as evidence.
- If the answer is not clearly supported by the SOURCES, say exactly:
  "I don't know based on the provided documents."
- Do NOT use outside knowledge.
- Do NOT invent names, dates, addresses, codes, or rules.
- Your answer MUST include citations like [1], [2] referring to the SOURCES.
- Keep the answer short and factual (2–6 sentences).
"""


def _format_sources(docs: List[Document], *, max_chars_each: int = 900) -> str:
    """
    Turn retrieved docs into a clean context block.
    max_chars_each limits noise so the model focuses.
    """
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        typ = meta.get("type", "n/a")

        content = (d.page_content or "").strip()
        if len(content) > max_chars_each:
            content = content[:max_chars_each].rstrip() + "…"

        blocks.append(
            f"SOURCE [{i}]\n"
            f"type: {typ}\n"
            f"source: {src}\n"
            f"content:\n{content}\n"
        )
    return "\n---\n".join(blocks).strip()


def answer_question(
    query: str,
    *,
    vectordb,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Query-only RAG:
    - retrieve top-k chunks from vectordb
    - generate an answer grounded in retrieved context
    Returns:
      {"answer": str, "sources": List[Document]}
    """
    docs: List[Document] = vectordb.similarity_search(query, k=k)

    if not docs:
        return {"answer": "I don't know based on the provided documents.", "sources": []}

    sources_text = _format_sources(docs)

    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)

    user_prompt = f"""USER QUESTION:
{query}

SOURCES:
{sources_text}

TASK:
Answer the question using ONLY the SOURCES.
- If unsupported, output exactly: "I don't know based on the provided documents."
- Include citations like [1], [2] in the answer.
- Do not mention the word "SOURCES" or "context" in the answer.
"""

    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = (resp.content or "").strip()

    # Safety net: if model forgot citations, force an "I don't know" instead of hallucinating.
    # (This keeps demo consistent.)
    has_citation = "[" in answer and "]" in answer
    if answer != "I don't know based on the provided documents." and not has_citation:
        answer = "I don't know based on the provided documents."

    return {"answer": answer, "sources": docs}
