# ingest.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup
from langchain_core.documents import Document

import config


# -------------------------
# Helpers
# -------------------------
def _read_text_file(path: Path) -> str:
    """
    Robust text reader that handles Windows UTF-16 files (BOM),
    plus utf-8 and latin-1 fallbacks.
    """
    raw = path.read_bytes()

    # UTF-16 BOM checks (LE/BE)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")

    # try utf-8
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # fallback latin-1
    return raw.decode("latin-1", errors="ignore")


def _safe_relpath(path: Path) -> str:
    """Best-effort relative path for metadata."""
    try:
        return str(path.relative_to(config.BASE_DIR))
    except Exception:
        return str(path.name)


def _make_doc(
    text: str,
    *,
    source: str,
    doc_type: str,
    extra_meta: Optional[dict] = None,
) -> Document:
    meta = {"source": source, "type": doc_type}
    if extra_meta:
        meta.update(extra_meta)
    return Document(page_content=text, metadata=meta)


# -------------------------
# TXT loader
# -------------------------
def load_txt(path: Path) -> List[Document]:
    text = _read_text_file(path).strip()
    if not text:
        return []
    return [_make_doc(text, source=_safe_relpath(path), doc_type="txt")]


# -------------------------
# HTML loader
# -------------------------
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove common non-content blocks
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up blank lines + trim
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join([ln for ln in lines if ln])

    return cleaned.strip()


def load_html(path: Path) -> List[Document]:
    html = _read_text_file(path)
    text = html_to_text(html)
    if not text:
        return []
    return [_make_doc(text, source=_safe_relpath(path), doc_type="html")]


# -------------------------
# Ingest all (TXT + HTML only)
# -------------------------
def ingest_all_text_only() -> List[Document]:
    docs: List[Document] = []

    # TXT
    for path in sorted(config.TXT_DIR.rglob("*.txt")):
        docs.extend(load_txt(path))

    # HTML
    html_files = list(config.HTML_DIR.rglob("*.html")) + list(config.HTML_DIR.rglob("*.htm"))
    for path in sorted(html_files):
        docs.extend(load_html(path))

    return docs


# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    docs = ingest_all_text_only()
    print(f"Loaded {len(docs)} documents (txt+html).")
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample preview:", docs[0].page_content[:300])
