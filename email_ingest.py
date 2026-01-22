# email_ingest.py
from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import List, Tuple, Dict, Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document

import config


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(config.BASE_DIR))
    except Exception:
        return str(path.name)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln]).strip()


def _cache_path_for_eml(eml_path: Path) -> Path:
    # outputs/email_text/<stem>.json
    return config.EMAIL_TEXT_DIR / f"{eml_path.stem}.json"


def _is_cache_valid(eml_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached.get("_eml_mtime") == eml_path.stat().st_mtime
    except Exception:
        return False


def _extract_text_from_eml(eml_path: Path) -> Tuple[str, Dict[str, Any]]:
    msg = BytesParser(policy=policy.default).parsebytes(eml_path.read_bytes())

    subject = str(msg.get("subject", "") or "").strip()
    date = str(msg.get("date", "") or "").strip()
    sender = str(msg.get("from", "") or "").strip()
    to = str(msg.get("to", "") or "").strip()

    body_text = ""

    def is_attachment(part) -> bool:
        disp = (part.get("Content-Disposition") or "").lower()
        return "attachment" in disp

    if msg.is_multipart():
        # Prefer text/plain
        for part in msg.walk():
            if is_attachment(part):
                continue
            if part.get_content_type() == "text/plain":
                body_text = (part.get_content() or "").strip()
                if body_text:
                    break

        # Fallback to text/html
        if not body_text:
            for part in msg.walk():
                if is_attachment(part):
                    continue
                if part.get_content_type() == "text/html":
                    body_text = _html_to_text(str(part.get_content() or ""))
                    if body_text:
                        break
    else:
        ctype = msg.get_content_type()
        content = str(msg.get_content() or "")
        if ctype == "text/html":
            body_text = _html_to_text(content)
        else:
            body_text = content.strip()

    body_text = (body_text or "").strip()
    rel = _safe_relpath(eml_path)

    # Retrieval-friendly normalized content
    content_text = (
        f"TYPE: email\n"
        f"SOURCE: {rel}\n\n"
        f"SUBJECT: {subject}\n"
        f"DATE: {date}\n"
        f"FROM: {sender}\n"
        f"TO: {to}\n\n"
        f"BODY:\n{body_text}\n"
    ).strip()

    meta = {
        "subject": subject,
        "date": date,
        "from": sender,
        "to": to,
    }
    return content_text, meta


def load_emails(use_cache: bool = True) -> List[Document]:
    docs: List[Document] = []
    eml_files = sorted(config.EML_DIR.rglob("*.eml"))

    if not eml_files:
        return docs

    for eml_path in eml_files:
        cache_path = _cache_path_for_eml(eml_path)

        if use_cache and _is_cache_valid(eml_path, cache_path):
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            text = cached["content_text"]
            meta = cached["meta"]
        else:
            text, meta = _extract_text_from_eml(eml_path)
            cached = {
                "_eml_mtime": eml_path.stat().st_mtime,
                "content_text": text,
                "meta": meta,
            }
            cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")

        rel = _safe_relpath(eml_path)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": rel, "type": "email", **meta},
            )
        )

    return docs


if __name__ == "__main__":
    docs = load_emails(use_cache=True)
    print(f"Loaded {len(docs)} email documents.")
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample preview:", docs[0].page_content[:500])
