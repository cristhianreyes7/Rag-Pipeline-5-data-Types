# images_ingest.py
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

import config

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _to_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(config.BASE_DIR))
    except Exception:
        return str(path.name)


def _cache_path_for(img_path: Path) -> Path:
    """
    Cache filename must be unique even if two images share the same stem.
    We hash the relative path.
    """
    rel = _safe_relpath(img_path)
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return config.IMAGE_TEXT_DIR / f"{img_path.stem}_{h}.json"


def _is_cache_valid(img_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_mtime = cached.get("_image_mtime", None)
        return cached_mtime == img_path.stat().st_mtime
    except Exception:
        return False


def describe_image(img_path: Path) -> Dict[str, Any]:
    """
    NO OCR libraries. We use the vision-capable model.
    Returns a structured dict with fields optimized for embeddings + retrieval.
    """
    llm = ChatOpenAI(model=config.VISION_MODEL, temperature=0)

    prompt = """
You are analyzing an image for a university RAG dataset.

Rules:
- Do NOT invent unreadable text.
- If you are unsure about a word, omit it or mark it as uncertain.
- Keep it grounded in what is visible.

Return ONLY valid JSON with this exact schema:
{
  "visible_text": ["..."],                // list of lines/phrases you can clearly read (empty list if none)
  "entities": {
    "places": ["..."],                    // place names, streets, campus names, building labels
    "addresses": ["..."],                 // any full/partial addresses visible
    "organizations": ["..."],             // SRH, etc.
    "transport": ["..."]                  // U-Bahn/S-Bahn/bus/tram names if visible
  },
  "image_summary": "...",                 // 2-4 sentences summary
  "details": ["..."],                     // bullet-like details (map arrows, labels, landmarks, directions shown)
  "search_keywords": ["..."]              // 8-15 keywords for retrieval
}

Now analyze the image.
""".strip()

    data_url = _to_data_url(img_path)

    resp = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
    )

    raw = resp.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "visible_text": [],
            "entities": {"places": [], "addresses": [], "organizations": [], "transport": []},
            "image_summary": "Model returned non-JSON output. Raw content stored in details.",
            "details": [raw[:2000]],
            "search_keywords": [],
        }

    data["_image_mtime"] = img_path.stat().st_mtime
    return data


def image_dict_to_document_text(img_path: Path, data: Dict[str, Any]) -> str:
    rel = _safe_relpath(img_path)

    visible_text = data.get("visible_text", []) or []
    entities = data.get("entities", {}) or {}
    places = entities.get("places", []) or []
    addresses = entities.get("addresses", []) or []
    orgs = entities.get("organizations", []) or []
    transport = entities.get("transport", []) or []
    summary = data.get("image_summary", "") or ""
    details = data.get("details", []) or []
    keywords = data.get("search_keywords", []) or []

    lines = []
    lines.append("TYPE: image")
    lines.append(f"SOURCE: {rel}")
    lines.append("")

    if summary:
        lines.append("SUMMARY:")
        lines.append(summary.strip())
        lines.append("")

    if visible_text:
        lines.append("VISIBLE_TEXT:")
        for t in visible_text:
            t = str(t).strip()
            if t:
                lines.append(f"- {t}")
        lines.append("")

    if places or addresses or orgs or transport:
        lines.append("ENTITIES:")
        if places:
            lines.append("Places: " + "; ".join([str(x).strip() for x in places if str(x).strip()]))
        if addresses:
            lines.append("Addresses: " + "; ".join([str(x).strip() for x in addresses if str(x).strip()]))
        if orgs:
            lines.append("Organizations: " + "; ".join([str(x).strip() for x in orgs if str(x).strip()]))
        if transport:
            lines.append("Transport: " + "; ".join([str(x).strip() for x in transport if str(x).strip()]))
        lines.append("")

    if details:
        lines.append("DETAILS:")
        for d in details:
            d = str(d).strip()
            if d:
                lines.append(f"- {d}")
        lines.append("")

    if keywords:
        lines.append("SEARCH_KEYWORDS:")
        lines.append(", ".join([str(k).strip() for k in keywords if str(k).strip()]))

    return "\n".join(lines).strip()


def load_images(use_cache: bool = True) -> List[Document]:
    docs: List[Document] = []

    image_files: List[Path] = [
        p for p in config.IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    for img_path in sorted(image_files):
        cache_path = _cache_path_for(img_path)

        if use_cache and _is_cache_valid(img_path, cache_path):
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            data = describe_image(img_path)
            cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        doc_text = image_dict_to_document_text(img_path, data)

        docs.append(
            Document(
                page_content=doc_text,
                metadata={
                    "source": _safe_relpath(img_path),
                    "type": "image",
                },
            )
        )

    return docs


if __name__ == "__main__":
    docs = load_images(use_cache=True)
    print(f"Loaded {len(docs)} image documents.")
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample preview:\n", docs[0].page_content[:600])

