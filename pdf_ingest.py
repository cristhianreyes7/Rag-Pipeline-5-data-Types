# pdf_ingest.py
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

import config

try:
    from pypdf import PdfReader
except ImportError as e:
    raise ImportError("Missing dependency: pypdf. Install it with: pip install pypdf") from e


def load_pdfs_text_only(min_chars: int = config.MIN_TEXT_CHARS) -> List[Document]:
    """
    Text-only PDF ingest (NO OCR, ignores images).
    Creates one Document per PAGE (better retrieval + citations).
    Skips pages with very little extracted text (often scanned/image-only pages).
    """
    docs: List[Document] = []

    pdf_files = sorted(config.PDF_DIR.rglob("*.pdf"))
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        rel = str(pdf_path.relative_to(config.BASE_DIR))

        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            # Skip basically-empty pages (often image-only/scanned)
            if len(text) < min_chars:
                continue

            # Include page marker in text for nicer retrieval/answers
            page_text = f"TYPE: pdf\nSOURCE: {rel}\nPAGE: {page_index}\n\n{text}".strip()

            docs.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": rel,
                        "type": "pdf",
                        "page": page_index,
                    },
                )
            )

    return docs


if __name__ == "__main__":
    docs = load_pdfs_text_only()
    print(f"Loaded {len(docs)} PDF page-documents.")
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample preview:", docs[0].page_content[:400])
