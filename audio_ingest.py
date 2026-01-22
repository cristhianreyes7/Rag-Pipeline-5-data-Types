# audio_ingest.py
from __future__ import annotations

from pathlib import Path
from typing import List

from openai import OpenAI
from langchain_core.documents import Document

import config

SUPPORTED_AUDIO_EXTS = {".mp3"}  # demo: only mp3


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(config.BASE_DIR))
    except Exception:
        return str(path.name)


def _transcript_path(audio_path: Path) -> Path:
    # transcript saved as outputs/transcripts/<stem>.txt
    return config.TRANSCRIPTS_DIR / f"{audio_path.stem}.txt"


def transcribe_audio(client: OpenAI, path: Path) -> str:
    """
    Transcribe audio using OpenAI transcription model.
    """
    with path.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model=config.TRANSCRIPTION_MODEL,
            file=f,
        )
    return (getattr(transcript, "text", "") or "").strip()


def load_audio_documents(use_cache: bool = True) -> List[Document]:
    """
    Loads mp3 files under data/audio.
    Demo mode: uses ONLY the first mp3 (sorted).
    - If use_cache=True and transcript file exists, reuse it (supports manual edits).
    - Otherwise, transcribe with OpenAI and save transcript.
    """
    audio_files = [
        p for p in config.AUDIO_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]

    if not audio_files:
        print(f"No mp3 files found in: {config.AUDIO_DIR}")
        return []

    audio_path = sorted(audio_files)[0]  # demo: first only
    out_path = _transcript_path(audio_path)

    if use_cache and out_path.exists():
        print(f"Using cached transcript: {out_path.name}")
        text = out_path.read_text(encoding="utf-8").strip()
    else:
        print(f"Transcribing audio: {audio_path.name}")
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        text = transcribe_audio(client, audio_path)
        out_path.write_text(text, encoding="utf-8")

    rel = _safe_relpath(audio_path)

    doc = Document(
        page_content=f"TYPE: audio\nSOURCE: {rel}\n\nTRANSCRIPT:\n{text}",
        metadata={"source": rel, "type": "audio"},
    )
    return [doc]


if __name__ == "__main__":
    docs = load_audio_documents(use_cache=True)
    print(f"Loaded {len(docs)} audio documents.")
    if docs:
        print("Sample metadata:", docs[0].metadata)
        print("Sample preview:", docs[0].page_content[:400])
