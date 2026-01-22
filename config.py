# config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# =========================
# Environment
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in the .env file.")

# =========================
# Base paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Raw data subfolders
PDF_DIR = DATA_DIR / "pdf"
TXT_DIR = DATA_DIR / "txt"
HTML_DIR = DATA_DIR / "html"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
EML_DIR = DATA_DIR / "eml"

# Processed outputs
TRANSCRIPTS_DIR = OUTPUTS_DIR / "transcripts"
IMAGE_TEXT_DIR = OUTPUTS_DIR / "image_text"
PDF_VISION_DIR = OUTPUTS_DIR / "pdf_page_vision"

# Create folders if they don't exist
for p in [
    DATA_DIR,
    PDF_DIR,
    TXT_DIR,
    HTML_DIR,
    IMAGES_DIR,
    AUDIO_DIR,
    EML_DIR,
    OUTPUTS_DIR,
    TRANSCRIPTS_DIR,
    IMAGE_TEXT_DIR,
    PDF_VISION_DIR,
    CHROMA_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# =========================
# OpenAI models
# =========================
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
VISION_MODEL = "gpt-4o-mini"  # vision-capable

# =========================
# Chunking & retrieval
# =========================
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

# =========================
# PDF rule (later)
# =========================
MIN_TEXT_CHARS = 200
