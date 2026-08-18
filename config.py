"""
Central configuration for Video Note Extractor AI.

Everything here is free / local by default:
- LLM        -> Ollama (local), no API key needed
- Speech-to-text -> faster-whisper (local), no API key needed
- Embeddings -> sentence-transformers (local), no API key needed
- Vector DB  -> FAISS (local, on-disk)

All values can be overridden via a `.env` file (see .env.example) or
environment variables, without touching this file.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; if it's not installed, env vars can still
    # be set the normal way (OS env / shell) and everything still works.
    pass

# --------------------------------------------------------------------------
# Paths (mirrors the folder structure of the project)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
NOTES_DIR = BASE_DIR / "notes"
VECTOR_DIR = BASE_DIR / "vectors"

for _dir in (UPLOAD_DIR, TRANSCRIPT_DIR, NOTES_DIR, VECTOR_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# LLM (Ollama - local, free)
# --------------------------------------------------------------------------
# Install Ollama from https://ollama.com and run e.g. `ollama pull llama3.2`
# before starting the app. Any chat-capable Ollama model works; smaller
# models (llama3.2:3b, gemma2:2b, phi3) run comfortably on CPU-only laptops.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))

# --------------------------------------------------------------------------
# Speech-to-text (openai-whisper - local, free)
# --------------------------------------------------------------------------
# Model sizes (accuracy vs speed): tiny, base, small, medium, large
# "base" is a good default for CPU-only machines.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # unused by openai-whisper; kept for a faster-whisper swap-back
# Force a transcription language instead of relying on Whisper's
# auto-detect from the first ~30s of audio - auto-detect can misfire badly
# (e.g. guessing "Chinese" for an English talk) when a video opens with
# music, applause, or a silent/noisy intro before speech starts. Set to
# "" (empty) to fall back to auto-detect.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# --------------------------------------------------------------------------
# Embeddings (sentence-transformers - local, free)
# --------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "700"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "100"))

# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# --------------------------------------------------------------------------
# Downloads (yt-dlp)
# --------------------------------------------------------------------------
# YouTube occasionally 403s a download request (per-format/CDN-edge
# flakiness, or short-lived anti-bot rate-limiting) even though the same
# link works moments later. Retry the whole format-fallback sequence this
# many times, waiting between attempts, before giving up.
DOWNLOAD_MAX_RETRIES = int(os.getenv("DOWNLOAD_MAX_RETRIES", "3"))
DOWNLOAD_RETRY_DELAY_SECONDS = int(os.getenv("DOWNLOAD_RETRY_DELAY_SECONDS", "15"))

# --------------------------------------------------------------------------
# FFmpeg
# --------------------------------------------------------------------------
# Path to the ffmpeg executable. Defaults to "ffmpeg" (expects it on PATH).
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg")

# openai-whisper shells out to a bare `ffmpeg` command internally (it
# doesn't know about FFMPEG_BINARY above), so make sure FFmpeg's directory
# is on PATH for this process - important right after a fresh FFmpeg
# install, before a shell restart has picked up the updated PATH.
if os.path.isabs(FFMPEG_BINARY):
    _ffmpeg_dir = str(Path(FFMPEG_BINARY).parent)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
