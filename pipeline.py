"""
Orchestrates the full "Overall Architecture" pipeline from the spec:

  Video Upload / YouTube Link
        -> Video Downloader
        -> Audio Extraction (FFmpeg)
        -> Speech-to-Text (Whisper)
        -> Transcript Generation
        -> Chunking Transcript
        -> [Embeddings -> Vector DB]  +  [LLM Pipeline: Notes / Timestamps / Action Items / Summary]
        -> (RAG Question Answering happens later, on demand, via rag/qa.py)

This module has no UI code in it - app.py (Streamlit) just calls
`run_pipeline(...)` and renders the result.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import config
from downloader import youtube
from utils.helpers import extract_audio, save_json, slugify
from speech.whisper import transcribe, segments_to_text, Segment
from preprocess.chunking import chunk_transcript
from embeddings.vector_store import VectorStore
from llm.notes import generate_notes
from llm.summary import generate_summary
from llm.action_items import extract_action_items
from llm.timestamps import extract_timestamps

ProgressFn = Optional[Callable[[str, float], None]]


def _report(progress_callback: ProgressFn, stage: str, fraction: float = 0.0):
    if progress_callback:
        progress_callback(stage, fraction)


def run_pipeline(
    source: str,
    is_url: bool,
    title_hint: Optional[str] = None,
    progress_callback: ProgressFn = None,
) -> dict:
    """
    source: a YouTube/video URL, or a local file path to an uploaded video.
    is_url: True if `source` is a URL, False if it's a local file path.

    Returns a dict with everything the UI needs: transcript, chunks, notes,
    summary, key_takeaways, timestamps, action_items, and a VectorStore
    for the "Ask AI" tab.
    """
    # ---- 1. Video Downloader ------------------------------------------------
    _report(progress_callback, "Downloading video...", 0.0)
    if is_url:
        title = title_hint or youtube.get_video_title(source) or "video"

        def _download_progress(status: str):
            _report(progress_callback, status, 0.0)

        video_path = youtube.download_from_url(source, progress_callback=_download_progress)
    else:
        title = title_hint or Path(source).stem
        video_path = youtube.register_local_file(source)

    slug = slugify(title)

    # ---- 2. Audio Extraction (FFmpeg) ---------------------------------------
    _report(progress_callback, "Extracting audio...", 0.15)
    audio_path = extract_audio(video_path, config.UPLOAD_DIR / f"{slug}.wav")

    # ---- 3. Speech-to-Text (Whisper) + Transcript Generation ---------------
    _report(progress_callback, "Generating transcript...", 0.25)

    def _whisper_progress(frac: float):
        _report(progress_callback, "Generating transcript...", 0.25 + 0.25 * frac)

    segments: list[Segment] = transcribe(audio_path, progress_callback=_whisper_progress)
    full_text = segments_to_text(segments)
    save_json([s.to_dict() for s in segments], config.TRANSCRIPT_DIR / f"{slug}.json")

    # ---- 4. Chunking Transcript ----------------------------------------------
    _report(progress_callback, "Chunking transcript...", 0.5)
    chunks = chunk_transcript(segments)

    # ---- 5a. Embeddings -> Vector Database -----------------------------------
    _report(progress_callback, "Building search index...", 0.55)
    store = VectorStore().build(chunks)
    store.save(config.VECTOR_DIR / slug)

    # ---- 5b. LLM Pipeline: Notes / Summary / Action Items / Timestamps ------
    def _sub_progress(base: float, span: float):
        def cb(frac: float):
            _report(progress_callback, "Creating notes...", base + span * frac)
        return cb

    _report(progress_callback, "Creating notes...", 0.6)
    notes_markdown = generate_notes(chunks, progress_callback=_sub_progress(0.60, 0.15))
    summary, key_takeaways = generate_summary(chunks, progress_callback=_sub_progress(0.75, 0.1))
    timestamps = extract_timestamps(chunks, progress_callback=_sub_progress(0.85, 0.08))
    action_items = extract_action_items(chunks, progress_callback=_sub_progress(0.93, 0.07))

    duration_seconds = max((s.end for s in segments), default=0.0)

    result = {
        "title": title,
        "slug": slug,
        "source": source,
        "is_url": is_url,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "transcript": full_text,
        "segments": [s.to_dict() for s in segments],
        "chunks": [c.to_dict() for c in chunks],
        "notes_markdown": notes_markdown,
        "summary": summary,
        "key_takeaways": key_takeaways,
        "timestamps": timestamps,
        "action_items": action_items,
        "vector_store_dir": str(config.VECTOR_DIR / slug),
        "duration_seconds": duration_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(
        {k: v for k, v in result.items() if k != "segments"},
        config.NOTES_DIR / f"{slug}.json",
    )

    _report(progress_callback, "Done!", 1.0)
    return result
