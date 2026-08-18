"""
Speech-to-text step, using openai-whisper (local, free, no API key).

Note: this project originally targeted `faster-whisper` (CTranslate2-based,
faster on CPU). It was swapped for `openai-whisper` here because
faster-whisper's audio decoding dependency (PyAV) was blocked by a Windows
Application Control policy on the dev machine — openai-whisper instead
shells out to the `ffmpeg` binary directly for audio decoding, which is
unaffected by that policy. If your machine doesn't have that restriction,
faster-whisper is a drop-in swap for extra speed (same Segment/transcribe
interface would need re-adding the streaming generator).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import config

_model = None  # lazily-loaded, cached whisper model


@dataclass
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _get_model():
    global _model
    if _model is None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "Could not import openai-whisper (either it's not "
                "installed - run `pip install openai-whisper` - or an "
                f"underlying dependency failed to load). Original error: {exc}"
            ) from exc
        _model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE)
    return _model


def transcribe(
    audio_path: str | Path,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[Segment]:
    """
    Transcribe an audio file and return a list of timestamped Segments.
    `progress_callback`, if given, is called with a 0..1 float once
    transcription completes (openai-whisper doesn't stream segments the
    way faster-whisper does, so this isn't fine-grained in-flight
    progress - just start/end).
    """
    model = _get_model()
    result = model.transcribe(
        str(audio_path),
        fp16=(config.WHISPER_DEVICE == "cuda"),
        verbose=False,
        language=(config.WHISPER_LANGUAGE or None),
    )

    segments: list[Segment] = []
    raw_segments = result.get("segments", [])
    total = max(1, len(raw_segments))
    for i, seg in enumerate(raw_segments):
        segments.append(
            Segment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip())
        )
        if progress_callback:
            progress_callback((i + 1) / total)

    if not segments and result.get("text"):
        # Fallback: no segment-level timing available, treat whole thing
        # as one segment.
        segments.append(Segment(start=0.0, end=0.0, text=result["text"].strip()))
        if progress_callback:
            progress_callback(1.0)

    return segments


def segments_to_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments).strip()
