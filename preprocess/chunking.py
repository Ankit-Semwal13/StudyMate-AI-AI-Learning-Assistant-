"""
Chunking step: turn a flat list of Whisper segments into overlapping
word-window chunks, each tagged with the [start, end] timestamp range it
covers. Chunks are what gets embedded and fed to the LLM (per the doc's
"500-1000 words per chunk, ~100 word overlap" spec).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import config
from speech.whisper import Segment


@dataclass
class Chunk:
    index: int
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_transcript(
    segments: list[Segment],
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """
    Flatten segments into a single stream of (word, start, end) triples,
    then slide a window of `chunk_size_words` words with `overlap_words`
    overlap across it. Each chunk keeps the timestamp of its first and
    last word so it can still be located in the video.
    """
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        seg_words = seg.text.split()
        if not seg_words:
            continue
        # Spread the segment's [start, end] evenly across its words so each
        # word gets an approximate timestamp.
        span = max(seg.end - seg.start, 0.001)
        step = span / len(seg_words)
        for i, w in enumerate(seg_words):
            w_start = seg.start + i * step
            w_end = seg.start + (i + 1) * step
            words.append((w, w_start, w_end))

    if not words:
        return []

    chunks: list[Chunk] = []
    stride = max(1, chunk_size_words - overlap_words)
    idx = 0
    chunk_index = 0
    while idx < len(words):
        window = words[idx: idx + chunk_size_words]
        if not window:
            break
        text = " ".join(w for w, _, _ in window)
        start = window[0][1]
        end = window[-1][2]
        chunks.append(Chunk(index=chunk_index, text=text, start=start, end=end))
        chunk_index += 1
        if idx + chunk_size_words >= len(words):
            break
        idx += stride

    return chunks
