"""
Important timestamps extraction: labels the topic that starts at each
transcript chunk, then collapses consecutive chunks that share the same
topic into a single entry - producing a clean, jump-to-able outline like
the doc's example:

    00:00 Introduction
    04:35 Types of Machine Learning
    ...
"""
from __future__ import annotations

from preprocess.chunking import Chunk
from utils.helpers import format_timestamp
from . import client

SYSTEM_PROMPT = (
    "You label the topic of a transcript excerpt in a very short phrase, "
    "as if writing a table of contents entry."
)

LABEL_PROMPT = """\
In 2-6 words, name the main topic being discussed in this transcript
excerpt (like a table-of-contents entry, e.g. "Introduction to Machine
Learning" or "Q&A"). Respond with ONLY the short phrase, nothing else.

Transcript excerpt:
\"\"\"
{chunk_text}
\"\"\"
"""


def extract_timestamps(chunks: list[Chunk], progress_callback=None) -> list[dict]:
    total = max(1, len(chunks))
    labeled = []
    for i, chunk in enumerate(chunks):
        label = client.chat(
            LABEL_PROMPT.format(chunk_text=chunk.text[:1500]),
            system=SYSTEM_PROMPT,
            temperature=0.2,
        ).strip().strip('"')
        labeled.append({"time": format_timestamp(chunk.start), "seconds": chunk.start, "label": label})
        if progress_callback:
            progress_callback((i + 1) / total)

    # Collapse consecutive chunks with (near-)identical labels.
    collapsed: list[dict] = []
    for item in labeled:
        if collapsed and collapsed[-1]["label"].lower() == item["label"].lower():
            continue
        collapsed.append(item)

    return collapsed
