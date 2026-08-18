"""
Summary + key-takeaways generation, using a map-reduce approach so it
scales to long videos even with a small local LLM's limited context
window: summarize each chunk first, then summarize the summaries.
"""
from __future__ import annotations

from preprocess.chunking import Chunk
from . import client

SYSTEM_PROMPT = (
    "You are an expert note taker. You write clear, faithful summaries of "
    "video transcripts, using only information present in the text."
)

MAP_PROMPT = """\
Summarize the following part of a video transcript in 2-4 sentences,
capturing only the concrete points actually made.

Transcript part:
\"\"\"
{chunk_text}
\"\"\"
"""

REDUCE_PROMPT = """\
Below are sequential partial summaries of a full video, in order.
Using only this information, write:

1. An OVERALL SUMMARY of the whole video, 4-8 sentences.
2. 3-6 KEY TAKEAWAYS as short bullet points.

Format exactly like this:

SUMMARY:
<paragraph>

KEY TAKEAWAYS:
- <point 1>
- <point 2>
...

Partial summaries:
\"\"\"
{combined}
\"\"\"
"""


def generate_summary(chunks: list[Chunk], progress_callback=None) -> tuple[str, list[str]]:
    """Returns (summary_text, key_takeaways_list)."""
    if not chunks:
        return "", []

    partials = []
    total = max(1, len(chunks) + 1)
    for i, chunk in enumerate(chunks):
        partial = client.chat(MAP_PROMPT.format(chunk_text=chunk.text), system=SYSTEM_PROMPT)
        partials.append(partial.strip())
        if progress_callback:
            progress_callback((i + 1) / total)

    combined = "\n\n".join(partials)
    raw = client.chat(REDUCE_PROMPT.format(combined=combined), system=SYSTEM_PROMPT)
    if progress_callback:
        progress_callback(1.0)

    summary = ""
    takeaways: list[str] = []
    mode = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SUMMARY:"):
            mode = "summary"
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                summary += rest + " "
            continue
        if stripped.upper().startswith("KEY TAKEAWAYS:"):
            mode = "takeaways"
            continue
        if mode == "summary" and stripped:
            summary += stripped + " "
        elif mode == "takeaways" and stripped:
            takeaways.append(stripped.lstrip("-*• ").strip())

    return summary.strip() or raw.strip(), takeaways
