"""
Notes generation step ("You are an expert note taker" prompt from the
spec): turns each transcript chunk into a chapter-style notes section with
a heading, key concepts, definitions, and examples.
"""
from __future__ import annotations

from preprocess.chunking import Chunk
from utils.helpers import format_timestamp
from . import client

SYSTEM_PROMPT = (
    "You are an expert note taker helping a student review a video. "
    "You write clear, concise, well-organized study notes from a raw "
    "speech transcript. Only use information present in the transcript; "
    "never invent facts."
)

CHUNK_PROMPT_TEMPLATE = """\
Below is one part of a longer video transcript (it may start or end
mid-sentence - that's fine, just focus on the content).

Write study notes for this part as:
1. A short chapter heading (3-8 words) that names the main topic.
2. 4-8 concise bullet points covering key concepts, definitions, and
   examples actually mentioned in the text.

Format your answer exactly like this (plain text, no markdown symbols
other than the dash for bullets):

HEADING: <heading>
- <bullet 1>
- <bullet 2>
...

Transcript part:
\"\"\"
{chunk_text}
\"\"\"
"""


def _generate_chapter(chunk: Chunk) -> tuple[str, list[str]]:
    raw = client.chat(
        CHUNK_PROMPT_TEMPLATE.format(chunk_text=chunk.text),
        system=SYSTEM_PROMPT,
    )
    heading = f"Chapter {chunk.index + 1}"
    bullets: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("HEADING:"):
            heading = line.split(":", 1)[1].strip() or heading
        elif line.startswith(("-", "*", "•")):
            bullets.append(line.lstrip("-*• ").strip())
    if not bullets:
        # Fallback: treat any non-heading lines as bullets so we never
        # silently drop content if the model didn't follow the format.
        bullets = [
            line.strip("-* ").strip()
            for line in raw.splitlines()
            if line.strip() and not line.upper().startswith("HEADING:")
        ]
    return heading, bullets


def generate_notes(chunks: list[Chunk], progress_callback=None) -> str:
    """
    Generate the full "Organized Notes" markdown document: one chapter per
    transcript chunk, each with a heading, bullets, and the timestamp where
    it starts in the video.
    """
    sections = []
    total = max(1, len(chunks))
    for i, chunk in enumerate(chunks):
        heading, bullets = _generate_chapter(chunk)
        ts = format_timestamp(chunk.start)
        section_lines = [f"### {heading}  _(starts at {ts})_", ""]
        section_lines += [f"- {b}" for b in bullets]
        sections.append("\n".join(section_lines))
        if progress_callback:
            progress_callback((i + 1) / total)

    return "\n\n---\n\n".join(sections)
