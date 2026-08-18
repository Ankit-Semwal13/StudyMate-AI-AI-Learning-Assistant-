"""
Action item extraction: scans each transcript chunk for explicitly
mentioned tasks, assignments, deadlines, or instructions ("read chapter 3",
"submit the project by Friday", "install TensorFlow", ...).
"""
from __future__ import annotations

from preprocess.chunking import Chunk
from . import client

SYSTEM_PROMPT = (
    "You extract concrete action items from a video/meeting transcript. "
    "An action item is something the listener/attendee is explicitly told "
    "to do (an assignment, a task, an install step, a deadline, a "
    "follow-up). Do not invent action items that aren't stated."
)

CHUNK_PROMPT = """\
Read this part of a transcript and list any explicit action items
mentioned (tasks, assignments, deadlines, instructions to do something).
One per line, starting with "- ". If there are none, respond with exactly:
NONE

Transcript part:
\"\"\"
{chunk_text}
\"\"\"
"""

DEDUPE_PROMPT = """\
Here is a raw list of action items pulled from different parts of a video,
which may contain duplicates or near-duplicates. Clean it up into a final
list of distinct, clearly-worded action items, one per line starting with
"- ". If the list is empty, respond with exactly: NONE

Raw list:
\"\"\"
{raw_list}
\"\"\"
"""


def extract_action_items(chunks: list[Chunk], progress_callback=None) -> list[str]:
    raw_items: list[str] = []
    total = max(1, len(chunks) + 1)
    for i, chunk in enumerate(chunks):
        raw = client.chat(CHUNK_PROMPT.format(chunk_text=chunk.text), system=SYSTEM_PROMPT)
        if raw.strip().upper() != "NONE":
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "•")):
                    raw_items.append(line.lstrip("-*• ").strip())
        if progress_callback:
            progress_callback((i + 1) / total)

    if not raw_items:
        if progress_callback:
            progress_callback(1.0)
        return []

    cleaned_raw = client.chat(
        DEDUPE_PROMPT.format(raw_list="\n".join(f"- {i}" for i in raw_items)),
        system=SYSTEM_PROMPT,
    )
    if progress_callback:
        progress_callback(1.0)

    if cleaned_raw.strip().upper() == "NONE":
        return []

    items = [
        line.lstrip("-*• ").strip()
        for line in cleaned_raw.splitlines()
        if line.strip().startswith(("-", "*", "•"))
    ]
    return items or raw_items
