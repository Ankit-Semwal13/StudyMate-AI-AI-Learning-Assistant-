"""
Flashcard generation: turns each transcript chunk into a handful of
Q/A study flashcards, grounded only in what the chunk actually says.
"""
from __future__ import annotations

from preprocess.chunking import Chunk
from . import client

SYSTEM_PROMPT = (
    "You are an expert study-aid creator. You write flashcards (a short "
    "question on the front, a concise answer on the back) that test "
    "understanding of facts, definitions, and concepts explicitly stated "
    "in a video transcript. Never invent facts that aren't in the text."
)

CHUNK_PROMPT = """\
Create 3-5 flashcards from this transcript excerpt to help a student
review it. Each flashcard should test one specific fact, definition, or
concept explicitly mentioned - not something you have to infer.

Format exactly like this, one flashcard per pair, nothing else:

Q: <question>
A: <answer>

Q: <question>
A: <answer>

Transcript excerpt:
\"\"\"
{chunk_text}
\"\"\"
"""


def _parse_cards(raw: str) -> list[dict]:
    cards = []
    question = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("Q:"):
            question = line.split(":", 1)[1].strip()
        elif line.upper().startswith("A:") and question:
            answer = line.split(":", 1)[1].strip()
            if question and answer:
                cards.append({"question": question, "answer": answer})
            question = None
    return cards


def generate_flashcards(chunks: list[Chunk], progress_callback=None) -> list[dict]:
    """Returns a list of {"question": str, "answer": str} dicts."""
    all_cards: list[dict] = []
    total = max(1, len(chunks))
    seen_questions = set()

    for i, chunk in enumerate(chunks):
        raw = client.chat(CHUNK_PROMPT.format(chunk_text=chunk.text), system=SYSTEM_PROMPT)
        for card in _parse_cards(raw):
            key = card["question"].lower().strip()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            all_cards.append(card)
        if progress_callback:
            progress_callback((i + 1) / total)

    return all_cards
