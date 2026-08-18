"""
Quiz generation: turns each transcript chunk into a couple of
multiple-choice questions (4 options, exactly one correct), grounded
only in what the chunk actually says.
"""
from __future__ import annotations

from preprocess.chunking import Chunk
from . import client

SYSTEM_PROMPT = (
    "You are an expert quiz writer. You write multiple-choice questions "
    "that test understanding of facts, definitions, and concepts "
    "explicitly stated in a video transcript. Exactly one option must be "
    "correct. Never invent facts that aren't in the text."
)

CHUNK_PROMPT = """\
Create 2 multiple-choice quiz questions from this transcript excerpt.
Each question must have exactly 4 options labeled A-D, with exactly one
correct answer.

Format exactly like this, nothing else:

QUESTION: <question text>
A) <option>
B) <option>
C) <option>
D) <option>
ANSWER: <A, B, C, or D>

QUESTION: <question text>
A) <option>
B) <option>
C) <option>
D) <option>
ANSWER: <A, B, C, or D>

Transcript excerpt:
\"\"\"
{chunk_text}
\"\"\"
"""

_LETTER_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _parse_questions(raw: str) -> list[dict]:
    questions = []
    current = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("QUESTION:"):
            if current and len(current["options"]) == 4 and current.get("answer_index") is not None:
                questions.append(current)
            current = {"question": line.split(":", 1)[1].strip(), "options": [], "answer_index": None}
        elif current is not None and len(line) > 2 and line[0] in "ABCDabcd" and line[1] in ").:":
            current["options"].append(line[2:].strip())
        elif current is not None and upper.startswith("ANSWER:"):
            letter = line.split(":", 1)[1].strip().upper()[:1]
            current["answer_index"] = _LETTER_INDEX.get(letter)

    if current and len(current["options"]) == 4 and current.get("answer_index") is not None:
        questions.append(current)

    return questions


def generate_quiz(chunks: list[Chunk], progress_callback=None) -> list[dict]:
    """Returns a list of {"question": str, "options": [4 str], "answer_index": int}."""
    all_questions: list[dict] = []
    total = max(1, len(chunks))
    seen = set()

    for i, chunk in enumerate(chunks):
        raw = client.chat(CHUNK_PROMPT.format(chunk_text=chunk.text), system=SYSTEM_PROMPT)
        for q in _parse_questions(raw):
            key = q["question"].lower().strip()
            if key in seen:
                continue
            seen.add(key)
            all_questions.append(q)
        if progress_callback:
            progress_callback((i + 1) / total)

    return all_questions
