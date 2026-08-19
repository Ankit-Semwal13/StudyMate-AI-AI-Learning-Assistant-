"""
Tests for the small, brittle text parsers that turn a local LLM's raw
chat completion into structured flashcards/quiz questions. These are
exactly the kind of "the model didn't follow the format perfectly"
scenarios worth locking down with tests, since a 3B local model's output
formatting is never 100% reliable.
"""
from llm.flashcards import _parse_cards
from llm.quiz import _parse_questions


# ---------------- flashcards ----------------

def test_parse_cards_happy_path():
    raw = (
        "Q: What is gravity?\n"
        "A: A force that attracts objects with mass.\n"
        "\n"
        "Q: Who discovered it?\n"
        "A: Isaac Newton.\n"
    )
    cards = _parse_cards(raw)
    assert cards == [
        {"question": "What is gravity?", "answer": "A force that attracts objects with mass."},
        {"question": "Who discovered it?", "answer": "Isaac Newton."},
    ]


def test_parse_cards_ignores_answer_without_preceding_question():
    raw = "A: orphaned answer\nQ: real question\nA: real answer\n"
    cards = _parse_cards(raw)
    assert cards == [{"question": "real question", "answer": "real answer"}]


def test_parse_cards_handles_lowercase_and_extra_whitespace():
    raw = "  q: lowercase question  \n  a: lowercase answer  \n"
    cards = _parse_cards(raw)
    assert cards == [{"question": "lowercase question", "answer": "lowercase answer"}]


def test_parse_cards_empty_input_returns_empty_list():
    assert _parse_cards("") == []
    assert _parse_cards("NONE") == []


def test_parse_cards_incomplete_trailing_question_is_dropped():
    # A question with no matching answer line should not produce a card.
    raw = "Q: What is gravity?\nA: A force.\nQ: dangling question with no answer"
    cards = _parse_cards(raw)
    assert cards == [{"question": "What is gravity?", "answer": "A force."}]


# ---------------- quiz ----------------

_GOOD_QUESTION = (
    "QUESTION: What is 2+2?\n"
    "A) 3\n"
    "B) 4\n"
    "C) 5\n"
    "D) 6\n"
    "ANSWER: B\n"
)


def test_parse_questions_happy_path():
    questions = _parse_questions(_GOOD_QUESTION)
    assert len(questions) == 1
    q = questions[0]
    assert q["question"] == "What is 2+2?"
    assert q["options"] == ["3", "4", "5", "6"]
    assert q["answer_index"] == 1  # "B" -> index 1


def test_parse_questions_multiple_questions_in_one_response():
    raw = _GOOD_QUESTION + "\n" + (
        "QUESTION: What is the capital of France?\n"
        "A) Berlin\n"
        "B) Madrid\n"
        "C) Paris\n"
        "D) Rome\n"
        "ANSWER: C\n"
    )
    questions = _parse_questions(raw)
    assert len(questions) == 2
    assert questions[1]["answer_index"] == 2  # "C" -> index 2


def test_parse_questions_accepts_period_style_option_labels():
    raw = (
        "QUESTION: Pick one\n"
        "A. first\n"
        "B. second\n"
        "C. third\n"
        "D. fourth\n"
        "ANSWER: A\n"
    )
    questions = _parse_questions(raw)
    assert questions[0]["options"] == ["first", "second", "third", "fourth"]
    assert questions[0]["answer_index"] == 0


def test_parse_questions_drops_incomplete_question():
    # Missing option D and no ANSWER line -> should not be included.
    raw = "QUESTION: incomplete\nA) one\nB) two\nC) three\n"
    assert _parse_questions(raw) == []


def test_parse_questions_lowercase_answer_letter():
    raw = (
        "QUESTION: lowercase answer test\n"
        "A) x\nB) y\nC) z\nD) w\n"
        "answer: d\n"
    )
    questions = _parse_questions(raw)
    assert questions[0]["answer_index"] == 3


def test_parse_questions_empty_input_returns_empty_list():
    assert _parse_questions("") == []
