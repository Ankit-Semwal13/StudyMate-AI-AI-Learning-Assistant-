"""
Tests for preprocess/chunking.py - the sliding-window chunker that turns
Whisper segments into overlapping word chunks for embedding/LLM input.
"""
from preprocess.chunking import chunk_transcript
from speech.whisper import Segment


def test_empty_segments_returns_no_chunks():
    assert chunk_transcript([]) == []


def test_segment_with_no_words_is_skipped():
    # A segment whose text is only whitespace contributes no words.
    segments = [Segment(start=0.0, end=1.0, text="   ")]
    assert chunk_transcript(segments) == []


def test_single_short_segment_becomes_one_chunk():
    segments = [Segment(start=0.0, end=5.0, text="hello world this is a test")]
    chunks = chunk_transcript(segments, chunk_size_words=10, overlap_words=2)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world this is a test"
    assert chunks[0].index == 0
    assert chunks[0].start == 0.0


def test_sliding_window_overlap_and_boundaries():
    # 10 words spread evenly over 10 seconds -> each word is exactly 1s long.
    words = "one two three four five six seven eight nine ten"
    segments = [Segment(start=0.0, end=10.0, text=words)]

    chunks = chunk_transcript(segments, chunk_size_words=4, overlap_words=1)

    # stride = chunk_size_words - overlap_words = 3, so we expect chunks
    # starting at word index 0, 3, 6.
    assert [c.text for c in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]

    # Consecutive chunks overlap by exactly one word ("four", then "seven").
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]
    assert chunks[1].text.split()[-1] == chunks[2].text.split()[0]

    # Chunk indices increment in order.
    assert [c.index for c in chunks] == [0, 1, 2]

    # Timestamps: word i spans [i, i+1) seconds, so chunk 0 covers 0..4,
    # chunk 1 covers 3..7, chunk 2 covers 6..10.
    assert chunks[0].start == 0.0 and chunks[0].end == 4.0
    assert chunks[1].start == 3.0 and chunks[1].end == 7.0
    assert chunks[2].start == 6.0 and chunks[2].end == 10.0


def test_multiple_segments_are_concatenated_before_chunking():
    segments = [
        Segment(start=0.0, end=2.0, text="alpha beta"),
        Segment(start=2.0, end=4.0, text="gamma delta"),
    ]
    chunks = chunk_transcript(segments, chunk_size_words=4, overlap_words=0)
    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta gamma delta"


def test_overlap_larger_than_chunk_size_still_makes_progress():
    # stride is clamped to at least 1 word so the loop can never get stuck,
    # even with a nonsensical overlap >= chunk_size.
    words = " ".join(f"w{i}" for i in range(20))
    segments = [Segment(start=0.0, end=20.0, text=words)]
    chunks = chunk_transcript(segments, chunk_size_words=5, overlap_words=99)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 5 for c in chunks)


def test_chunk_to_dict_round_trips():
    segments = [Segment(start=0.0, end=1.0, text="hi there")]
    chunk = chunk_transcript(segments, chunk_size_words=10, overlap_words=0)[0]
    d = chunk.to_dict()
    assert d == {"index": 0, "text": "hi there", "start": 0.0, "end": 1.0}
