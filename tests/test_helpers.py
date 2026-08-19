"""
Tests for utils/helpers.py - timestamp formatting, YouTube deep-links,
slugify, and Markdown export. (extract_audio/notes_to_pdf touch FFmpeg /
the filesystem and are exercised via manual/integration testing instead.)
"""
from utils.helpers import (
    format_timestamp,
    youtube_timestamp_url,
    slugify,
    notes_to_markdown,
)


# ---------------- format_timestamp ----------------

def test_format_timestamp_under_a_minute():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(5) == "00:05"


def test_format_timestamp_minutes_and_seconds():
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(599) == "09:59"


def test_format_timestamp_includes_hours_when_needed():
    assert format_timestamp(3661) == "01:01:01"
    assert format_timestamp(7325) == "02:02:05"


def test_format_timestamp_rounds_to_nearest_second():
    # round(59.6) == 60 -> rolls over into the next minute
    assert format_timestamp(59.6) == "01:00"


def test_format_timestamp_clamps_negative_to_zero():
    assert format_timestamp(-5) == "00:00"


# ---------------- youtube_timestamp_url ----------------

def test_youtube_timestamp_url_appends_query_param():
    url = "https://www.youtube.com/watch?v=abc123"
    assert youtube_timestamp_url(url, 90) == "https://www.youtube.com/watch?v=abc123&t=90s"


def test_youtube_timestamp_url_adds_question_mark_if_missing():
    url = "https://youtu.be/abc123"
    assert youtube_timestamp_url(url, 5) == "https://youtu.be/abc123?t=5s"


def test_youtube_timestamp_url_truncates_fractional_seconds():
    url = "https://youtu.be/abc123"
    assert youtube_timestamp_url(url, 12.9) == "https://youtu.be/abc123?t=12s"


def test_youtube_timestamp_url_empty_url_returns_empty_string():
    assert youtube_timestamp_url("", 10) == ""


# ---------------- slugify ----------------

def test_slugify_lowercases_and_replaces_punctuation():
    assert slugify("Hello, World!!") == "hello_world"


def test_slugify_collapses_repeated_underscores():
    assert slugify("a --- b") == "a_b"


def test_slugify_strips_leading_and_trailing_underscores():
    assert slugify("!!!wow!!!") == "wow"


def test_slugify_falls_back_to_video_when_nothing_left():
    assert slugify("") == "video"
    assert slugify("!!!") == "video"


def test_slugify_truncates_to_max_len():
    long_title = "a" * 100
    result = slugify(long_title, max_len=10)
    assert result == "a" * 10


# ---------------- notes_to_markdown ----------------

def test_notes_to_markdown_includes_all_sections():
    result = {
        "title": "My Video",
        "summary": "A short summary.",
        "key_takeaways": ["Point one", "Point two"],
        "notes_markdown": "### Chapter 1\n- detail",
        "timestamps": [{"time": "00:00", "label": "Intro"}],
        "action_items": ["Do the thing"],
    }
    md = notes_to_markdown(result)
    assert "# My Video" in md
    assert "## Summary" in md and "A short summary." in md
    assert "- Point one" in md and "- Point two" in md
    assert "### Chapter 1" in md
    assert "`00:00` — Intro" in md
    assert "- [ ] Do the thing" in md


def test_notes_to_markdown_omits_empty_sections():
    result = {"title": "Bare Video"}
    md = notes_to_markdown(result)
    assert md.startswith("# Bare Video")
    assert "## Summary" not in md
    assert "## Key Takeaways" not in md
    assert "## Action Items" not in md
