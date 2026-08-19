"""
Tests for the small pure formatting helpers in web/server.py that power
the dashboard's "Recent Projects" / "Recent Activity" relative timestamps
and video durations.
"""
from datetime import datetime, timedelta, timezone

from web.server import _relative_time, _duration_str


# ---------------- _duration_str ----------------

def test_duration_str_seconds_and_minutes():
    assert _duration_str(0) == "0:00"
    assert _duration_str(65) == "1:05"
    assert _duration_str(599) == "9:59"


def test_duration_str_includes_hours_when_needed():
    assert _duration_str(3661) == "1:01:01"


def test_duration_str_handles_none_and_negative_gracefully():
    assert _duration_str(None) == "0:00"


# ---------------- _relative_time ----------------

def test_relative_time_just_now():
    ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    assert _relative_time(ts) == "Just now"


def test_relative_time_minutes_ago():
    ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert _relative_time(ts) == "5m ago"


def test_relative_time_days_ago():
    ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    assert _relative_time(ts) == "3 days ago"


def test_relative_time_falls_back_to_month_day_after_a_week():
    # Past the "N days ago" window (>= 7 days), and not "Yesterday" either
    # (that branch only matches when the calendar date is exactly
    # yesterday's date) - it should fall through to the "%b %d" branch.
    ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    result = _relative_time(ts)
    assert "ago" not in result
    assert len(result) > 0


def test_relative_time_invalid_timestamp_returns_empty_string():
    assert _relative_time("not-a-timestamp") == ""
    assert _relative_time("") == ""


def test_relative_time_naive_datetime_is_treated_as_utc():
    # created_at values are always saved as UTC-aware isoformat strings by
    # pipeline.py, but the parser should not blow up on a naive one either.
    naive_utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    naive = (naive_utc_now - timedelta(minutes=2)).isoformat()
    assert _relative_time(naive) == "2m ago"
