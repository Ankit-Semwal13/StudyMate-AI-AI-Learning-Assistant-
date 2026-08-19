"""
Tests for the pure logic in downloader/youtube.py. The actual network
calls (download_from_url, get_video_title) are exercised via manual/
integration testing instead - they need real yt-dlp + network access.
"""
from downloader.youtube import is_url


def test_is_url_recognizes_http_and_https():
    assert is_url("http://example.com") is True
    assert is_url("https://www.youtube.com/watch?v=abc123") is True


def test_is_url_is_case_insensitive_and_trims_whitespace():
    assert is_url("  HTTPS://example.com  ") is True


def test_is_url_rejects_local_paths():
    assert is_url("C:\\Users\\me\\video.mp4") is False
    assert is_url("/home/me/video.mp4") is False
    assert is_url("video.mp4") is False


def test_is_url_rejects_other_schemes():
    assert is_url("ftp://example.com/file") is False
    assert is_url("www.youtube.com/watch?v=abc123") is False
