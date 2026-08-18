"""
Video acquisition step: download a YouTube (or any yt-dlp supported URL)
video, or register an already-uploaded local file (upload / Zoom recording /
lecture recording). Output is always a local video/audio file path that the
rest of the pipeline (audio extraction -> transcription) can consume.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import config


# Format strings to try in order. YouTube intermittently 403s specific
# audio itags/CDN edges (observed: m4a blocked while webm/opus for the
# same video succeeded moments later) - trying a couple of alternatives
# before giving up makes downloads noticeably more reliable in practice.
_FORMAT_FALLBACKS = [
    "bestaudio[ext=m4a]/bestaudio/best",
    "bestaudio[ext=webm]/bestaudio/best",
    "bestaudio/best",
]


def _try_all_formats(url: str, output_dir: Path) -> tuple[Optional[Path], Optional[Exception]]:
    """One pass through _FORMAT_FALLBACKS. Returns (path, None) on success,
    or (None, last_exception) if every format failed."""
    import yt_dlp

    last_error: Optional[Exception] = None
    for fmt in _FORMAT_FALLBACKS:
        ydl_opts = {
            # We only ever need the audio track (speech-to-text), so grab
            # an audio-only stream directly rather than a full video+audio
            # one. This is faster/lighter, and sidesteps YouTube's
            # stricter, frequently-403ing restrictions on progressive
            # video formats.
            "format": fmt,
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filepath = ydl.prepare_filename(info)

            path = Path(filepath)
            if not path.exists():
                raise RuntimeError(f"yt-dlp reported success but file not found: {path}")
            return path, None
        except Exception as exc:  # noqa: BLE001 - genuinely want to try the next format
            last_error = exc
            continue

    return None, last_error


def download_from_url(
    url: str,
    output_dir: Path | str = config.UPLOAD_DIR,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Download a video from a URL (YouTube, or anything yt-dlp supports) using
    yt-dlp. Returns the local path to the downloaded file.

    Tries a few different audio format preferences in sequence, since
    YouTube sometimes 403s one specific itag/CDN edge while others for the
    same video work fine moments later. If every format fails, waits and
    retries the whole sequence (YouTube's blocks are frequently transient -
    a rate-limit or a bad CDN edge that clears up within seconds to
    minutes), up to config.DOWNLOAD_MAX_RETRIES times.

    `progress_callback`, if given, is called with a short human-readable
    status string before each retry wait (e.g. to surface in a UI).
    """
    try:
        import yt_dlp  # noqa: F401 - import check only; used inside _try_all_formats
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Run `pip install yt-dlp` first."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for attempt in range(1, config.DOWNLOAD_MAX_RETRIES + 1):
        path, last_error = _try_all_formats(url, output_dir)
        if path is not None:
            return path

        if attempt < config.DOWNLOAD_MAX_RETRIES:
            delay = config.DOWNLOAD_RETRY_DELAY_SECONDS * attempt  # linear backoff
            if progress_callback:
                progress_callback(
                    f"Download blocked (attempt {attempt}/{config.DOWNLOAD_MAX_RETRIES}) - "
                    f"retrying in {delay}s..."
                )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to download audio after {config.DOWNLOAD_MAX_RETRIES} attempts "
        f"(each trying multiple formats). This usually means YouTube is "
        f"rate-limiting this network right now - wait a few minutes and try "
        f"again. Last error: {last_error}"
    ) from last_error


def get_video_title(url: str) -> Optional[str]:
    """Best-effort fetch of the video's title without downloading it."""
    try:
        import yt_dlp
    except ImportError:
        return None

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title")
    except Exception:
        return None


def register_local_file(source_path: str | Path, output_dir: Path | str = config.UPLOAD_DIR) -> Path:
    """
    Copy an already-uploaded local file (video upload, Zoom/meeting/lecture
    recording) into the uploads directory so the rest of the pipeline can
    treat every input source uniformly.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / source_path.name
    if source_path.resolve() != dest.resolve():
        shutil.copy2(source_path, dest)
    return dest


def is_url(source: str) -> bool:
    return source.strip().lower().startswith(("http://", "https://"))
