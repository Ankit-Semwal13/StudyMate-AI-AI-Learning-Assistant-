"""
Small shared utilities: timestamp formatting, FFmpeg audio extraction,
JSON persistence, and Markdown/PDF export of the generated notes.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import config


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS (or MM:SS if under an hour)."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def youtube_timestamp_url(url: str, seconds: float) -> str:
    """Build a YouTube deep-link that jumps to a given second, if possible."""
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(seconds)}s"


def extract_audio(input_path: str | Path, output_path: str | Path) -> Path:
    """
    Extract mono 16kHz WAV audio from a video/audio file using FFmpeg.
    This is the "Audio Extraction (FFmpeg)" pipeline step.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        config.FFMPEG_BINARY,
        "-y",  # overwrite output
        "-i", str(input_path),
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz, what Whisper expects
        "-vn",             # no video stream
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found on your PATH. Install it "
            "(e.g. `winget install Gyan.FFmpeg` on Windows) and try again."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to extract audio:\n{result.stderr}")

    return output_path


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def notes_to_markdown(result: dict) -> str:
    """Render the full pipeline result dict as a single Markdown document."""
    lines = [f"# {result.get('title', 'Video Notes')}", ""]

    if result.get("summary"):
        lines += ["## Summary", "", result["summary"], ""]

    if result.get("key_takeaways"):
        lines += ["## Key Takeaways", ""]
        lines += [f"- {kt}" for kt in result["key_takeaways"]]
        lines.append("")

    if result.get("notes_markdown"):
        lines += ["## Notes", "", result["notes_markdown"], ""]

    if result.get("timestamps"):
        lines += ["## Important Timestamps", ""]
        for ts in result["timestamps"]:
            lines.append(f"- `{ts['time']}` — {ts['label']}")
        lines.append("")

    if result.get("action_items"):
        lines += ["## Action Items", ""]
        for item in result["action_items"]:
            lines.append(f"- [ ] {item}")
        lines.append("")

    return "\n".join(lines)


def notes_to_pdf(result: dict, output_path: str | Path) -> Path:
    """Render the pipeline result dict as a simple PDF using fpdf2."""
    from fpdf import FPDF

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text: str, size: int = 16):
        pdf.set_font("Helvetica", "B", size)
        pdf.multi_cell(0, 10, text)
        pdf.ln(2)

    def body(text: str, size: int = 11):
        pdf.set_font("Helvetica", "", size)
        pdf.multi_cell(0, 7, text)
        pdf.ln(1)

    heading(result.get("title", "Video Notes"), 20)

    if result.get("summary"):
        heading("Summary", 14)
        body(result["summary"])

    if result.get("key_takeaways"):
        heading("Key Takeaways", 14)
        for kt in result["key_takeaways"]:
            body(f"- {kt}")

    if result.get("notes_markdown"):
        heading("Notes", 14)
        body(result["notes_markdown"])

    if result.get("timestamps"):
        heading("Important Timestamps", 14)
        for ts in result["timestamps"]:
            body(f"{ts['time']} - {ts['label']}")

    if result.get("action_items"):
        heading("Action Items", 14)
        for item in result["action_items"]:
            body(f"[ ] {item}")

    pdf.output(str(output_path))
    return output_path


def slugify(text: str, max_len: int = 60) -> str:
    keep = [c if c.isalnum() else "_" for c in text.strip().lower()]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:max_len] or "video"
