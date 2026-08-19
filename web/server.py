"""
FastAPI backend for the StudyMate AI dashboard.

Serves the dashboard UI (Jinja2 templates) and a small JSON API that
drives it, reusing every pipeline module already built for the
Streamlit app (pipeline.py, rag/qa.py, embeddings/, utils/helpers.py) -
this is a new front end on the same backend, not a rewrite.

Run with:  python -m web.server        (from the project root)
or:        uvicorn web.server:app --reload --port 8000

Note: templates are rendered via a plain jinja2.Environment rather than
fastapi.templating.Jinja2Templates - the current fastapi/starlette combo
on this machine hits an internal template-cache TypeError through that
wrapper (a version-compat bug, not a template bug), so this sidesteps it.
"""
from __future__ import annotations

import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

import config
from utils.helpers import (
    format_timestamp,
    load_json,
    save_json,
    notes_to_markdown,
    notes_to_pdf,
)
from embeddings.vector_store import VectorStore
from rag.qa import answer_question
from llm.client import OllamaUnavailableError
from llm.flashcards import generate_flashcards
from llm.quiz import generate_quiz
from preprocess.chunking import Chunk
import pipeline

app = FastAPI(title="StudyMate AI")

WEB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

jinja_env = Environment(
    loader=FileSystemLoader(str(WEB_DIR / "templates")),
    autoescape=select_autoescape(["html"]),
)


def render(name: str, **context) -> HTMLResponse:
    template = jinja_env.get_template(name)
    return HTMLResponse(template.render(**context))


USER_NAME = "Ankit Semwal"
USER_PLAN = "Free Plan"

# --------------------------------------------------------------------------
# In-memory job registry for background video-processing jobs
# --------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _run_pipeline_job(job_id: str, source: str, is_url: bool, title_hint: Optional[str]):
    def on_progress(stage: str, fraction: float):
        _set_job(job_id, stage=stage, fraction=fraction)

    try:
        _set_job(job_id, status="running", stage="Starting...", fraction=0.0)
        result = pipeline.run_pipeline(source, is_url, title_hint=title_hint, progress_callback=on_progress)
        _set_job(job_id, status="done", fraction=1.0, stage="Done!", slug=result["slug"])
    except Exception as exc:
        traceback.print_exc()
        _set_job(job_id, status="error", error=str(exc))


def _load_chunks(slug: str) -> tuple[dict, list[Chunk]]:
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise FileNotFoundError(slug)
    v = load_json(fp)
    chunks = [Chunk(**c) for c in v.get("chunks", [])]
    return v, chunks


def _run_flashcards_job(job_id: str, slug: str):
    try:
        _set_job(job_id, status="running", stage="Generating flashcards...", fraction=0.0)
        v, chunks = _load_chunks(slug)

        def on_progress(fraction: float):
            _set_job(job_id, fraction=fraction)

        cards = generate_flashcards(chunks, progress_callback=on_progress)
        v["flashcards"] = cards
        save_json(v, config.NOTES_DIR / (slug + ".json"))
        _set_job(job_id, status="done", fraction=1.0, stage="Done!", slug=slug)
    except Exception as exc:
        traceback.print_exc()
        _set_job(job_id, status="error", error=str(exc))


def _run_quiz_job(job_id: str, slug: str):
    try:
        _set_job(job_id, status="running", stage="Generating quiz...", fraction=0.0)
        v, chunks = _load_chunks(slug)

        def on_progress(fraction: float):
            _set_job(job_id, fraction=fraction)

        questions = generate_quiz(chunks, progress_callback=on_progress)
        v["quiz"] = questions
        save_json(v, config.NOTES_DIR / (slug + ".json"))
        _set_job(job_id, status="done", fraction=1.0, stage="Done!", slug=slug)
    except Exception as exc:
        traceback.print_exc()
        _set_job(job_id, status="error", error=str(exc))


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def _all_videos() -> list[dict]:
    """Load every processed video's saved notes JSON, newest first."""
    videos = []
    for fp in sorted(config.NOTES_DIR.glob("*.json")):
        try:
            d = load_json(fp)
        except Exception:
            continue
        d["_path"] = str(fp)
        videos.append(d)
    videos.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return videos


def _relative_time(iso_ts: str) -> str:
    try:
        ts = datetime.fromisoformat(iso_ts)
    except Exception:
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - ts
    seconds = delta.total_seconds()
    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        return str(int(seconds // 60)) + "m ago"
    if ts.date() == now.date():
        return "Today - " + ts.strftime("%I:%M %p").lstrip("0")
    yesterday = (now - timedelta(days=1)).date()
    if ts.date() == yesterday:
        return "Yesterday - " + ts.strftime("%I:%M %p").lstrip("0")
    days = delta.days
    if days < 7:
        suffix = "s" if days != 1 else ""
        return str(days) + " day" + suffix + " ago"
    return ts.strftime("%b %d")


def _duration_str(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return str(h) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)
    return str(m) + ":" + str(s).zfill(2)


def _dashboard_context() -> dict:
    videos = _all_videos()
    total_notes = sum(len(v.get("chunks", [])) for v in videos)

    recent_projects = []
    for v in videos[:4]:
        recent_projects.append({
            "slug": v["slug"],
            "title": v["title"],
            "when": _relative_time(v.get("created_at", "")),
            "duration": _duration_str(v.get("duration_seconds", 0)),
            "notes_count": len(v.get("chunks", [])),
            "timestamps_count": len(v.get("timestamps", [])),
            "action_items_count": len(v.get("action_items", [])),
        })

    recent_activity = []
    for v in videos[:6]:
        recent_activity.append({
            "icon": "notes",
            "text": "Generated notes for \"" + v["title"] + "\"",
            "when": _relative_time(v.get("created_at", "")),
            "sort_key": v.get("created_at", ""),
        })
    recent_activity.sort(key=lambda a: a["sort_key"], reverse=True)

    total_flashcards = sum(len(v.get("flashcards", [])) for v in videos)
    total_quiz = sum(len(v.get("quiz", [])) for v in videos)

    return {
        "user_name": USER_NAME,
        "user_plan": USER_PLAN,
        "stats": {
            "videos": len(videos),
            "notes": total_notes,
            "flashcards": total_flashcards,
            "quiz": total_quiz,
        },
        "recent_projects": recent_projects,
        "recent_activity": recent_activity[:6],
        "active_page": "dashboard",
    }


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return render("dashboard.html", **_dashboard_context())


@app.get("/notes", response_class=HTMLResponse)
def notes_list(q: str = ""):
    videos = _all_videos()
    q_norm = q.strip().lower()
    if q_norm:
        videos = [v for v in videos if q_norm in v.get("title", "").lower()]
    for v in videos:
        v["when"] = _relative_time(v.get("created_at", ""))
        v["duration"] = _duration_str(v.get("duration_seconds", 0))
    return render(
        "notes_list.html",
        videos=videos, active_page="notes", search_query=q,
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/analytics", response_class=HTMLResponse)
def analytics():
    videos = _all_videos()

    total_duration = sum(v.get("duration_seconds", 0) for v in videos)
    rows = []
    for v in videos:
        rows.append({
            "title": v["title"],
            "when": _relative_time(v.get("created_at", "")),
            "duration": _duration_str(v.get("duration_seconds", 0)),
            "chapters": len(v.get("chunks", [])),
            "timestamps": len(v.get("timestamps", [])),
            "action_items": len(v.get("action_items", [])),
            "flashcards": len(v.get("flashcards", [])),
            "quiz": len(v.get("quiz", [])),
        })

    return render(
        "analytics.html",
        videos=videos,
        rows=rows,
        stats={
            "videos": len(videos),
            "notes": sum(len(v.get("chunks", [])) for v in videos),
            "flashcards": sum(len(v.get("flashcards", [])) for v in videos),
            "quiz": sum(len(v.get("quiz", [])) for v in videos),
            "total_duration": _duration_str(total_duration),
            "action_items": sum(len(v.get("action_items", [])) for v in videos),
        },
        active_page="analytics",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/notes/{slug}", response_class=HTMLResponse)
def notes_detail(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    v = load_json(fp)
    for ts in v.get("timestamps", []):
        ts.setdefault("time", format_timestamp(ts.get("seconds", 0)))
    return render(
        "notes_detail.html",
        v=v, active_page="notes",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/notes/{slug}/export.md")
def export_markdown(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    v = load_json(fp)
    return PlainTextResponse(notes_to_markdown(v), media_type="text/markdown")


@app.get("/notes/{slug}/export.pdf")
def export_pdf(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    v = load_json(fp)
    out_path = notes_to_pdf(v, config.NOTES_DIR / (slug + ".pdf"))
    return FileResponse(out_path, media_type="application/pdf", filename=slug + ".pdf")


@app.get("/chat", response_class=HTMLResponse)
def chat_picker():
    videos = _all_videos()
    return render(
        "chat_picker.html",
        videos=videos, active_page="chat",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/chat/{slug}", response_class=HTMLResponse)
def chat_detail(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    v = load_json(fp)
    return render(
        "chat_detail.html",
        v=v, active_page="chat",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/flashcards", response_class=HTMLResponse)
def flashcards_picker():
    videos = _all_videos()
    for v in videos:
        v["flashcards_count"] = len(v.get("flashcards", []))
    return render(
        "flashcards_picker.html",
        videos=videos, active_page="flashcards",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/flashcards/{slug}", response_class=HTMLResponse)
def flashcards_detail(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    v = load_json(fp)
    return render(
        "flashcards_detail.html",
        v=v, active_page="flashcards",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/quiz", response_class=HTMLResponse)
def quiz_picker():
    videos = _all_videos()
    for v in videos:
        v["quiz_count"] = len(v.get("quiz", []))
    return render(
        "quiz_picker.html",
        videos=videos, active_page="quiz",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/quiz/{slug}", response_class=HTMLResponse)
def quiz_detail(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    v = load_json(fp)
    return render(
        "quiz_detail.html",
        v=v, active_page="quiz",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page():
    cfg = {
        "OLLAMA_MODEL": config.OLLAMA_MODEL,
        "OLLAMA_HOST": config.OLLAMA_HOST,
        "WHISPER_MODEL_SIZE": config.WHISPER_MODEL_SIZE,
        "WHISPER_LANGUAGE": config.WHISPER_LANGUAGE or "auto-detect",
        "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
        "CHUNK_SIZE_WORDS": config.CHUNK_SIZE_WORDS,
        "CHUNK_OVERLAP_WORDS": config.CHUNK_OVERLAP_WORDS,
        "RAG_TOP_K": config.RAG_TOP_K,
    }
    return render(
        "settings.html",
        cfg=cfg, active_page="settings",
        user_name=USER_NAME, user_plan=USER_PLAN,
    )


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/api/process")
async def api_process(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    job_id = uuid.uuid4().hex[:12]

    if url and url.strip():
        source, is_url, title_hint = url.strip(), True, None
    elif file is not None:
        dest = config.UPLOAD_DIR / file.filename
        dest.write_bytes(await file.read())
        source, is_url, title_hint = str(dest), False, Path(file.filename).stem
    else:
        raise HTTPException(400, "Provide either a YouTube URL or a video file.")

    _set_job(job_id, status="queued", stage="Queued...", fraction=0.0)
    thread = threading.Thread(
        target=_run_pipeline_job, args=(job_id, source, is_url, title_hint), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.post("/api/chat/{slug}")
async def api_chat(slug: str, request: Request):
    body = await request.json()
    question = (body or {}).get("question", "").strip()
    if not question:
        raise HTTPException(400, "question is required")

    store = VectorStore.load(config.VECTOR_DIR / slug)
    if store is None:
        raise HTTPException(404, "No search index for this video")

    try:
        result = answer_question(question, store)
    except OllamaUnavailableError as exc:
        return JSONResponse({"answer": str(exc), "sources": []}, status_code=200)
    return result


@app.get("/api/stats")
def api_stats():
    videos = _all_videos()
    return {
        "videos": len(videos),
        "notes": sum(len(v.get("chunks", [])) for v in videos),
        "flashcards": sum(len(v.get("flashcards", [])) for v in videos),
        "quiz": sum(len(v.get("quiz", [])) for v in videos),
    }


@app.post("/api/flashcards/{slug}")
def api_generate_flashcards(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    job_id = uuid.uuid4().hex[:12]
    _set_job(job_id, status="queued", stage="Queued...", fraction=0.0)
    thread = threading.Thread(target=_run_flashcards_job, args=(job_id, slug), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/api/quiz/{slug}")
def api_generate_quiz(slug: str):
    fp = config.NOTES_DIR / (slug + ".json")
    if not fp.exists():
        raise HTTPException(404, "Video not found")
    job_id = uuid.uuid4().hex[:12]
    _set_job(job_id, status="queued", stage="Queued...", fraction=0.0)
    thread = threading.Thread(target=_run_quiz_job, args=(job_id, slug), daemon=True)
    thread.start()
    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=False)
