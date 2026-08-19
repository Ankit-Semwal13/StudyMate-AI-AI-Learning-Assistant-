# 🎬 Video Note Extractor AI

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

Convert hours of video into structured notes in minutes.

Paste a YouTube link or upload a video (lecture, meeting, Zoom recording)
and get back:

1. **Organized Notes** — chapter-style headings, bullets, key concepts
2. **Important Timestamps** — jump straight to the part you need
3. **Action Items** — tasks, assignments, deadlines mentioned in the video
4. **Key Takeaways / Summary**
5. **Ask AI** — chat with the video using Retrieval-Augmented Generation (RAG)

Everything runs **100% free and local** — no API keys, no per-minute costs:

| Component     | Technology                                   |
|----------------|-----------------------------------------------|
| Frontend       | FastAPI + Jinja2 + vanilla JS (the StudyMate AI dashboard) |
| Video Download | yt-dlp                                         |
| Audio Extract  | FFmpeg                                         |
| Speech-to-Text | openai-whisper (local Whisper)                 |
| LLM            | Ollama (local — e.g. Llama 3.2, Gemma 2, Phi-3)|
| Embeddings     | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Vector DB      | FAISS                                          |
| Notes Storage  | local JSON files (`notes/`, `transcripts/`)    |

## Project layout

```
video-note-extractor/
├── web/server.py            # FastAPI dashboard (entry point - run this)
│   ├── templates/             # Jinja2 pages (dashboard, notes, chat, flashcards, quiz, settings)
│   └── static/                 # style.css, app.js
├── pipeline.py               # Orchestrates the full video-processing pipeline
├── config.py                  # All settings (env-overridable)
├── requirements.txt
├── downloader/youtube.py      # yt-dlp download + local upload handling
├── speech/whisper.py          # openai-whisper transcription
├── preprocess/chunking.py     # transcript -> overlapping chunks
├── embeddings/vector_store.py    # sentence-transformers + FAISS
├── llm/
│   ├── client.py                # Ollama wrapper
│   ├── notes.py                  # chapter notes generation
│   ├── summary.py                # summary + key takeaways
│   ├── action_items.py           # action item extraction
│   ├── timestamps.py             # timestamp/topic labeling
│   ├── flashcards.py             # flashcard (Q/A) generation
│   └── quiz.py                   # multiple-choice quiz generation
├── rag/qa.py                  # RAG question answering
├── utils/helpers.py            # ffmpeg, formatting, export (MD/PDF)
├── tests/                      # pytest unit tests for the pure logic
├── requirements-dev.txt        # requirements.txt + pytest
├── LICENSE                     # MIT
├── uploads/ transcripts/ notes/ vectors/   # local data (gitignored)
```

## 1. Prerequisites (one-time setup)

You need three free things installed on your machine:

### a) Python packages

```bash
pip install -r requirements.txt
```

### b) FFmpeg (system binary, used for audio extraction)

Windows:
```bash
winget install Gyan.FFmpeg
```
(or `choco install ffmpeg`, or download from https://ffmpeg.org and add it to PATH)

Verify with:
```bash
ffmpeg -version
```

### c) Ollama (local LLM server, free, no API key)

1. Download & install from **https://ollama.com**
2. Pull a free model (pick one that fits your RAM):

```bash
ollama pull llama3.2      # ~2GB, good default
# or a lighter option:
ollama pull gemma2:2b      # ~1.6GB, faster on CPU-only machines
```

3. Ollama runs as a background service after install — no need to
   manually start a server. Verify with:

```bash
ollama list
```

That's it — no OpenAI/Anthropic API key needed anywhere.

## 2. Run the app

**Use this one - it's the full app.**

```bash
python -m web.server
```

Open **http://localhost:8000**. This is the StudyMate AI dashboard:
video processing, Notes (Summary/Timestamps/Action Items), Ask AI chat,
Flashcards, Quiz, and Settings, all in one place.

1. Click **Upload Video** or **Paste YouTube Link** from the dashboard.
2. Watch the progress bar through download → transcript → notes.
3. Open the processed video from **Recent Projects** or the **Notes** tab
   to read its Summary/Notes/Timestamps/Action Items, chat with it under
   **AI Chat**, or generate **Flashcards**/**Quiz** for it on demand.

## Configuration

Copy `.env.example` to `.env` to override any default (model choice,
Whisper size, chunk size, etc.) without touching the code:

```bash
cp .env.example .env
```

Notable knobs:
- `WHISPER_MODEL_SIZE` — `tiny`/`base`/`small`/`medium`/`large-v3`. Bigger
  = more accurate but slower. `base` is a good CPU default.
- `OLLAMA_MODEL` — any model you've pulled with `ollama pull <name>`.
- `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` — how the transcript is split
  before embedding/summarizing (defaults: 700 / 100, per the spec).

## How it works (pipeline)

```
YouTube URL / Upload
        │
        ▼
  Video Downloader (yt-dlp)
        │
        ▼
  Audio Extraction (FFmpeg)
        │
        ▼
  Speech-to-Text (faster-whisper)
        │
        ▼
  Transcript → Chunking (overlapping word windows + timestamps)
        │
   ┌────┴─────┐
   ▼          ▼
Embeddings   LLM pipeline (Ollama)
   │        ├─ Notes generator
   ▼        ├─ Summary + key takeaways
FAISS       ├─ Timestamp/topic labeling
index       └─ Action item extraction
   │
   ▼
RAG Question Answering (Ask AI tab)
```

## Notes on this build

- The pipeline modules (`pipeline.py`, `downloader/`, `speech/`, `llm/`,
  `rag/`, ...) are fully decoupled from the UI - `web/server.py` is a
  FastAPI layer on top of them, so the same modules could power another
  frontend later without changes.
- Flashcards and Quiz are generated **on demand per video** (not during
  the main processing run, to keep that from getting even slower) -
  visit a video's Flashcards or Quiz page and click Generate.
- "Advanced Features" from the original spec that are **not yet built**:
  AI mind map, speaker identification, meeting minutes, calendar
  integration, multi-language support, email notes. Markdown/PDF export,
  Flashcards, and Quiz **are** included.
- First run will download the Whisper model weights (`base` ≈ 145 MB) and
  the embedding model (`bge-small-en-v1.5` ≈ 130 MB) automatically from
  Hugging Face and cache them locally — only needs internet the first time.

## Running the tests

The pure logic (chunking math, timestamp formatting, the flashcard/quiz
text parsers, slugify, etc.) has unit test coverage under `tests/`. These
don't need Ollama, Whisper, or FFmpeg running - they're fast, isolated
tests of the actual parsing/formatting logic:

```bash
pip install -r requirements-dev.txt
pytest
```

Pipeline steps that need real external tools (FFmpeg, yt-dlp downloads,
Whisper transcription, Ollama calls) are covered by manual/integration
testing instead, since mocking all of them wouldn't actually prove much.

## Troubleshooting

- **"Could not reach Ollama model..."** → Make sure the Ollama app is
  running and you've pulled the model (`ollama pull llama3.2`).
- **"FFmpeg was not found on your PATH"** → Install FFmpeg (see above)
  and restart your terminal.
- **YouTube download fails with `HTTP Error 403`** → The downloader
  already tries multiple audio formats and automatically retries the
  whole sequence up to `DOWNLOAD_MAX_RETRIES` times (default 3, waiting
  `DOWNLOAD_RETRY_DELAY_SECONDS` x attempt between tries - default 15s,
  30s). If it still fails after all retries, YouTube is very likely
  rate-limiting your IP from too many requests in a short window - wait
  a few minutes and try again, or process a local file in the meantime.
  Also worth trying: `pip install -U yt-dlp` (it's updated frequently to
  keep up with YouTube changes).
- **Slow transcription** → Use a smaller Whisper model
  (`WHISPER_MODEL_SIZE=tiny` or `base`) if you're on CPU only.

## License

MIT - see [LICENSE](LICENSE).
