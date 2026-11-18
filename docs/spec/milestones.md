# Milestone Specifications

## M0 - Scaffold & CLI (Complete)
- Project scaffolding with all directory structure
- Basic config system using pydantic-settings
- SQLite DB with basic migrations
- Core data models (contracts)
- CLI with doctor command only
- Device detection and locking
- Testing framework setup

## M1 - Input processing pipeline (Complete)
- File watching/processing
- Proxy video generation
- Audio extraction 
- Metadata extraction
- Job state management

M1 re-verification (2026-11-18, first real-video run):
The 4 latent bugs below all surfaced only when running a real video (test_media/source.mp4) and are now fixed:
- `FFmpegMediaAdapter.probe()` never passed `-i <path>` to ffprobe, so the probe stage always crashed; fixed
- `FFmpegMediaAdapter.make_proxy()` passed `scale=...` filters as top-level CLI options; now a proper `-vf "scale=960:540:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"` chain
- `cli.py ingest` caught its own `typer.Exit(0)` with `except Exception` (printed `Error: 0`, exited 1); added `except typer.Exit: raise`
- Failed jobs could never be retried: the failed branch never reset stages, and the `INSERT OR IGNORE` / `rowcount == 0` early-return bailed before any stage ran. Now a `failed` job reuses its job_id, clears its stages, resets state to `queued`, re-runs (CLI prints "Retrying failed job")
- Verified end-to-end on a real 80.8s 1280x720 AV1/opus video: ingest/probe/proxy/audio stages all `completed`

## M2a - Transcript Foundations (Complete)
- CPU-only transcript contracts (WordsArtifact, SentencesArtifact)
- Deterministic sentence building logic with punctuation/gap/sentence duration rules
- Fixture ASR adapter for testing without GPU or model downloads
- Stage names: transcribe and sentences

## M2b - Real ASR Adapter and Transcription Stages (Complete)
- FasterWhisperASR adapter with lazy loading from config  
- get_adapter() factory function
- Stage implementation: transcribe and sentences
- Manual transcription script (scripts/transcribe_job.py)
- Stage skips completed work based on DB state

M2 progress notes:
- Sentence builder (`src/clipper/sentences.py`) fixed: gap splits (>0.6s) now close the sentence before the word after the gap; punctuation and gap checks are independent; word_idx tracking corrected
- tests/test_sentences.py: 6/6 passing, CPU-only

M2 verification on real audio (2026-11-18) — COMPLETE:
- Fixed `stages/transcribe.py` (nonexistent `DB` import, `DB(conn)`, CWD-based paths, wrong stage failure marking) and `scripts/transcribe_job.py` (same API bugs + wrong db path)
- Added `device` / `compute_type` to `ASRConfig`; `clipper.models` now re-exports `WordsArtifact` / `SentencesArtifact`
- GPU0–2 all occupied by an Ollama LLM (user instruction: do not test on GPUs) → defaults set to `device="cpu"`, `compute_type="int8"` (recover `cuda:0` / `float16` via config/env when GPU0 frees up)
- ctranslate2 4.9 / faster-whisper 1.2.1 fixes in `FasterWhisperASR`: pass `cpu_threads` (intra_threads=0 rejected) and an int `device_index` (None rejected on CPU)
- `large-v3-turbo` downloaded to HF cache (user-approved)
- Transcribed job `72e7bb26e760466ea2ca31cc0d1ac021` (test_media/source.mp4, 80.8s): 152 words / 22 sentences, ~1 min CPU runtime
- Artifact validation passed: schema v1 exact field sets, monotonic positive-duration words, all inter-word gaps >0.6s correspond to splits, all sentences ≤12.0s, contiguous word_idx tiling of [0, 151], sentence boundaries match word start/end
- Re-run is idempotent (skips completed transcribe stage, 0.2s), all 6 stages `completed`, job `queued`

## Current Status (2026-11-18)
- M0, M1: complete and verified end-to-end on real media
- M2: complete and verified — first real faster-whisper transcription done and validated (26 CPU tests green, touched files ruff-clean)

Next: start M3 (Scene analysis) per your instructions.
Known minor gaps: `clipper run` CLI command is still a stub (use `python scripts/transcribe_job.py <job-id>`); a handful of pre-existing E501s remain in untouched files (db.py, cli.py, etc.)

## M2 - Transcription
- ASR integration (faster-whisper)
- Word timestamps
- VAD filtering
- Audio quality checks

## M3 - Scene analysis  
- PySceneDetect shot detection
- Silence detection
- RMS energy analysis
- Face detection on CPU
- Face tracking across frames

## M4 - Clip candidates generation
- Candidate window generation
- Heuristic scoring
- Candidate storage and retrieval
- Window filtering

## M5 - LLM scoring
- LLM integration (OpenAI-compatible)
- Batch processing 
- Retry logic
- Scoring result storage

## M6 - Quality gate & selection
- Weighted scoring system
- Quality thresholds
- Duplicate detection
- Final clip selection

## M7 - Cut refinement
- In/out timing optimization
- Word boundary snapping
- Silence and shot boundary snapping
- Cut validation 

## M8 - Reframe rendering 
- Original video frame server
- Frame resolution conversion
- CRF 14 encoding

## M9 - Caption generation/composition
- ASS subtitle generation  
- Karaoke effects
- Font handling
- Caption positioning
- Loudnorm audio processing

## M10 - QC workflow + Outbox
- Web UI with FastAPI/Jinja2/HTMX
- Preview functionality
- Approval/rejection workflow
- Layout override capability
- Export system 
- Future posting adapter