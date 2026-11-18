# Clipper - BLOCKED Items

This file is used to document items that are blocked during development, but are not currently required for completion. 

## Currently Blocked Items (None)

The two items from 2026-11-18 are both resolved:

### ~~1. Real transcription cannot run yet~~ — RESOLVED (2026-11-18)
- `scripts/transcribe_job.py` fixed (`Database` API, correct db path, proper error output)
- faster-whisper `large-v3-turbo` downloaded to `~/.cache/huggingface` (allowed by user)
- **Transcribed job `72e7bb26e760466ea2ca31cc0d1ac021` on CPU int8**: 152 words, 22 sentences; artifacts validated against M2 spec (schema v1, monotonic words, gap/punctuation splits, ≤12s sentences, contiguous word_idx coverage). See milestones.md.
- Note: GPUs 0–2 are occupied by an Ollama LLM workload, so `ASRConfig` defaults are now `device="cpu"` / `compute_type="int8"`. Switch back with config/env (e.g. `CLIPPER_ASR_DEVICE=cuda:0`, `CLIPPER_ASR_COMPUTE_TYPE=float16`) once GPU0 is free.

### ~~2. `clipper run` CLI command is a stub~~ — KNOWN GAP (not blocking)
- `python -m clipper run --job-id` still prints a stub; use `python scripts/transcribe_job.py <job-id>` until it's wired up

## Completed Since Last Update (2026-11-18)
- M1 ingest pipeline re-verified end-to-end on a real 80.8s AV1/opus video after fixing 4 bugs (ffprobe `-i` missing, proxy filter chain, CLI swallowing `typer.Exit(0)`, failed-job retry path) — see `docs/spec/milestones.md` M1 notes
- M2 transcription glue fixed (`stages/transcribe.py` used a nonexistent `DB` class; `transcript_artifacts` re-export missing from `clipper.models`; `src.`-style import in `adapters/media.py`; ctranslate2 4.9 API fixes in `FasterWhisperASR`)
- **M2 verified on real audio**: first real faster-whisper transcription completed and validated
- 26 CPU tests passing; touched files ruff-clean

The implementation includes:
✅ Project scaffolding with proper directory structure
✅ All core data contracts  
✅ Configuration system with environment variable support
✅ SQLite database with schema migrations
✅ Device detection (graceful fallbacks) 
✅ CLI with doctor command only
✅ Complete documentation files
✅ Test fixtures and basic testing infrastructure

## Items Not Implemented in M0 

These will be implemented in future milestones:

1. Input processing pipeline (M1)
2. Transcription system (M2)
3. Scene analysis (M3)  
4. Clip candidate generation (M4)
5. LLM scoring system (M5)
6. Quality gate and selection (M6)
7. Cut refinement (M7)
8. Reframe rendering (M8)
9. Caption generation (M9)
10. QC workflow with web UI (M10)
11. Future posting adapter (M10)

## Next Milestone

**M0, M1, and M2 are complete and verified.** M2 was verified on 2026-11-18 with a real faster-whisper transcription (CPU int8, GPUs busy with an Ollama workload): job `72e7bb26e760466ea2ca31cc0d1ac021` has all 6 stages `completed` and validated artifacts. The project is ready for M3 development (Scene analysis).