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
- `python -m clipper run --job-id` still prints a stub; use the per-job scripts instead: `scripts/transcribe_job.py` (M2), `scripts/analyze_job.py` (M3), `scripts/candidates_job.py` (M4)

## Progress Update (2026-11-19)
- **M0, M1, M2, M3a, M3b, M4a, M4b complete.** M4 (clip candidates) is stage-complete:
  - M4a: candidate-generation foundations (contract, config `clips.niche_keywords`, pure `generate_candidates` logic, fixtures, 41 tests)
  - M4b: `stages/candidates.py` (`candidates` stage, `candidates.json` artifact, `candidates` table upsert with `score_json = NULL` for M5) + `scripts/candidates_job.py`; 14 stage tests
  - Verified end-to-end on real media (job `72e7bb26e760466ea2ca31cc0d1ac021`): 4 pre-ranked candidates, stage skip-deterministic, re-run <1s
- 113 CPU tests passing (`pytest -m "not gpu"`); `ruff check .` clean
- **Next: M5 (LLM scoring into `candidates.score_json`) — awaiting instructions.**

The implementation includes:
✅ Project scaffolding with proper directory structure
✅ All core data contracts  
✅ Configuration system with environment variable support
✅ SQLite database with schema migrations
✅ Device detection (graceful fallbacks) 
✅ CLI with doctor command only
✅ Complete documentation files
✅ Test fixtures and basic testing infrastructure

## Items Not Implemented Yet 

These will be implemented in future milestones:

5. LLM scoring system (M5) — next
6. Quality gate and selection (M6)
7. Cut refinement (M7)
8. Reframe rendering (M8)
9. Caption generation (M9)
10. QC workflow with web UI (M10)
11. Future posting adapter (M10)

(Completed since 2026-11-18: M1 re-verified on a real 80.8s video after 4 bug fixes; M2 transcription glue fixed + verified on real audio; M3a/M3b analysis foundations + stage; M4a/M4b candidate generation + stage — see `docs/spec/milestones.md`.)

## Next Milestone

**M0–M4 are complete and verified (M4 = candidate generation + `candidates` stage, `score_json` awaiting M5).** The project is ready for M5 development (LLM scoring & selection).