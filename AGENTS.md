# Clipper Project Rules

## Development Environment
- Python 3.11
- pydantic v2
- typer
- ruff
- pytest

## Architecture
- All data contracts live ONLY in `src/clipper/models/`
- All GPU/ffmpeg/network/model access must go through `adapters/`
- Every adapter must have a CPU-only Fixture/Mock for tests
- Pure logic must be separated from I/O
- Tests must never require GPU, CUDA, live LLM, model downloads, or root

## Constraints
- No new dependencies without asking first
- No sudo/systemd/model downloads
- One milestone at a time only
- Blocked after 3 attempts -> write BLOCKED.md and ask
- Definition of done: acceptance commands pass, ruff clean, tests green

## Future Milestones (M0-M10)
M0 - Scaffold & CLI
M1 - Input processing pipeline
M2 - Transcription
M3 - Scene analysis  
M4 - Clip candidates generation
M5 - LLM scoring
M6 - Quality gate & selection
M7 - Cut refinement
M8 - Reframe rendering
M9 - Caption generation/composition
M10 - QC workflow + Outbox

## Testing Rules
- All tests must run in CPU-only environment
- No GPU/CUDA dependencies
- No live LLM calls
- No model downloads
- No root privileges required

## Progress Update

M1 - Input processing pipeline is now complete.
The following components have been implemented:
- Ingest pipeline with probe, proxy, and audio extraction stages  
- FFmpeg-based media adapter with error handling
- Database migration v2 schema
- CLI ingest command functionality 
- Content hash idempotency system

M2a - Transcript Foundations
The following components have been implemented:
- CPU-only transcript contracts (WordsArtifact, SentencesArtifact)
- Deterministic sentence builder logic 
- Fixture ASR adapter for testing
- All tests run in CPU-only environment without GPU or model dependencies

M2b - Real ASR Adapter and Transcription Stages 
The following components have been implemented:
- FasterWhisperASR adapter with lazy loading
- get_adapter() factory function  
- transcribe stage implementation
- Manual transcription script (scripts/transcribe_job.py)
- All tests run in CPU-only environment without GPU or model dependencies

M2 (Complete) - Transcription
- Sentence builder fixed: gap-based splits now close the sentence BEFORE the word arriving after the gap (previously the late word was appended to the previous sentence)
- Punctuation and gap checks are independent (a punctuation-ending word no longer blocks a gap split, and vice versa)
- word_idx tracking corrected to use the index at each split point
- All tests in tests/test_sentences.py pass (6/6), CPU-only, no model dependencies

M1 re-verification + fixes (2026-11-18) - first real-video run exposed 4 latent bugs, all fixed:
- media.probe(): ffprobe was invoked without `-i <path>` (probe stage always crashed)
- media.make_proxy(): scale filters were passed as top-level CLI options; now one proper `-vf` chain
- cli.py: generic `except Exception` swallowed the command's own `typer.Exit(0)` ("Error: 0", exit 1); added `except typer.Exit: raise`
- stages/ingest.py: failed jobs could never be retried (early-return on `rowcount == 0` + no stage reset); failed jobs now reset and re-run, CLI prints "Retrying failed job"
- Verified end-to-end on real 80.8s AV1/opus video (job 72e7bb26e760466ea2ca31cc0d1ac021): ingest/probe/proxy/audio all completed; 26 CPU tests green

M1+M2 verification (2026-11-18):
- M1: 4 latent bugs fixed and pipeline verified end-to-end on a real 80.8s AV1/opus video (details in milestones.md)
- M2: transcription glue fixed (`stages/transcribe.py`, `scripts/transcribe_job.py`, `clipper.models` re-exports, ctranslate2 4.9 API fixes in `FasterWhisperASR`; note `adapters/media.py` had a `src.`-style import that also broke script launches)
- M2 verified on real audio: job `72e7bb26e760466ea2ca31cc0d1ac021` transcribed (152 words, 22 sentences) and artifacts validated against the M2 spec
- **IMPORTANT**: GPUs 0–2 are occupied by an Ollama LLM — do NOT run GPU tests. ASR defaults are `device="cpu"`, `compute_type="int8"`; restore `cuda:0` / `float16` via config/env when GPU0 is free
- 26 CPU tests green; touched files ruff-clean

**Current status: M0, M1, M2 all complete and verified on real media. Next: M3 (Scene analysis) — awaiting instructions.**