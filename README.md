# Clipper - Local, Headless Video Clipping Pipeline

Clipper is a local, headless video clipping pipeline designed for content creation workflows. It processes video files through multiple stages including probing, proxy generation, audio extraction, and candidate analysis.

## Current Status (2026-11-18)

| Milestone | State |
|---|---|
| M0 – Scaffold & CLI | ✅ complete |
| M1 – Ingest pipeline | ✅ complete, verified end-to-end on real video (probe/proxy/audio) |
| M2 – Transcription | ✅ complete, verified on real audio (152 words / 22 sentences, validated) |
| M3+ – Scene analysis etc. | ⏳ not started |

Notes:
- ASR currently runs on **CPU int8** — GPUs 0–2 are occupied by another workload. When GPU0 is free: set `CLIPPER_ASR_DEVICE=cuda:0` and `CLIPPER_ASR_COMPUTE_TYPE=float16` (or edit `ASRConfig` defaults in `src/clipper/config.py`)
- `clipper run --job-id` is still a stub; use the transcription script (see below)
- See `BLOCKED.md` for open items (none blocking) and `docs/spec/milestones.md` for full progress notes

## Requirements

- Python 3.11+
- FFmpeg (for media processing)
- NVIDIA GPU (optional, for accelerated processing)

## Installation

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e .
```

## Usage

### Running Tests
```bash
# Run all tests (excluding GPU tests)
python -m pytest -m "not gpu"

# Run ruff checks
python -m ruff check .
```

> [!NOTE]
> ASR currently runs on **CPU int8** — GPUs 0–2 are occupied by another workload. To use GPU0 once it is free: set `CLIPPER_ASR_DEVICE=cuda:0` and `CLIPPER_ASR_COMPUTE_TYPE=float16` (or edit `ASRConfig` defaults in `src/clipper/config.py`).

### Ingest a Video
```bash
python -m clipper ingest --source-path <path-to-video> --client <client-name> --niche <niche-topic>
```
Re-ingesting the same file is idempotent: existing job is reused; a `failed` job is automatically reset and retried.

### Transcribe a Job (M2)
Runs the transcribe/sentences stages for an existing job (large-v3-turbo is already in the HF cache; CPU-only unless you point it at a GPU):
```bash
python scripts/transcribe_job.py <job-id>
# skips stages already completed; e.g. job 72e7bb26e760466ea2ca31cc0d1ac021 is fully transcribed
# `python -m clipper run --job-id <job-id>` is not implemented yet
```

## Project Structure

- `src/clipper/` - Core application code
  - `cli.py` - Command line interface
  - `config.py` - Configuration management
  - `db.py` - Database operations  
  - `devices.py` - Device detection and management
  - `stages/` - Pipeline stage implementations
  - `adapters/` - Media processing adapters
  - `models/` - Data models

- `tests/` - Test suite
- `scripts/make_fixtures.py` - Script to create test fixtures

## Development

To run all development checks:
```bash
python -m pytest -m "not gpu"
python -m ruff check .
```

## License

MIT License