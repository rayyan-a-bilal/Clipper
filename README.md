# Clipper - Local, Headless Video Clipping Pipeline

Clipper is a local, headless video clipping pipeline designed for content creation workflows. It processes video files through multiple stages: probe, proxy generation, audio extraction, transcription, shot/silence/energy analysis, and pre-ranked clip candidate generation.

## Current Status (2026-11-19)

| Milestone | State |
|---|---|
| M0 – Scaffold & CLI | ✅ complete |
| M1 – Ingest pipeline | ✅ complete, verified end-to-end on real video (probe/proxy/audio) |
| M2 – Transcription | ✅ complete, verified on real audio (152 words / 22 sentences, validated) |
| M3 – Scene analysis (M3a foundations + M3b analysis stage) | ✅ complete, verified on real video (12 shots / 22 silence regions / 162 energy windows) |
| M4 – Clip candidates (M4a foundations + M4b candidates stage) | ✅ complete, verified on real video (4 candidates pre-ranked, skip-deterministic) |
| M5+ – LLM scoring, selection, cuts | ⏳ not started |

Notes:
- ASR currently runs on **CPU int8** — GPUs 0–2 are occupied by another workload. When GPU0 is free: set `CLIPPER_ASR_DEVICE=cuda:0` and `CLIPPER_ASR_COMPUTE_TYPE=float16` (or edit `ASRConfig` defaults in `src/clipper/config.py`) — analysis (M3) is always CPU-only
- `clipper run --job-id` is still a stub; use the per-milestone scripts (see below)
- **Headless OpenCV rule:** this server must use `opencv-python-headless` and NEVER `opencv-python` (GUI build fails with `libGL.so.1`). scenedetect ≥0.7 declares `opencv-python` as a dependency, so install it with `pip install scenedetect --no-deps` and get `cv2` from `opencv-python-headless` (`pip check` then shows an expected false-positive)
- See `BLOCKED.md` for open items (none blocking) and `docs/spec/milestones.md` for full progress notes

## Requirements

- Python 3.11+
- FFmpeg (for media processing)
- scenedetect + opencv-python-headless (M3 shot/silence/energy analysis — headless build ONLY, see Installation note)
- NVIDIA GPU (optional, ASR only; analysis is CPU-only)

## Installation

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e .

# ⚠️ scenedetect ≥0.7 pulls full opencv-python (breaks on this headless
# server with libGL.so.1). Reinstall it without its declared deps and
# provide cv2 via the headless build instead:
pip install click tqdm          # scenedetect runtime deps
pip install scenedetect --no-deps
pip install --force-reinstall opencv-python-headless
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

### Analyze a Job (M3)
Runs the shots → silence → energy analysis stages on an ingested job (CPU-only; requires `scenedetect` + `opencv-python-headless` as above):
```bash
python scripts/analyze_job.py <job-id>
# prints job id, shot count, silence region count, energy window count, artifact paths
# artifacts: data/jobs/<job-id>/{shots.json,silence.json,energy.json} (schema_version 1)
# each stage is skipped when already done (state=done + artifact + schema v1); exits 0/1
```
- **shots** — PySceneDetect `ContentDetector` (threshold 27.0) on `proxy.mp4`; always ≥1 shot, contiguous, covering `[0, duration]`
- **silence** — ffmpeg `silencedetect=noise=-35dB:d=0.5` on `audio.wav`
- **energy** — 0.5s RMS windows from `audio.wav` (16kHz mono pcm_s16le validated; silent windows clamp at −120.0 dB)

### Generate Clip Candidates (M4)
Runs the `candidates` stage: pre-ranks 3–8 sentence windows (15–75s) from M2 `sentences.json` + M3 `energy.json`/`silence.json` (hook/density/energy/silence/niche features → `cheap_score` PRE-RANK, greedy NMS, top 20), writes `candidates.json` and the `candidates` table (`score_json` stays NULL until M5):
```bash
python scripts/candidates_job.py <job-id>
# prints job id, candidate count, top 5 as [start-end] duration score first-words-preview, artifact path
# requires M2b (sentences.json) and M3b (energy.json, silence.json) artifacts
# niche keywords: configure clips.niche_keywords[job.niche] in config.yaml (neutral 0.5 when unset)
# artifact: data/jobs/<job-id>/candidates.json (schema_version 1); skipped when already done; exits 0/1
```

## Project Structure

- `src/clipper/` - Core application code
  - `cli.py` - Command line interface
  - `config.py` - Configuration management
  - `db.py` - Database operations  
  - `devices.py` - Device detection and management
  - `stages/` - Pipeline stage implementations (`ingest`, `transcribe`, `analyze`, `candidates`)
  - `adapters/` - Media processing adapters (`media`, `asr`, `scenes` — all with CPU-only fixtures)
  - `models/` - Data models and artifact contracts
  - `audio_analysis.py` - Pure CPU silence parsing + RMS windowing
  - `candidates.py` - Pure CPU candidate windowing + `cheap_score` pre-rank (M4a)

- `tests/` - Test suite
- `scripts/transcribe_job.py` - M2 transcription runner
- `scripts/analyze_job.py` - M3 analysis runner
- `scripts/candidates_job.py` - M4 candidates runner
- `scripts/make_fixtures.py` - Script to create test fixtures

## Development

To run all development checks:
```bash
python -m pytest -m "not gpu"
python -m ruff check .
```

## License

MIT License