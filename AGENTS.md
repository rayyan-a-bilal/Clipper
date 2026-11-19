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

M3a - Analysis Foundations (Complete)
The following components have been implemented:
- `docs/spec/m3.md` — fixed decisions for M3 (CPU-only, inputs, stage names, artifacts, constants, silence/energy/shots rules, job states, M3b constraints)
- New contracts (`src/clipper/models/analysis_artifacts.py`, schema_version=1, re-exported from `clipper.models`): `SilenceRegion`, `EnergyWindow`, `ShotsArtifact`, `SilenceArtifact`, `EnergyArtifact` (existing contracts untouched)
- Pure CPU logic (`src/clipper/audio_analysis.py`): `parse_silence_log(text, duration_s)` and `rms_windows(samples, sample_rate, window_s)` — no I/O, no ffmpeg, no heavy libs beyond numpy
- Module-level constants (no config changes): `SILENCE_NOISE_DB=-35.0`, `SILENCE_MIN_DUR=0.5`, `ENERGY_WINDOW_S=0.5` (audio_analysis.py), `SHOT_THRESHOLD=27.0` (adapters/scenes.py)
- `adapters/media.py`: `detect_silence()` added to `MediaAdapter` + `FFmpegMediaAdapter` (fixed silencedetect command, 600s timeout, `MediaError` on failure) and `FixtureMediaAdapter` (configurable canned regions, default one region)
- New `adapters/scenes.py`: `SceneAdapter` protocol + `FixtureSceneAdapter` (default single full-duration shot); NO PySceneDetect yet (M3b, must use opencv-python-headless)
- Dependency added: numpy only (scenedetect/opencv explicitly deferred to M3b)
- 21 new CPU-only tests (`tests/test_analysis.py`); 47 total tests green, `ruff check .` clean (also cleared 16 pre-existing E501s in db.py/sentences.py/tests)
- M3a scope respected: no PySceneDetect integration, no analysis stage/script, no faces/layouts/candidates (M4/M6)

M3b - Shot/Silence/Energy Analysis Stage (Complete)
The following components have been implemented:
- `adapters/scenes.py`: `PySceneDetectAdapter` added — lazy imports of `scenedetect` + `cv2` inside `detect_shots()` (module import stays light; M3a tests still pass without those packages), convenient API `detect(...) + ContentDetector(threshold=SHOT_THRESHOLD)`, FrameTimecode→seconds conversion (`.seconds` property with `get_seconds()` fallback for older releases), clamped, contiguous shots over `[0, duration_s]`, single full-duration shot when no cut is found; `SceneError` exception; `get_scene_adapter()` factory (tests monkeypatch this)
- `stages/analyze.py`: `run_analysis(job_id, cfg, conn, media, scenes)` — loads job from the DB (clean `StageError` if missing), requires `proxy.mp4`/`audio.wav` from M1, reads duration from `probe.json` (no re-probe), runs `shots` → `silence` → `energy` in order; each stage skips only when DB state=`done` AND artifact file exists AND `schema_version == 1`; writes `shots.json`/`silence.json`/`energy.json`; marks stages `running`/`done` (with started/finished timestamps + artifact_path) and job `running`→`queued`; on failure marks the current stage `failed` + error text, job `failed`, raises a clean `StageError`; energy stage reads WAV via stdlib `wave` and validates 16kHz mono pcm_s16le before `rms_windows`
- `scripts/analyze_job.py` (mirrors `transcribe_job.py`): argparse JOB_ID, loads config+DB, calls `run_analysis`, prints job id / shot count / silence region count / energy window count / artifact paths; exits 0/1. Adapters come from the module-level `get_adapter()` / `get_scene_adapter()` factories so tests can monkeypatch them
- Tests (`tests/test_analysis_stage.py`, 11 new): artifacts + stage rows + job state; second run skips all 3 stages (adapter call counts stay at 1); missing proxy/audio and missing job fail cleanly; adapter exception marks stage+job failed; energy rejects stereo and non-16kHz WAV cleanly; script tests with monkeypatched `get_scene_adapter()`/`get_adapter()`; real integration test (skipped when ffmpeg/scenedetect/cv2 missing) — ingests `tests/fixtures/sample.mp4` (regenerated: 6s two-segment testsrc/sine tone with a hard cut; the previous 178-byte file was corrupt), then asserts ≥1 contiguous shot covering [0, duration], valid silence artifact, energy windows covering the audio with rms_db > -120 for the sine tone
- Dependencies: `scenedetect` (0.7.3) + `opencv-python-headless` (+ small transitive `click`/`tqdm`) added to pyproject. **IMPORTANT:** scenedetect ≥0.7 declares `opencv-python` as a hard dependency — install with `pip install scenedetect --no-deps` and provide cv2 via opencv-python-headless. `pip check` reports a false-positive requirement warning for this pair (expected); `opencv-python` must NEVER be installed on this headless server (fails with libGL.so.1)
- Verified end-to-end on real media: job `72e7bb26e760466ea2ca31cc0d1ac021` (80.8s video) — 12 contiguous shots over [0, 80.781s], 22 silence regions, 162 energy windows (max -21.0 dB, silent windows clamped at exactly -120.0 dB), all 3 stages `done`, job `queued`, re-run skips all stages in <1s
- 58 CPU tests green (`pytest -m "not gpu"`), `ruff check .` clean
- M3b scope respected: no faces/layouts/candidates/worker/web (M4/M6), no M4 started

M4a - Candidate Generation Foundations (Complete)
The following components have been implemented:
- `docs/spec/m4.md` — fixed decisions for M4a: the `candidates` stage / `candidates.json` (schema_version=1); `cheap_score` is a PRE-RANK (not a quality verdict); inputs = M2 sentences.json + M3 energy.json/silence.json; all module-level constants (`MIN_CLIP_S=15.0`, `MAX_CLIP_S=75.0`, `MIN_SENTENCES=3`, `MAX_SENTENCES=8`, `STRIDE=1`, `TOP_K=20`, `NMS_IOU=0.5`, weights `W_HOOK=0.35/W_DENSITY=0.20/W_ENERGY=0.20/W_SILENCE=0.10/W_NICHE=0.15`); the window rule (inclusive `sentence_range`, 15–75s); `cand_id = f"{job_id}-c{start_idx:04d}-{end_idx:04d}"` (globally unique via job_id prefix + unique index pair); the five feature definitions (hook/density/energy/silence/niche, all clamped to [0,1]); ordering (cheap_score DESC, start ASC, end ASC); greedy NMS (keep if IoU < 0.5, stop at TOP_K); determinism (byte-identical output)
- New contract `CandidatesArtifact{schema_version=1, candidates: list[Candidate]}` in `src/clipper/models/candidate_artifacts.py` (additive; the existing `Candidate` model and all prior contracts untouched), re-exported from `clipper.models`
- Config (backward compatible): `ClipsConfig.niche_keywords: dict[str, list[str]] = {}`, plus `clips.niche_keywords: {}` in `config.yaml` with a commented example; existing configs still load unchanged
- Pure CPU logic `src/clipper/candidates.py` (no I/O, DB, ffmpeg, network): `generate_candidates(sentences, energy_windows, silence_regions, job_id, niche_keywords) -> list[Candidate]` plus individually testable helpers — `hook_score`, `density_score`, `energy_score`, `silence_score`, `niche_score`, `cheap_score`, `temporal_iou`, `_nms`, `_build_windows`
- Fixtures `tests/fixtures/m4/` (deterministic synthetic inputs): `sentences.json` (30 sentences over ~271s with questions/digits/negations/contrast words), `energy.json` (542 x 0.5s windows), `silence.json` (3 regions), and the committed golden `candidates.json` (20 candidates, generated from the implementation and verified byte-identical on regen)
- Tests `tests/test_candidates.py` (41 CPU-only tests): artifact JSON round-trip; window invariants (3–8 sentences, 15–75s, inclusive indices, start<end); no-candidates for too-short/too-long sentences; per-feature tests (hook patterns + word-boundary, density, energy std + midpoint filter, silence penalty, niche boost + neutral 0.5); cheap_score in [0,1] + rounding; deterministic ordering + tie-break by position; NMS suppresses IoU >= 0.5 and respects TOP_K; cand_id format + global uniqueness; golden byte-identical; seeded property test asserting all invariants
- Acceptance: `.venv/bin/python -m pytest -m "not gpu"` -> 99 passed; `.venv/bin/python -m ruff check .` -> clean
- M4a scope respected: no `candidates` pipeline stage/run script yet, no LLM scoring / selection / cut snapping (M5), no M6/M7

**Current status: M0, M1, M2, M3a, M3b, M4a complete. M3 fully done. M4a (candidate-generation foundations) done — contracts, config, pure logic, fixtures, 41 tests. Next: M4b (candidates stage + script) per instructions; M4 is not yet "done" (stage wiring + M5 remain).**