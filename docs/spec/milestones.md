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
- M3a: complete — analysis foundations (contracts, pure logic, fixture adapters; 47 CPU tests green, ruff clean) per `docs/spec/m3.md`

Next: M3b (PySceneDetect, opencv-python-headless only) per your instructions.
Known minor gaps: `clipper run` CLI command is still a stub (use `python scripts/transcribe_job.py <job-id>`)

(Prior note about pre-existing E501s in db.py / sentences.py / tests/test_doctor.py: cleared as part of M3a — `ruff check .` is now fully clean.)

## M2 - Transcription
- ASR integration (faster-whisper)
- Word timestamps
- VAD filtering
- Audio quality checks

## M3a - Analysis Foundations (Complete)
- Fixed decisions in `docs/spec/m3.md`: CPU-only analysis; inputs proxy.mp4 + audio.wav from M1; independent of M2 transcript artifacts; stage names shots/silence/energy; artifacts shots.json / silence.json / energy.json (schema_version=1); job states queued -> running -> queued on success, failed on error
- Module-level constants (no config changes): SHOT_THRESHOLD=27.0 (ContentDetector, M3b), SILENCE_NOISE_DB=-35.0, SILENCE_MIN_DUR=0.5, ENERGY_WINDOW_S=0.5
- Contracts (new file `models/analysis_artifacts.py`, existing contracts untouched): `SilenceRegion{start,end}`, `EnergyWindow{start,end,rms_db}`, `ShotsArtifact{schema_version,shots}`, `SilenceArtifact{schema_version,noise_db=-35.0,min_dur=0.5,regions}`, `EnergyArtifact{schema_version,window_s=0.5,windows}`; re-exported from `clipper.models`
- Pure logic `audio_analysis.py` (no I/O, no ffmpeg, numpy only): `parse_silence_log` (pairs silence_start/end, trailing unpaired start closes at duration, clamps to [0,duration], ignores malformed lines) and `rms_windows` (non-overlapping 0.5s tiling, partial final window only if >=0.1s, rms_db=20*log10(max(rms,1e-6)) -> -120.0 dB floor)
- `MediaAdapter.detect_silence()` abstract method + `FFmpegMediaAdapter.detect_silence` (ffmpeg -hide_banner -nostats -af silencedetect, 600s timeout, MediaError on failure) + `FixtureMediaAdapter.detect_silence` (canned, default one region)
- New `adapters/scenes.py`: `SceneAdapter.detect_shots(video_path, duration_s) -> list[Shot]` + `FixtureSceneAdapter` (default single shot spanning full duration)
- Scope discipline: NO PySceneDetect/opencv/scenedetect yet (M3b — and it MUST use opencv-python-headless, never opencv-python); NO analysis stage/script; NO faces/layouts/candidates (M4/M6)
- Dependency: numpy added only
- Tests: 21 new CPU-only tests in tests/test_analysis.py (contracts round-trip, parse_silence_log, rms_windows, fixtures); 47 total green
- Ruff: fixed 16 pre-existing E501s (db.py, sentences.py, tests/test_doctor.py); `ruff check .` clean

## M3 - Scene analysis  (M3b/M3c pending — awaiting instructions)
- [x] M3a: contracts, pure logic, fixture adapters
- [ ] M3b: PySceneDetect shot detection (opencv-python-headless only)
- [ ] M3c: analysis stage (shots/silence/energy) + run script, job states
- RMS energy analysis (logic done in M3a; stage pending)
- Face detection on CPU (deferred to M4/M6)
- Face tracking across frames (deferred to M4/M6)

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