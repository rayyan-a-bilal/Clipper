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

## M3b - Shot/Silence/Energy Analysis Stage (Complete)
- `PySceneDetectAdapter` (`adapters/scenes.py`): lazy `scenedetect`/`cv2` imports, `detect()` + `ContentDetector(threshold=27.0)`, contiguous clamped shots over `[0, duration]`, `SceneError`, `get_scene_adapter()` factory (monkeypatchable)
- `stages/analyze.py`: `run_analysis(job_id, cfg, conn, media, scenes)` — shots → silence → energy; skip only when stage state=`done` + artifact exists + `schema_version==1`; job `queued→running→queued`, failure marks stage `failed` + job `failed` and raises clean `StageError`; energy reads WAV via stdlib `wave` (validates 16kHz mono pcm_s16le) + `rms_windows`
- `scripts/analyze_job.py`: mirror of `transcribe_job.py`; prints job/shots/silence/energy counts + artifact paths; exit 0/1
- 11 new tests (`tests/test_analysis_stage.py`) incl. guarded real integration test; `tests/fixtures/sample.mp4` regenerated (6s two-segment tone fixture, prior file was corrupt)
- Dependencies: `scenedetect` + `opencv-python-headless` (install scenedetect with `--no-deps`; opencv-python NEVER; `pip check` false-positive expected)
- Acceptance: 58 CPU tests green (`pytest -m "not gpu"`), `ruff check .` clean
- Verified on real media (job `72e7bb...`): 12 shots / 22 silence regions / 162 energy windows, re-run idempotent
- Scope: no faces/layouts/candidates/worker/web (M4/M6), M4 not started

## Current Status (updated for M3b)
- M0, M1, M2: complete and verified end-to-end on real media
- M3a: complete — analysis foundations (contracts, pure logic, fixture adapters)
- M3b: complete — real detectors wired, analysis stage + script, verified on fixture and real video; **M3 done**
- M4a: complete — candidate-generation foundations (see below)

Next: M4b (candidates pipeline stage + run script) per instructions; M5 (LLM scoring) after that.

(Prior note about pre-existing E501s in db.py / sentences.py / tests/test_doctor.py: cleared as part of M3a — `ruff check .` is now fully clean.)

## M4a - Candidate Generation Foundations (Complete)
- Fixed decisions recorded in `docs/spec/m4.md`: `cheap_score` is a PRE-RANK (not a quality verdict); inputs M2 sentences.json + M3 energy.json/silence.json; stage name `candidates`, artifact `candidates.json` (schema_version=1)
- Module-level constants in `src/clipper/candidates.py`: MIN_CLIP_S=15.0, MAX_CLIP_S=75.0, MIN_SENTENCES=3, MAX_SENTENCES=8, STRIDE=1, TOP_K=20, NMS_IOU=0.5, W_HOOK=0.35/W_DENSITY=0.20/W_ENERGY=0.20/W_SILENCE=0.10/W_NICHE=0.15
- Window rule: for each start idx i (stride 1) and length L in 3..8, end_idx=i+L-1; keep only if 15.0<=duration<=75.0; `sentence_range` INCLUSIVE `(start_idx,end_idx)`
- `cand_id = f"{job_id}-c{start_idx:04d}-{end_idx:04d}"` — globally unique (job_id prefix + unique index pair) since `candidates.cand_id` is a PRIMARY KEY
- Features (clamped [0,1]): hook (first sentence, +0.25 each of 6 classes, word-boundary keyword matching, cap 1.0), density (speech/duration), energy (population std of rms_db over windows whose MIDPOINT is in-window, /12, 0.0 if <2 windows), silence (1 - min(fraction/0.25,1)), niche (fraction of configured keywords present, neutral 0.5 when none)
- cheap_score = clamp(weighted sum, 0,1), rounded to 6 decimals; ordering cheap_score DESC then start ASC then end ASC; greedy NMS keep if temporal IoU < 0.5, stop at TOP_K; byte-identical determinism
- Contract (additive): `CandidatesArtifact{schema_version=1, candidates: list[Candidate]}` in `models/candidate_artifacts.py`, re-exported; existing contracts untouched
- Config (backward compatible): `ClipsConfig.niche_keywords: dict[str, list[str]] = {}` + `clips.niche_keywords: {}` in config.yaml (commented example); existing configs load unchanged
- `src/clipper/candidates.py`: pure `generate_candidates(...)` + helpers `hook_score/density_score/energy_score/silence_score/niche_score/cheap_score/temporal_iou/_nms/_build_windows` (no I/O)
- Fixtures `tests/fixtures/m4/`: sentences.json (30/~271s), energy.json (542 x 0.5s), silence.json (3 regions), golden candidates.json (20 candidates, committed, byte-identical on regen)
- Tests `tests/test_candidates.py`: 41 CPU-only tests (round-trip, window invariants, no-candidates short/long, each feature, cheap_score bounds/rounding, deterministic ordering + tie-break, NMS IoU/TOP_K, cand_id format/uniqueness, golden byte-identical, seeded property test)
- Acceptance: `pytest -m "not gpu"` 99 passed; `ruff check .` clean
- M4a scope respected: NO candidates stage/script yet, NO LLM scoring/selection/cut-snapping (M5), M6/M7 untouched

## Current Status (updated for M4a)
- M0, M1, M2, M3a, M3b, M4a complete
- M3 fully done; M4 foundations done (contracts + config + pure logic + fixtures + tests)
- Next: M4b (candidates pipeline stage + run script, job states, wiring to config `clips.niche_keywords`), then M5

(Prior note about pre-existing E501s in db.py / sentences.py / tests/test_doctor.py: cleared as part of M3a — `ruff check .` is now fully clean.)

## M4b - Candidates Stage Wiring (Complete)
- `stages/candidates.py`: `run_candidates(job_id, cfg, conn)` — single `candidates` stage wiring the M4a pure `generate_candidates`; requires `sentences.json` (M2b, `scripts/transcribe_job.py`) and `energy.json`/`silence.json` (M3b, `scripts/analyze_job.py`); missing input fails with a clean `StageError` naming the artifact AND the milestone stage/script that produces it; each input validated for `schema_version == 1` (clean failure on mismatch) and as its typed model; niche keywords resolved from `cfg.clips.niche_keywords[job.niche]` (neutral 0.5 when unset)
- Writes `candidates.json` as `CandidatesArtifact` (`model_dump_json(indent=2)` + newline — the same canonical form as the M4a golden fixture; byte-identical across runs; deterministic ordering from M4a)
- Upsert into the EXISTING `candidates` table (no migration): `cand_id, job_id, candidate_json = Candidate.model_dump_json()`, `score_json = NULL` (M5 fills this); the job's rows are replaced on each real (non-skipped) run so retries are exact
- Skip only when DB stage state=`done` AND artifact exists AND `schema_version == 1`; on success mark stage `done` (started/finished timestamps + artifact_path), job `queued→running→queued`; on failure mark stage `failed` + error text, job `failed`, raise clean `StageError`; retry after failure works
- Zero candidates is a VALID success (empty artifact, no `candidates` rows, stage done)
- `scripts/candidates_job.py` (mirrors `analyze_job.py`): argparse JOB_ID; prints job id, candidate count and the top 5 as `[start-end] duration score first-words-preview` + artifact path; exit 0 success (incl. zero candidates), exit 1 clean error
- `stages/__init__.py` re-exports `run_candidates`
- Tests `tests/test_candidates_stage.py` (14 new, CPU-only, temp DB, M4a fixtures): artifact+stage row+candidate rows (`score_json NULL`, `cand_id` job prefix, one row per candidate); second run skips (no rewrite, stage row unchanged); byte-identical `candidates.json` across two fresh runs; missing sentences/energy/silence each fail with the correct milestone hint; `schema_version` mismatch fails cleanly; missing job fails cleanly; zero-candidate input succeeds (empty artifact, zero rows); failure→retry succeeds; `clips.niche_keywords[job.niche]` changes the pre-rank vs neutral; script success/missing-job/failure exit codes
- Acceptance: `pytest -m "not gpu"` 113 passed (99 prior + 14 new); `ruff check .` clean
- M4b scope respected: NO LLM scoring / selection / cut snapping (M5); M4a constants/formulas untouched

## Current Status (updated for M4b)
- M0, M1, M2, M3a, M3b, M4a, M4b complete
- M3 fully done; M4 pipeline stage done — `candidates.json` + `candidates` table rows are produced and skip-deterministic; `score_json` awaits M5
- Next: M5 (LLM scoring into `score_json`, selection) — STOP after M4b per instructions

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
- [x] M4a: candidate-generation foundations (contract, config, pure logic, fixtures, tests)
- [x] M4b: candidates pipeline stage + run script (wires config `clips.niche_keywords`)
- Candidate window generation (M4a logic done)
- Heuristic scoring (M4a done)
- Candidate stage storage and retrieval (M4b done — `candidates.json` + `candidates` table, `score_json` NULL for M5)
- Window filtering (M4a logic done)

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