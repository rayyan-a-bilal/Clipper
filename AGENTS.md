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

M4b - Candidates Stage Wiring (Complete)
The following components have been implemented:
- `stages/candidates.py`: `run_candidates(job_id, cfg, conn)` — wires the M4a `generate_candidates` pure logic into the single `candidates` stage; requires M2b `sentences.json` + M3b `energy.json`/`silence.json` (clean `StageError` naming the missing artifact and the milestone stage/script that produces it: `scripts/transcribe_job.py` / `scripts/analyze_job.py`); validates `schema_version == 1` on each input; resolves `cfg.clips.niche_keywords[job.niche]`; writes `candidates.json` as `CandidatesArtifact` (deterministic, `model_dump_json(indent=2)` — byte-identical across runs, same canonical form as the M4a golden fixture); upserts candidates into the existing `candidates` table with `candidate_json = Candidate.model_dump_json()` and `score_json = NULL` (M5 fills it; no migration); skip only when stage state=`done` AND artifact exists AND `schema_version==1`; job `queued→running→queued`, failure marks stage `failed` + job `failed` and raises clean `StageError`; zero candidates is a VALID (empty) success; reuses M3b stage-bookkeeping helpers from `stages/analyze.py`
- `scripts/candidates_job.py` (mirrors `analyze_job.py`): argparse JOB_ID; prints job id, candidate count and top 5 as `[start-end] duration score first-words-preview` + artifact path; exit 0 success (incl. zero candidates), exit 1 clean error
- `stages/__init__.py` re-exports `run_candidates`
- Tests `tests/test_candidates_stage.py` (14 new, CPU-only, temp DB, M4a fixtures): artifact+stage row+candidate rows (score_json NULL, job prefix); second run skips; byte-identical `candidates.json` across two fresh runs; missing sentences/energy/silence each fail with correct milestone hint; `schema_version` mismatch fails cleanly; missing job fails cleanly; zero-candidate success (empty artifact, no rows); retry after failure succeeds; niche keywords from config change the pre-rank; script tests (success, missing job, failure exit codes)
- Acceptance: `.venv/bin/python -m pytest -m "not gpu"` -> 113 passed; `.venv/bin/python -m ruff check .` -> clean
- M4b scope respected: no LLM scoring / selection / cut snapping (M5), no changes to M4a constants/formulas

**Current status: M0, M1, M2, M3a, M3b, M4a, M4b complete. M3 fully done. M4 is "stage-done, scoring-pending" — candidate generation + stage wiring verified; M5 (LLM scoring) is next.**
M5a - Managed Ollama Runtime, LLM Client, MockLLM, Contracts, Prompts (Complete)
The following components have been implemented:
- `docs/spec/llm.md` + `docs/spec/m5.md` — fixed the M5 runtime design (main Ollama on 11434 is NEVER touched; ephemeral child on 127.0.0.1:11435 pinned to GPU1 via CUDA_VISIBLE_DEVICES=1 at spawn; child dies with the stage run; num_ctx per-request) and ALL M5 constants (axes 0-10, weighted formula, gate 7.2/hook>=6/self_contained>=7, selection 3/5, dedupe IoU>0.30, cut constants + order, batch-failure => reject not crash, zero clips is valid, DRAFT ClipPlans center_crop/pending_qc, clip_id format)
- `config.yaml` + `LLMConfig`: full new `llm:` block (backward compatible; legacy endpoint/api_key/timeout still load)
- `src/clipper/llm_runtime.py`: `managed_ollama(cfg)` — spawn/ready/model/warm-up/GPU-verify, clean `LLMError` on port-occupied/wrong-GPU/low-VRAM/missing-model/missing-nvidia-smi, and guaranteed process-group teardown (SIGTERM->SIGKILL->port release) on success AND failure
- `src/clipper/adapters/llm.py`: `LLMClient` protocol, `OllamaLLMClient` (httpx only; strict schema `format` with documented `format="json"` fallback; retry-then-`LLMError`; no model/endpoint fallback), `get_llm_client` (tests monkeypatch), `MockLLM` (canned queue, records calls, zero I/O)
- New contracts (additive, schema_version=1): `CandidateContext`, `ScoringBatch`, `ScoringResponse`, `ScoresArtifact` (+ `docs/spec/contracts.md` sync)
- `src/clipper/prompts.py::SCORING_SYSTEM` (single source of truth) mirrored into `docs/spec/scoring_prompt.md`
- `scripts/llm_preflight.py` (managed runtime + tiny generate_json sanity + GPU/VRAM print, exit 0/1, always cleaned up)
- Dependency added: httpx only (no `ollama` package); all tests CPU-only (httpx/subprocess/nvidia-smi mocked)
- Acceptance: `.venv/bin/python -m pytest -m "not gpu" -v` -> 145 passed; `.venv/bin/python -m ruff check .` -> All checks passed
- M5a scope respected: NO scoring (M5b), NO selection/cut snapping (M5c), M6/M7 GPU spec untouched

**Current status: M0, M1, M2, M3a, M3b, M4a, M4b, M5a complete. M5a delivered the managed Ollama runtime + client + contracts + prompt; M5b (scoring stage + selection) is next. PI agent must be stopped before real M5 runs (see docs/spec/llm.md).**
M5b - Scoring Stage (Complete)
The following components have been implemented:
- `src/clipper/scoring.py` (PURE functions, no I/O): `build_context(candidate, sentences, cfg)` (inclusive transcript slice + ±`cfg.llm.context_before_s/context_after_s` sentence-overlap windows), `estimate_tokens = max(1, len//4)`, `pack_batches` (greedy, order-preserving, 1500-token prompt-overhead reserve, per-candidate `est(transcript)+est(before)+est(after)+60`, max 5 per batch, oversized single candidate truncates context_before/context_after FIRST — never the transcript), `validate_scores` (exactly one score per cand_id, no extras/dupes; axes int 0..10; structural violations => `ValueError` => batch retry; out-of-bounds suggested_in/out within ±`SUGGEST_SLACK=8.0` keeps scores but sets `reject_reason="suggestion_out_of_bounds"`), `weighted_score` (spec weights)
- `src/clipper/stages/score.py`: `run_scoring(job_id, cfg, conn, client: LLMClient)` — stage `scoring`; requires `candidates.json` (M4b) + `sentences.json` (M2b) + `probe.json` (M1) with clean `StageError` naming the missing artifact AND producing milestone/script, `schema_version==1` on all inputs; skip only when DB stage=done AND valid `scores.json` exists; job queued->running->queued / failed on error; retries up to `cfg.llm.max_retries` per batch on `LLMError`/semantic `ValueError` (incl. schema-validation failure); batch failure after retries REJECTS those candidates (all axes 0, `reject_reason="llm_batch_failed: <msg>"`) — never fails the job; writes `scores.json` (`ScoresArtifact(model=cfg.llm.model)`, ALL candidates in original artifact order); updates `candidates.score_json` per cand_id; zero candidates is a valid success (zero LLM calls); `stages/__init__` re-exports `run_scoring`
- `scripts/score_job.py JOB_ID [--mock]`: default = ONE managed ephemeral Ollama (`managed_ollama(cfg)`) for all batches, killed at the end (success AND failure), `get_llm_client` inside; `--mock` = `_MockScorer` deterministic wiring-test ONLY (position j in batch -> `10-(2j)%11` on all axes, suggested_in=out=candidate window; never a real judgement); prints job/model/batches/scored/rejected/top-5-by-weighted/artifact; exit 0 on success (incl. all-rejected), exit 1 clean error
- Tests (47 new, CPU-only: MockLLM + temp DB + synthetic artifacts, no Ollama/GPU/nvidia-smi): `tests/test_scoring.py` (build_context windows + edges, estimate_tokens, pack_batches budget/size/order/truncation, validate_scores id/axis/OOB/edge cases) + `tests/test_scoring_stage.py` (artifact/stage row/score_json updates, skip on second run, batch-failure policy per batch AND all batches, retry-then-success, semantic-validation retries, missing-input milestone hints x3, schema mismatch, missing job, zero-candidate success, byte-identical determinism across fresh runs, script exit codes + mock derivation)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -v` -> 192 passed; `python -m ruff check .` -> All checks passed; `python scripts/score_job.py 72e7bb26e760466ea2ca31cc0d1ac021 --mock` -> exit 0 (1 batch, 4 scored, 0 rejected; second run skips the completed stage)
- M5b scope respected: NO selection / cut refinement / metadata (M5c); M5a constants, prompts, contracts and config defaults untouched; no new dependencies

**Current status: M0, M1, M2, M3a, M3b, M4a, M4b, M5a, M5b complete. M5b delivered the scoring stage (pure scoring helpers, `scoring` stage, `scores.json`, `candidates.score_json`, batch-failure rejection policy, mock/managed-ollama runner script). M5c (selection + cut refinement) is next. PI agent must be stopped before real M5 runs (see docs/spec/llm.md).**
