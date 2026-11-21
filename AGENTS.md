# Clipper Project Rules

> Coding agents: a project skill with full operating rules (paths, venv ritual, milestone map, M5 rules, acceptance commands) lives in this repo at `.agents/skills/clipper-project/SKILL.md` — load/use it for any clipper task. This project's root is `/home/rbilal/clipper`.

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

M5C-0 — Audit (Complete)
- Audit-only pass over the partial M5c WIP; no features added. Verdict: all M5c pieces existed on disk (uncommitted) — `selection.py`, `cuts.py`, `models/selection_artifacts.py`, `METADATA_SYSTEM`, `stages/select.py`, `scripts/select_job.py`, `tests/test_selection.py` + `tests/test_cuts.py`; 224/224 tests green; collection clean. No spec drift on gate weights/thresholds, cut constants/order, backfill, zero-clip path, or all-candidate decisions. Gaps: 19 ruff style errors (tests + one script line) and no stage-level tests. Status doc: `docs/spec/m5c_status.md`.

M5C-1 — Selection pure logic + contracts (Complete)
- Branch: `m5c1-selection-pure`. Scope: selection pure logic + contracts ONLY — no cuts, no metadata, no stage changes.
- Contracts (`src/clipper/models/selection_artifacts.py`, schema_version=1, registered in `models/__init__`): `ClipDecision` (cand_id, weighted, eligible, gate_reasons, suppressed_by, selected, cut_reject_reason=None), `SelectionArtifact`
- `src/clipper/selection.py` (PURE, no I/O, no LLM): `weighted_score` (0.30/0.25/0.20/0.15/0.10 via `scoring.weighted_score`), `passes_gate(score, cfg)` — thresholds now read from **`cfg.clips.gate.{weighted, hook, self_contained}`** (defaults 7.2/6/7 per docs/spec/m5.md; one reason per violated axis; int-0..10 axis validation), `temporal_iou` (M4 helper re-export), `select_clips` — eligible = reject_reason None AND axes valid AND gate pass; rank weighted DESC / cheap DESC / start ASC; greedy suppress IoU > 0.30; take `cfg.clips.target_count` (3) hard-capped at `cfg.clips.max_count` (5); decision record for EVERY candidate (selected or not)
- `src/clipper/config.py`: new `ClipGateConfig` (weighted=7.2, hook=6, self_contained=7) + `ClipsConfig.gate` field (backward compatible: existing configs load unchanged)
- Tests (CPU-only): existing 17 `test_selection.py` coverage (weighted math exact, per-axis gate boundaries, out-of-range axes, IoU values + re-export identity, suppression strictly >0.30, ranking tie-breaks, target/max caps, zero-eligible valid, all-candidate decision records, spec-constant mirror) + NEW `test_gate_thresholds_come_from_config` (each gate threshold independently driven by `cfg.clips.gate.*`) + gate defaults asserted in `test_llm_config.py::test_selection_config_fields`
- Ruff cleanup of remaining M5c style errors (`tests/test_selection.py` E501/F841/C408/I001, `tests/test_cuts.py` I001/F401, `scripts/select_job.py` E501)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -k "selection" -v` -> 20 passed; `python -m pytest -m "not gpu" -v` -> **225 passed**; `python -m ruff check .` -> All checks passed
- Scope respected: NO cuts, NO metadata, NO stage work touched beyond the gate-config read; no new dependencies; M6 not started

M5C-2 — Cut refinement pure logic (Complete)
- Branch: `m5c2-cuts-pure`. Scope: `src/clipper/cuts.py` PURE cut logic ONLY — no stage, no metadata, no LLM, no file I/O.
- Implementation verified against `docs/spec/m5.md` (present from M5c WIP, already committed on main as `be5205d`; no logic changes needed): constants verbatim (`SUGGEST_SLACK=8.0`, `SENTENCE_SNAP=1.0`, `PAD_START=0.120`, `PAD_END=0.250`, `SILENCE_SNAP=0.300`, `SHOT_SNAP=0.400`); `refine_cut(candidate, score, sentences, words, silence_regions, shots, duration_s) -> (start, end)` in the STRICT spec order — clamp to `[cand.start-8, cand.end+8] ∩ [0,duration]` → sentence snap (nearest within 1.0s; beyond-1.0s fallback is the documented open judgment note in `docs/spec/m5c_status.md`) → word padding (`first word .start - PAD_START` / `last word .end + PAD_END`) → clamp `[0,duration]` → silence snap ≤ 0.300 → shot snap ≤ 0.400 → validate LAST (`start < end`, `15.0 ≤ dur ≤ 75.0`, ≥ 1 full sentence); `CutError` carries a specific reason
- Tests `tests/test_cuts.py` (16 CPU-only golden cases): sentence snap; exact word padding; silence snap within 0.300; shot snap within 0.400 (+ beyond-tolerance ignored); clamping to candidate ±8 AND video bounds; mid-gap suggestion; too-short/too-long/inverted/no-sentence rejections with specific reasons; no silence/shots available still works (NEW); determinism (NEW)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -k "cuts" -v` -> 16 passed; `python -m pytest -m "not gpu" -v` -> **227 passed**; `python -m ruff check .` -> All checks passed
- Scope respected: NO selection stage, NO clip_plans stage, NO metadata, NO script touched; no new dependencies; M6 not started

M5C-3 — Metadata contracts, prompt, validation, fallback (Complete)
- Branch: `m5c3-metadata-pure` (based on `m5c2-cuts-pure` — it contains main + the M5C-1/M5C-2 docs/tests; main itself already has all M5c code from `be5205d`). Scope: metadata contracts + prompt + PURE helpers ONLY — NO selection stage, NO clip_plans stage, NO script, no live LLM.
- Contracts verified (present from M5c WIP on main, NOT re-invented): `ClipMetadata{cand_id, title, hook_overlay, description, hashtags}`, `MetadataResponse{items: list[ClipMetadata]}` in `src/clipper/models/selection_artifacts.py`, re-exported from `clipper.models`. Judgment note (documented per no-drift rule): `schema_version=1` applies to STORED artifacts (`SelectionArtifact`, `ClipPlansArtifact` both carry it); LLM-facing response contracts intentionally carry NO `schema_version`, matching the established `ScoringResponse` pattern — adding one would force the LLM to emit the field.
- `prompts.py::METADATA_SYSTEM` verified against the card (audited M5C-0, untouched): title ≤80 chars; hook_overlay ≤6 words AND ≤40 chars; description ≤300 chars; hashtags 3-6, each `#`-prefixed, no spaces; JSON only matching `MetadataResponse`; no clickbait lies; input contains ONLY the selected clips' final transcript — never the full video transcript.
- NEW `src/clipper/metadata.py` (PURE, no I/O, no LLM): constants `TITLE_MAX=80`, `HOOK_MAX_WORDS=6`, `HOOK_MAX_CHARS=40`, `DESCRIPTION_MAX=300`, `HASHTAG_MIN=3`, `HASHTAG_MAX=6`, `FALLBACK_TITLE_WORDS=8`, `FALLBACK_HOOK_WORDS=5`, `FALLBACK_DESCRIPTION_CHARS=200`, `REQUEST_CLIP_KEYS`; `validate_metadata(response, selected_cand_ids) -> list[ClipMetadata]` (exactly one item per selected cand_id, no extras/dupes; every field limit enforced; `ValueError` names the cand_id); `fallback_metadata(cand_id, transcript, niche) -> ClipMetadata` (deterministic: first 8 words title / first 5 words hook / first 200 chars description / `['#niche']` or `['#clip']` when niche empty — char-clipped to limits as a safety net so ANY transcript stays within bounds and validatable); `build_metadata_request(selected) -> str` (compact JSON `{"clips":[...]}`; whitelists exactly `cand_id`, `transcript`, `niche`, `duration_s` per clip — extras dropped; input order preserved)
- Tests `tests/test_metadata.py` (35 new, CPU-only): contracts round-trip + required/typed-field failures; validate missing/extra/wrong/duplicate cand_id, count mismatch, empty title/hook/description, over-length fields (7-word hook, 41-char hook, 81-char title, 301-char description), boundary accepts (80/40/300, 3 and 6 hashtags), bad hashtags (no `#`, inner space, bare `#`, <3, >6), error names cand_id; fallback exact values, `#clip` default for empty/blank niche, determinism, within limits for normal AND hostile inputs (500-char word, empty transcript); request whitelisting (full-video-transcript marker NOT carried even if passed, unknown keys dropped), order, compactness, zero-clip `{"clips":[]}`; prompt limits mirror module constants verbatim + no-clickbait / JSON-only / one-item-per-clip-ordered rules
- Acceptance: venv verified (`/home/rbilal/clipper/.venv/bin/python`); `python -m pytest -m "not gpu" -k "metadata" -v` -> 35 passed; `python -m pytest -m "not gpu" -v` -> **262 passed**; `python -m ruff check .` -> All checks passed
- Scope respected: NO stages touched (`stages/select.py` internal helpers left as-is for the M5C stage-test milestone), NO script, no changes to existing contracts/prompt/constants; no new dependencies; M6 not started

**Current status: M0–M5b + M5C-0 (audit) + M5C-1 (selection pure logic & contracts) + M5C-2 (cut refinement pure logic & tests) + M5C-3 (metadata contracts/prompt + pure validation/fallback/request builder) complete — 262 CPU tests green, ruff clean. Next: M5C stage-level tests (docs/spec/m5c_status.md). PI agent must be stopped before real LLM runs (docs/spec/llm.md).**

M5C-4 — Selection Stage (Complete)
- Branch: `m5c4-selection-stage`. Scope: the `selection` stage ONLY — NO metadata, NO clip_plans, NO script. NO LLM anywhere (no Ollama import, no client parameter on `run_selection_stage`).
- `src/clipper/stages/select.py`: `run_selection_stage(job_id, cfg, conn)` — single `selection` stage; requires `candidates.json` (M4b), `scores.json` (M5b), `sentences.json` + `words.json` (M2b), `silence.json` + `shots.json` (M3b), `probe.json` (M1) — each with clean `StageError` naming the missing artifact AND the producing milestone/script, `schema_version == 1` validated on every input; skip only when DB stage=done AND a valid `selection.json` exists; job `queued->running->queued`, failure marks stage+job `failed`
- Rank/gate/dedupe via pure `select_clips`; refine in RANK order via `refine_cut`; on `CutError` record `cut_reject_reason` and BACKFILL from the next eligible, non-overlapping (IoU > 0.30) candidate until `min(target_count, max_count)` reached or the list exhausted
- `ClipDecision` extended with `refined_start`/`refined_end` (default `None`) — the refined cut is persisted on SELECTED decisions so M5C-5 (`clip_plans`) can reuse the exact cut; `stages/__init__` re-exports `run_selection_stage` (keeps the `run_selection` alias; its client parameter is ignored / never called)
- Writes `selection.json` (`SelectionArtifact`, schema_version=1) with a decision for EVERY candidate (artifact order); zero selected is a VALID success
- Tests `tests/test_selection_stage.py` (14 new, CPU-only, temp DB, synthetic artifacts): artifact + stage row + job state; ONE decision per candidate with `refined_start`/`refined_end` matching `refine_cut` exactly; backfill after `CutError` reaches target (3 selected, 1 cut-rejected); zero-selected valid success (stage done, job queued); skip on second run (byte-identical artifact); per-file missing-input failures (all 7 inputs) with the correct milestone hint marking stage+job failed; `schema_version` mismatch fails cleanly; missing job (no stage row); LLM-free assertion (`OllamaLLMClient` / `managed_ollama` monkeypatched to RAISE, no `client` in the signature, no LLM surface in the module)
- Acceptance: venv verified (`/home/rbilal/clipper/.venv/bin/python`); `python -m pytest -m "not gpu" -k "selection_stage or select_stage" -v` -> 14 passed; `python -m pytest -m "not gpu" -v` -> 276 passed; `python -m ruff check .` -> All checks passed
- Scope respected: NO clip_plans stage, NO metadata call, NO script, no changes to M5a/b/c pure constants-contracts; no new dependencies; M6 not started

**Current status: M0–M5b + M5C-0 (audit) + M5C-1 (selection pure) + M5C-2 (cuts pure) + M5C-3 (metadata pure) + M5C-4 (selection stage) complete — 276 CPU tests green, ruff clean. (M5C-4 line retained; superseded below.) PI agent must be stopped before real LLM runs (docs/spec/llm.md).**

M5C-5 — Clip Plans Stage (M5C-5)
- Branch: `m5c5-clip-plans-stage`. Scope: the `clip_plans` stage + the rewritten `scripts/select_job.py` ONLY; NO M6 work (DRAFT plans stay `layout=center_crop` / `crop_path=[]` / `status=pending_qc`); the metadata failure NEVER fails the job (always fallback metadata).
- `src/clipper/stages/select.py`: `run_clip_plans_stage(job_id, cfg, conn, client)` — single `clip_plans` stage; requires a DONE `selection` stage and valid `selection.json` (+ `candidates.json`/`sentences.json` for rank order and clip transcripts) — clean `StageError` otherwise; skip only when DB stage=done AND valid `clip_plans.json` exists (byte-identical artifact on re-run)
- ZERO selected: writes `clip_plans.json = {schema_version:1, plans:[]}`, NO LLM call, job `queued`; valid metadata (ONE batched `generate_json` on `METADATA_SYSTEM` + `build_metadata_request(selected)` with ONLY the selected clips' final transcripts — never the full video) → 1 DRAFT `ClipPlan` per selected clip, `clip_id = f"{job_id}-clip{n:02d}"` (rank order, unique, format-verified); metadata `LLMError`/timeout/invalid-schema → retries (`cfg.llm.max_retries`) then deterministic `fallback_metadata`, job STILL succeeds; every plan upserted into `clips` (`plan_json`, `status='pending_qc'`)
- `client.num_predict` capped at `METADATA_NUM_PREDICT_CAP = 512` for the metadata call (config untouched); `OllamaLLMClient` already passes `httpx` timeout = `cfg.llm.timeout_s` on every request (asserted in tests)
- `scripts/select_job.py` rewritten per card: `python scripts/select_job.py <DB_JOB_ID> [--no-llm|--mock]` — `--no-llm` (fallback metadata, managed Ollama is not started even for selected>0) / `--mock` (canned `MetadataResponse`, never a real judgement) / default (`get_llm_client(cfg, managed_ollama(cfg))` inside the managed context); selection (M5C-4) runs first (skipped when done); prints selected clips (clip_id/start-end/duration/weighted/title) + `selection.json` gate-failure reasons + a note (LLM / fallback / mock / NO LLM needed on zero selected); exit 0 on success (valid with zero clips), exit 1 on clean error; `managed_ollama` failure (port occupied etc.) → warning + fallback, still exit 0
- Tests (19 new, all CPU-only, MockLLM/fake-httpx + temp DB + synthetic artifacts; NO Ollama spawned — spawn attempts are provably prevented via a raising monkeypatch of `managed_ollama`): `tests/test_clip_plans_stage.py` (14: zero-selected empty artifact + no LLM call; valid metadata → `clip_plans.json` + `clips` rows; `LLMError`/timeout (parametrized) and invalid-schema responses → 3 attempts, fallback metadata exactly equal to `fallback_metadata`, job `queued`; `client=None` path; draft plan shape + `clip_id` format/uniqueness + verbatim cut reuse; second run skipped + byte-identical; missing `selection.json` / `schema_version` mismatch / missing job fail cleanly with hint; request carries ONLY selected-clip transcripts (out-of-window sentences absent); `OllamaLLMClient` httpx timeout per request; `num_predict` capped at 512 in the wire body) + `tests/test_select_job_script.py` (5: `--no-llm` no-spawn exit 0 with fallback; zero selected → exit 0 + reason breakdown, no spawn; default path with `managed_ollama` raising LLMError → 1 spawn attempt, warning, fallback, exit 0; `--mock` canned metadata, no spawn; missing job → exit 1)
- Also: `tests/test_selection_stage.py::test_stage_never_touches_llm_or_ollama` re-scoped to `run_selection_stage` (the module now legitimately hosts the M5C-5 `clip_plans` LLM path; the selection stage itself stays LLM-free — signature + source assertion)
- Acceptance: venv verified (`/home/rbilal/clipper/.venv/bin/python`); `python -m pytest -m "not gpu" -k "clip_plans or select_job" -v` -> 19 passed; `python -m pytest -m "not gpu" -v` -> **295 passed**; `python -m ruff check .` -> All checks passed; `python scripts/select_job.py 72e7bb26e760466ea2ca31cc0d1ac021 --no-llm` -> exit 0, 2 clips (clip01 14.04–41.21 w=10.0 "But if I'm being honest, I don't care"; clip02 36.00–52.10 w=8.0 "You see, back in 2009, we all had"), 2 gate failures printed, `clip_plans.json` + 2 `clips` rows, job `queued`; port scan confirms 11434 (user's Ollama) untouched and 11435 free
- Scope respected: NO M6 (faces/layout/crop_path) — DRAFT plans only; NO M7 render; NO changes to M5a/b/c pure logic, constants or contracts (additive `stage` wiring + script only); no new dependencies; **M5c complete**

**Current status: M0–M5b + M5C-0 (audit) + M5C-1 (selection pure) + M5C-2 (cuts pure) + M5C-3 (metadata pure) + M5C-4 (selection stage) + M5C-5 (clip_plans stage + select_job script) complete — M5C done; 295 CPU tests green, ruff clean. Next: M6 (faces/layout/crop_path) per `docs/spec/milestones.md`. PI agent must be stopped before real LLM runs (docs/spec/llm.md).**
