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
- LLM integration + managed Ollama runtime (M5a done)
- Batch processing (M5b done)
- Retry logic (M5b done)
- Scoring result storage (M5b done)
- Selection / cut refinement (M5c — not implemented yet)

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
## M5a - Managed Ollama Runtime, LLM Client, Contracts, Prompts (Complete)
- Cleanup first: `scripts/ollama_gpu1.sh` / `deploy/clipper-ollama-gpu1.service` were NOT present (nothing to delete); no docs mentioning "pipeline mode" / "coding mode" / systemd overrides existed; scope left untouched otherwise
- `docs/spec/llm.md` — fixed M5 runtime design: the user's existing Ollama on `127.0.0.1:11434` is **NEVER** modified; M5 manages an ephemeral `ollama serve` on `127.0.0.1:11435` with `CUDA_VISIBLE_DEVICES=1` because Ollama pins GPUs at process start (not per request), guaranteeing GPU1 (first P100); child reuses the existing model store (gemma4:12b, no re-download); child lives only during the M5 stage run and is always terminated (SIGTERM group -> SIGKILL after 5s -> wait for port release); GPU0 = media-only, GPU2 idle; PI agent stopped during real M5 runs; `num_ctx` is PER REQUEST (never restart per clip); MockLLM is a test double, not a model; client fallback modes recorded (strict `format=<schema>` default, `format="json"` + pydantic fallback when the server rejects the schema format)
- `docs/spec/m5.md` — ALL M5 constants fixed so M5b/M5c cannot drift: axes hook/payoff/self_contained/emotion/clarity (ints 0-10); `weighted = 0.30*hook + 0.25*payoff + 0.20*self_contained + 0.15*emotion + 0.10*clarity`; gate `weighted >= 7.2 AND hook >= 6 AND self_contained >= 7`; selection target 3 / max 5 (`cfg.clips.target_count`/`max_count`), dedupe temporal IoU > 0.30, backfill from next eligible; cut constants module-level in `src/clipper/cuts.py` (M5c): `SUGGEST_SLACK=8.0, SENTENCE_SNAP=1.0, PAD_START=0.120, PAD_END=0.250, SILENCE_SNAP=0.300, SHOT_SNAP=0.400`; cut order clamp -> sentence snap -> word padding -> silence snap -> shot snap -> validate 15..75s; batch failure after retries REJECTS those candidates (never crashes); zero selected clips is a VALID success; M5 writes DRAFT ClipPlans (`layout=center_crop`, `crop_path=[]`, `status=pending_qc`; M6 replaces layout/crop_path); `clip_id = f"{job_id}-clip{n:02d}"` (globally unique, `clips.clip_id` is PK)
- Config (`config.yaml` + `clipper.config.LLMConfig`, backward compatible): full `llm:` block per spec (provider/manage_runtime/base_url/runtime_port=11435/gpu_index=1/model=gemma4:12b/models_dir=null/num_ctx=16384/num_batch=256/temperature=0.1/num_predict=1024/timeout_s=300/startup_timeout_s=60/max_retries=2/batch_size=5/max_batch_tokens=6000/context_before_s=10.0/context_after_s=10.0); legacy `endpoint/api_key/timeout` still load (old endpoint mirrored into `base_url`); legacy configs load unchanged
- `src/clipper/llm_runtime.py`: `managed_ollama(cfg)` context manager -> base_url. Bypass when `manage_runtime=false` (yields `cfg.llm.base_url` unchanged). Fails cleanly when the port is occupied (never kills unknown processes), when it would be asked to manage 11434, when GPU1 free VRAM < 12000 MiB (`MIN_FREE_VRAM_MIB`, via nvidia-smi), when nvidia-smi is missing, when the model is absent from `/api/tags`, or when no `ollama`/`ollama_llama_server` compute process is on GPU1's index/UUID. Spawns `["ollama","serve"]` with `start_new_session=True`, env exactly `CUDA_VISIBLE_DEVICES=1`, `OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=5m`, `OLLAMA_MODELS=<cfg.llm.models_dir -> $OLLAMA_MODELS -> ~/.ollama/models>`; stdout/stderr -> `data/llm/m5-ollama.log`; polls `GET /api/tags` until `startup_timeout_s`; validates the model; warm-up `POST /api/chat` (`"Reply with exactly: OK"`, `stream=false`, `options={num_ctx:16384,num_batch:256,num_predict:8}`, `keep_alive="5m"`); verifies the compute app via `nvidia-smi --query-compute-apps`. ANY failure -> `LLMError` with reason + log path; exit ALWAYS tears the process group down (SIGTERM -> SIGKILL after 5s -> wait for port release), never orphans `ollama_llama_server`, never touches 11434
- `src/clipper/adapters/llm.py`: `LLMError`; `LLMClient` protocol `generate_json(system, user, schema) -> BaseModel`; `OllamaLLMClient(base_url, cfg)` — POST `{base_url}/api/chat` with `model`, `messages=[system,user]`, `stream=false`, `format=schema.model_json_schema()` (options `{temperature,num_ctx,num_batch,num_predict}`); documented fallback to `format="json"` + `schema.model_validate` when the server rejects the schema format (`last_mode` records which); retries up to `max_retries` on HTTP error/timeout/invalid JSON/validation error -> `LLMError`; NO silent fallback to another model/endpoint; `get_llm_client(cfg, base_url)` factory (tests monkeypatch); `MockLLM` queue of canned responses (valid/invalid JSON, invalid schema, exceptions) that records calls with zero network I/O
- New contracts (additive, existing untouched, schema_version=1) in `src/clipper/models/scoring_artifacts.py`: `CandidateContext{cand_id,start,end,transcript,context_before="",context_after=""}`, `ScoringBatch{schema_version=1, candidates: list[CandidateContext]}` (up to 5 — the REQUEST), `ScoringResponse{scores: list[ScoredClip]}` (what the model returns), `ScoresArtifact{schema_version=1, model, scores: list[ScoredClip]}`; re-exported from `clipper.models`; mirrored in `docs/spec/contracts.md`
- `src/clipper/prompts.py`: `SCORING_SYSTEM` single source of truth (harsh 0-10 rubric, "7+ means you'd bet money on it", ±8s sentence-boundary suggested in/out, reject mid-thought starts / pre-payoff ends, JSON-only ScoringResponse, one good + one bad few-shot example); mirrored VERBATIM into `docs/spec/scoring_prompt.md` (that file previously held a different draft rubric — the prompt constant is now canonical)
- `scripts/llm_preflight.py`: `with managed_ollama(cfg):` tiny `generate_json` sanity check; prints model/base_url/gpu_index/GPU UUID/free VRAM/compute apps; exit 0/1; child cleaned up on success AND failure
- Dependency: `httpx` only (0.28.1); NO `ollama` python package; NO real Ollama/GPU/nvidia-smi in tests (httpx, subprocess and nvidia-smi mocked; teardown exercised against a real `sleep` child in its own session)
- Tests (32 new, all CPU-only): `tests/test_llm_config.py` (defaults + legacy configs), `tests/test_scoring_contracts.py` (round-trips + existing `Candidate` untouched), `tests/test_llm_client.py` (request body correctness, valid response validates, invalid JSON / invalid schema / HTTP 500 / timeout each retry 3x -> `LLMError`, retry-then-success, schema-rejected -> `format="json"` fallback recorded, MockLLM queue/exceptions/empty/records), `tests/test_llm_runtime.py` (spawn cmd/env/port/session + log path, process group killed on normal exit AND on exception, clean failures: port occupied / model missing / wrong GPU / insufficient VRAM / nvidia-smi missing / refuses 11434, `manage_runtime=false` bypass, preflight success+failure exit codes)
- Acceptance: `.venv/bin/python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"` OK; `.venv/bin/python -m pytest -m "not gpu" -v` -> 145 passed (113 prior + 32 new); `.venv/bin/python -m ruff check .` -> All checks passed
- M5a scope respected: NO scoring stage, NO selection, NO cut snapping (M5b/M5c), NO `clips` writes, M6/M7 GPU spec untouched
## M5b — Scoring Stage (Complete)
- `src/clipper/scoring.py` (PURE, no I/O) — fixed M5b rules from `docs/spec/m5.md`:
  - `build_context(candidate, sentences, cfg) -> CandidateContext`: transcript = sentences[sentence_range] (INCLUSIVE) joined by single spaces; `context_before` = sentences overlapping `[start - cfg.llm.context_before_s, start)`; `context_after` = sentences overlapping `(end, end + cfg.llm.context_after_s]`
  - `estimate_tokens(text) = max(1, len(text)//4)`
  - `pack_batches(contexts, batch_size, max_batch_tokens) -> list[ScoringBatch]`: greedy, order-preserving; 1500-token prompt-overhead reserve (`PROMPT_OVERHEAD_TOKENS`); per-candidate cost = est(transcript)+est(before)+est(after)+60 (`PER_CANDIDATE_OVERHEAD_TOKENS`); new batch at `batch_size` (5) OR when adding would exceed budget; >=1 candidate per batch; an oversized single candidate truncates context_before/context_after FIRST (before first, then after) — NEVER the transcript
  - `validate_scores(response, batch) -> list[ScoredClip]`: exactly one score per cand_id (missing/extra/duplicate => `ValueError` => batch retry); axes must be `int` in 0..10 (violation => `ValueError`); `suggested_in/out` outside `[cand.start-8, cand.end+8]` (`SUGGEST_SLACK=8.0`) => scores KEPT, `reject_reason="suggestion_out_of_bounds"`; returns clips in batch candidate order
  - `weighted_score(clip)`: 0.30*hook + 0.25*payoff + 0.20*self_contained + 0.15*emotion + 0.10*clarity (selection itself is M5c)
- `src/clipper/stages/score.py`: `run_scoring(job_id, cfg, conn, client: LLMClient)`:
  - requires `candidates.json` (M4b, `scripts/candidates_job.py`) + `sentences.json` (M2b, `scripts/transcribe_job.py`) + `probe.json` (M1 ingest) — clean `StageError` naming the missing artifact AND its producing milestone/script; `schema_version == 1` validated on all inputs
  - stage name `scoring`; skip only when DB state=done AND `scores.json` exists AND schema valid
  - job `queued -> running -> queued` on success; stage + job `failed` + clean `StageError` on error
  - builds contexts for ALL candidates in artifact order; `client.generate_json(SCORING_SYSTEM, batch_json, ScoringResponse)` per batch with `validate_scores`; retries up to `cfg.llm.max_retries` on `LLMError`/semantic `ValueError` (incl. schema-validation failure); after retries: batch's candidates are REJECTED (all axes 0, `reject_reason="llm_batch_failed: <msg>"`) — the job NEVER fails for a batch
  - writes `scores.json` as `ScoresArtifact(model=cfg.llm.model, scores=<ALL candidates in original order>)`; updates `candidates.score_json` per `cand_id` (M4b rows); zero candidates is a valid success (zero LLM calls)
  - re-exports: `stages/__init__` now exposes `run_scoring`
- `scripts/score_job.py <JOB_ID> [--mock]`: default = ONE managed ephemeral Ollama (`managed_ollama(cfg)`, docs/spec/llm.md) serving every batch then killed (success AND failure), `get_llm_client(cfg, base_url)` inside; `--mock` = `_MockScorer` deterministic wiring-test ONLY (position j in batch -> `10 - (2j)%11` on all axes, `suggested_in=cand.start`, `suggested_out=cand.end`; never a real judgement); prints job/model/batches/scored/rejected/top-5-by-weighted/artifact; exit 0 on success (incl. all-rejected), exit 1 clean error
- Tests (47 new, all CPU-only — MockLLM + temp DB + synthetic artifacts, NO Ollama/GPU/nvidia-smi): `tests/test_scoring.py` (25: build_context transcript slice + before/after windows + video start/end edges + cfg window sizes; estimate_tokens; pack_batches budget/size/order/oversized-truncates-context/not-transcript/whole-budget-exceeded; validate_scores missing/extra/duplicate/axis-range/bool/float-axes/OOB-keeps-scores/edge-in-bounds/model-reject-preserved) and `tests/test_scoring_stage.py` (22: artifact + stage row + `candidates.score_json` updates; skip on second run; OOB policy in stage; zero-candidate success (no LLM calls); batch failure rejects only that batch (job succeeds, score_json reflects it); retry-then-success; all batches fail -> artifact written, job queued, zero usable scores; semantic-validation failures are retryable; missing candidates/sentences/probe fail with the right milestone hint; schema_version mismatch; missing job; retry-after-failure; byte-identical `scores.json` across two fresh runs with the same mock; script success/missing-job/failure exit codes + mock position-derivation)
- Acceptance: `source .venv/bin/activate` -> venv python verified; `python -m pytest -m "not gpu" -v` -> 192 passed (145 prior + 47 new); `python -m ruff check .` -> All checks passed; `python scripts/score_job.py 72e7bb26e760466ea2ca31cc0d1ac021 --mock` -> exit 0 (1 batch, 4 scored, 0 rejected; re-run skips the completed stage)
- M5b scope respected: NO selection/cut refinement/metadata (M5c), M5a runtime/client/contracts/constants/prompts/config defaults untouched, no new dependencies

## M5C-1 — Selection pure logic + contracts (Complete)
- Branch `m5c1-selection-pure`; selection pure logic + contracts ONLY (no cuts, no metadata, no stage changes)
- Contracts `src/clipper/models/selection_artifacts.py` (schema_version=1): `ClipDecision`, `SelectionArtifact`
- `src/clipper/selection.py` (PURE): `weighted_score` (delegates to `scoring.weighted_score`), `passes_gate` from `cfg.clips.gate.*` (7.2/6/7), `temporal_iou` re-export, `select_clips` (eligible -> rank weighted/cheap/start -> IoU > 0.30 suppress -> target 3 / max 5 -> decision record for EVERY candidate)
- `config.py`: `ClipGateConfig` + `ClipsConfig.gate` (backward compatible)
- Tests: `tests/test_selection.py` (20 passed); suite 225 passed; ruff clean

## M5C-2 — Cut refinement pure logic (Complete)
- Branch `m5c2-cuts-pure`; scope: `src/clipper/cuts.py` PURE logic only — NO stage, NO metadata, NO LLM, NO file I/O
- Implementation verified against `docs/spec/m5.md` (already present from M5c WIP, committed on main as `be5205d`; no logic changes needed):
  - constants verbatim: `SUGGEST_SLACK=8.0`, `SENTENCE_SNAP=1.0`, `PAD_START=0.120`, `PAD_END=0.250`, `SILENCE_SNAP=0.300`, `SHOT_SNAP=0.400`
  - `refine_cut(candidate, score, sentences, words, silence_regions, shots, duration_s) -> (start, end)` in the STRICT spec order: clamp to `[cand.start-8, cand.end+8] ∩ [0,duration]` -> sentence snap (nearest within 1.0s; documented fallback for beyond-1.0s per `docs/spec/m5c_status.md` judgment note) -> word padding (`first word .start - 0.120` / `last word .end + 0.250`) -> clamp `[0,duration]` -> silence snap (<= 0.300) -> shot snap (<= 0.400) -> validate `start < end`, `15.0 <= dur <= 75.0`, >= 1 full sentence; `CutError` carries a specific reason
- Tests `tests/test_cuts.py` (16 CPU-only golden cases): constants mirror; sentence snap + exact word padding; silence snap within 0.300; shot snap within 0.400 + beyond-tolerance ignored; clamping to candidate ±8 and video bounds ([0, duration]); mid-gap suggestion; too-short/too-long/inverted/no-sentence rejections with specific reasons; no silence/shots still works (new); determinism (new)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -k "cuts" -v` -> 16 passed; `python -m pytest -m "not gpu" -v` -> 227 passed; `python -m ruff check .` -> All checks passed
- M5C-2 scope respected: NO selection stage, NO clip_plans stage, NO metadata, NO script; no new dependencies; M6 not started

## M5C-3 — Metadata contracts, prompt, validation, fallback (Complete)
- Branch `m5c3-metadata-pure` (based on `m5c2-cuts-pure`, which contains main + M5C-1/2 docs/tests); scope: metadata contracts + prompt + PURE helpers ONLY — NO selection stage, NO clip_plans stage, NO script
- Contracts already existed from the M5c WIP (on main as `be5205d`) and were verified, not re-invented: `ClipMetadata{cand_id, title, hook_overlay, description, hashtags}`, `MetadataResponse{items: list[ClipMetadata]}` in `src/clipper/models/selection_artifacts.py`, registered + re-exported from `clipper.models`. Judgment note (recorded per project rule): the `schema_version=1` project rule applies to STORED artifacts (`SelectionArtifact`, `ClipPlansArtifact` both carry it); LLM-facing *response* contracts intentionally carry NO `schema_version`, matching the established `ScoringResponse` pattern (adding one would force the LLM to emit the field)
- `prompts.py::METADATA_SYSTEM` already present (audited M5C-0): title ≤80; hook ≤6 words AND ≤40 chars; description ≤300; 3-6 hashtags each `'#'`-prefixed, no spaces; JSON only, `{"items":[...]}` matching `MetadataResponse`; no clickbait lies; input contains ONLY the selected clips' final transcript — never the full video transcript
- NEW `src/clipper/metadata.py` (PURE, no I/O, no LLM): constants `TITLE_MAX=80`, `HOOK_MAX_WORDS=6`, `HOOK_MAX_CHARS=40`, `DESCRIPTION_MAX=300`, `HASHTAG_MIN=3`, `HASHTAG_MAX=6`, `FALLBACK_TITLE_WORDS=8`, `FALLBACK_HOOK_WORDS=5`, `FALLBACK_DESCRIPTION_CHARS=200`, `REQUEST_CLIP_KEYS=(cand_id, duration_s, niche, transcript)`. `validate_metadata(response, selected_cand_ids) -> list[ClipMetadata]` (exactly one item per selected cand_id, no extras/dupes/omissions; all field limits enforced; `ValueError` names the cand_id). `fallback_metadata(cand_id, transcript, niche) -> ClipMetadata` (deterministic: title = first 8 words, hook = first 5 words, description = first 200 chars, hashtags = `['#<niche>']` or `['#clip']` when niche is empty/blank; char-clipped to the limits as a safety net so ANY transcript stays within bounds). `build_metadata_request(selected) -> str` (compact JSON `{"clips":[...]}`; whitelists exactly the 4 per-clip keys — extras like a full-video transcript are dropped; input order preserved)
- Tests `tests/test_metadata.py` (35 new, CPU-only, no LLM): contract round-trips + required-field / field-type failures; validation (missing/extra/wrong/duplicate cand_id, count mismatch, empty title/hook/description, over-length title & hook [7 words / 41 chars] & description, boundary accepts [80/40/300 chars, 3 and 6 hashtags], bad hashtags [no `#`, inner space, bare `#`, <3, >6], cand_id named in errors); fallback (exact values, `#clip` default for empty/blank niche, determinism, within limits for normal AND hostile inputs incl. 500-char word and empty transcript); request (whitelisted keys only — full-video-transcript marker NOT carried, unknown keys dropped; order preserved; compact output; zero-clip `{"clips":[]}`); prompt (limits mirror the module constants verbatim; `no clickbait lies`; `Return JSON ONLY`; `"items"` schema; one item per clip same order; selected-clips-only input)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -k "metadata" -v` -> 35 passed; `python -m pytest -m "not gpu" -v` -> 262 passed; `python -m ruff check .` -> All checks passed
- M5C-3 scope respected: NO stage work, NO script, no changes to existing contracts / prompt / constants; no new dependencies; M6 not started

## M5C-4 — Selection Stage (Complete)
- Branch `m5c4-selection-stage`; scope: the `selection` stage ONLY — NO metadata, NO clip_plans, NO script; NO LLM anywhere (no Ollama import, no client parameter on `run_selection_stage`)
- `src/clipper/stages/select.py`: `run_selection_stage(job_id, cfg, conn)` — single `selection` stage; requires `candidates.json` (M4b), `scores.json` (M5b), `sentences.json` + `words.json` (M2b), `silence.json` + `shots.json` (M3b), `probe.json` (M1) — clean `StageError` naming the missing artifact AND producing milestone/script; `schema_version == 1` validated on every input; skip only when DB stage=done AND valid `selection.json` exists; job `queued->running->queued`, failure marks stage+job `failed`
- Rank/gate/dedupe via pure `select_clips`; refine in RANK order via `refine_cut`; on `CutError` record `cut_reject_reason` and BACKFILL from the next eligible, non-overlapping (IoU > 0.30) candidate until `min(target_count, max_count)` reached or the eligible list exhausted
- `ClipDecision` extended with `refined_start`/`refined_end` (default `None`): the refined cut persisted on SELECTED decisions so M5C-5 (`clip_plans`) can reuse the exact cut; `stages/__init__` re-exports `run_selection_stage` (keeps the `run_selection` alias; client parameter ignored / never called)
- Writes `selection.json` (`SelectionArtifact`, schema_version=1) with a decision for EVERY candidate (artifact order); zero selected is a VALID success
- Tests `tests/test_selection_stage.py` (14 new, CPU-only, temp DB, synthetic artifacts): artifact + stage row + job state; one decision per candidate with `refined_start`/`refined_end` matching `refine_cut` exactly; backfill after `CutError` reaches target (3 selected, 1 cut-rejected with specific reason); zero-selected valid success (stage done, job queued); skip on second run (byte-identical artifact); per-file missing-input failures (all 7 inputs) with correct milestone hints, stage+job marked failed; `schema_version` mismatch fails cleanly; missing job (no stage row created); LLM-free assertion (`OllamaLLMClient`/`managed_ollama` monkeypatched to RAISE, no `client` in the call signature, no LLM surface in the module)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -k "selection_stage or select_stage" -v` -> 14 passed; `python -m pytest -m "not gpu" -v` -> 276 passed; `python -m ruff check .` -> All checks passed
- M5C-4 scope respected: NO clip_plans stage, NO metadata call, NO script; no changes to M5a/b/c pure logic, constants or contracts (only additive `ClipDecision` fields); no new dependencies; M6 not started

## M5C-5 — Clip Plans Stage (Complete)
- Branch `m5c5-clip-plans-stage`; scope: the `clip_plans` stage + `scripts/select_job.py` ONLY; NO M6 work (DRAFT plans stay `layout=center_crop` / `crop_path=[]` / `status=pending_qc`); the metadata failure NEVER fails the job (always fallback metadata)
- `src/clipper/stages/select.py`: `run_clip_plans_stage(job_id, cfg, conn, client)` — single `clip_plans` stage; requires DONE `selection` stage + valid `selection.json` (and `candidates.json`/`sentences.json`) — clean `StageError` otherwise; skip only when DB stage=done AND valid `clip_plans.json` exists (byte-identical on re-run); ZERO selected → `clip_plans.json = {schema_version:1, plans:[]}`, NO LLM call, job `queued`
- One metadata call (`generate_json`, `METADATA_SYSTEM` + `build_metadata_request(selected)`; request contains ONLY the selected clips' final transcripts — never the full video) → 1 DRAFT `ClipPlan` per selected clip, `clip_id = f"{job_id}-clip{n:02d}"` (rank order, unique, format-verified); retries (`cfg.llm.max_retries`) then deterministic `fallback_metadata` on `LLMError`/invalid-schema — job STILL succeeds; every plan upserted into `clips` (`plan_json`, `status='pending_qc'`); `client.num_predict` capped at `METADATA_NUM_PREDICT_CAP = 512` for the call (config untouched)
- `scripts/select_job.py` rewritten per card: `python scripts/select_job.py <DB_JOB_ID> [--no-llm|--mock]` — `--no-llm` (fallback, managed Ollama never started) / `--mock` (canned `MetadataResponse`, wiring test only) / default (`get_llm_client(cfg, managed_ollama(cfg))` inside the managed context, child killed on success AND failure); selection (M5C-4) first (skipped when done); prints selected clips (clip_id/start-end/duration/weighted/title) + gate-failure reasons from `selection.json` + source note; exit 0 on success (valid with zero clips), exit 1 clean error; `managed_ollama` failure → warning + fallback, still exit 0
- Tests: `tests/test_clip_plans_stage.py` (14 new) + `tests/test_select_job_script.py` (5 new) — 19 total, all CPU-only (MockLLM / fake httpx transport + temp DB + synthetic artifacts; spawn attempts provably prevented via a raising monkeypatch of `managed_ollama`): zero-selected empty artifact + zero LLM calls; valid metadata → artifact + `clips` rows; `LLMError`/timeout + invalid-schema retry-exhaustion → exact `fallback_metadata`, job `queued`; `client=None` path; draft-shape + `clip_id` format/uniqueness + verbatim cut reuse; second run skipped (byte-identical); missing `selection.json` / `schema_version` mismatch / missing job fail cleanly with hint; request-scoped transcripts (out-of-window sentences absent); `OllamaLLMClient` httpx timeout = `cfg.llm.timeout_s` on every request; `num_predict` capped at 512 in the wire body; script: `--no-llm` no-spawn exit 0, zero-selected exit 0 + breakdown + no spawn, default path managed-Ollama-failure → 1 spawn attempt + fallback + exit 0, `--mock` canned no spawn, missing job exit 1
- `tests/test_selection_stage.py::test_stage_never_touches_llm_or_ollama` re-scoped to `run_selection_stage` (the module now legitimately hosts the M5C-5 `clip_plans` LLM path; the selection stage itself stays LLM-free)
- Acceptance: venv verified; `python -m pytest -m "not gpu" -k "clip_plans or select_job" -v` -> 19 passed; `python -m pytest -m "not gpu" -v` -> **295 passed**; `python -m ruff check .` -> All checks passed; `python scripts/select_job.py 72e7bb26e760466ea2ca31cc0d1ac021 --no-llm` -> exit 0 (2 clips: clip01 14.04–41.21 w=10.0; clip02 36.00–52.10 w=8.0; 2 gate failures printed; `clip_plans.json` + 2 `clips` rows; job `queued`); user's Ollama on 11434 untouched, ephemeral port 11435 verified free (no residual child)
- M5C-5 scope respected: NO M6 (faces/layout/crop_path) — DRAFT plans only; NO M7; no changes to M5a/b/c pure logic, constants or contracts (additive stage + script wiring only); no new dependencies; **M5c complete**

## M6a — Per-Clip Vision Segments & Contracts (Complete)
- Branch `m6a-vision-segments`; scope: decisions + config + contracts + vision-segment adapter + CPU-only tests ONLY — NO face detection / tracking / layout / crop-path math / stage / script (M6b+), M7 untouched, DRAFT ClipPlans unchanged (`layout=center_crop`, `crop_path=[]`, `status=pending_qc`)
- `docs/spec/m6.md` fixed M6 decisions: GPU0 (2080 Super, `cuda:0`) only — P100s never do video; analyze ONLY selected clip ranges; 960x540 fit, aspect preserved, NO padding, even dims, `fps=min(native,30.0)` (native 0/unknown → 30.0); NVDEC (`-hwaccel cuda`) + NVENC (`h264_nvenc -preset p4 -cq 23`); detect EVERY frame (no sparse sampling — the pipeline.md 3fps YuNet line is superseded for M6); crop tracks at NATIVE fps via linear interpolation, `t` CLIP-RELATIVE (t=0 at clip start), coords in SOURCE pixels; no face → static `center_crop` (tracking never forced); roles `main|top|bottom|face`; CPU faces only behind explicit `--cpu-faces`, logged loudly (M6b); per-clip artifacts in `data/jobs/<job_id>/clips/<clip_id>/`. Card's `docs/spec/gpu.md` does not exist — rules sourced from the project skill + `docs/spec/pipeline.md` (noted in spec)
- Config (backward compatible): `devices.vision="cuda:0"`; `faces.vision_fps_cap=30.0`, `faces.vision_width=960`, `faces.vision_height=540`; existing configs load unchanged
- Contracts `src/clipper/models/vision_artifacts.py` (re-exported from `clipper.models`, mirrored in `docs/spec/contracts.md`): `VisionSegmentInfo` (not a persisted artifact — mirrors ProxyInfo/AudioInfo), `FaceDetection` (x/y/w/h normalized 0..1 to the vision frame), `RegionTrack {role, keys: list[CropKey]}`, `ClipLayoutArtifact` (schema_version=1; M6b writes it)
- `src/clipper/adapters/vision.py`: `VisionAdapter` protocol; `FFmpegVisionAdapter` (exact spec command — input-seek `-ss/-to`, fit-then-even `-vf` scale chain, result probed for width/height/fps/`nb_frames` with `round((end-start)*fps)` fallback; NVDEC/NVENC failure/timeout → `MediaError` with stderr tail, NO silent CPU/libx264 fallback); `FixtureVisionAdapter` (placeholder + canned info); `get_vision_adapter(cfg)` factory (singleton like `get_adapter`; `fixture`/`cpu`/`mock` → fixture). Judgment notes in m6.md: optional `clip_id="unknown"` kwarg on top of the card's 7 positional params; internal 30.0 fps clamp in `make_segment`
- Tests `tests/test_vision.py` (31 new, CPU-only, subprocess mocked): contract round-trips + role/layout_source restrictions; fps 60→30 / 24→24 / 0→30 (+cap override); command asserts (hwaccel cuda, h264_nvenc, p4/cq23, no pad filter, no libx264, seek order); fixture validity; failure → `MediaError` + stderr tail, exactly one ffmpeg call (no fallback/retry); factory + caching; protocol conformance; config defaults
- Acceptance: venv verified; `python -m pytest -m "not gpu" -v` → **334 passed** (303 pre-M6a + 31 new; ~7 min wall — slow tests, no failures); `python -m ruff check .` → All checks passed
- M6a scope respected: NO face detection / tracking / layout / crop path / stage / script; NO M7; no changes to M0–M5 logic, constants, contracts or defaults (additive config keys + new modules only); no new dependencies (ffmpeg via subprocess, reuses `FFmpegMediaAdapter` probe)

## M6b — YuNet ONNX Face Detector on GPU0 (Complete)
- Branch `m6b-gpu-face-detector`; scope: face DETECTOR ONLY — decode/NMS helpers + `FaceDetector` protocol/impls + factory + model install script + GPU preflight + CPU-only tests. **NO tracking, NO layout decision, NO crop path, NO stage, NO stage script** (later M6); M7 untouched; DRAFT ClipPlans unchanged
- `docs/spec/m6.md` M6b section: GPU path = YuNet ONNX via onnxruntime `CUDAExecutionProvider` on GPU0 (fails LOUDLY via `FaceDetectorError` when the CUDA EP is unavailable/unused — never silent CPU); CPU path = `cv2.FaceDetectorYN` explicit debug only behind `get_face_detector(cfg, cpu=True)` (decision 9, logged loudly); lazy `onnxruntime` import + dependency-free pure `clipper.facedec`; YuNet anchor constants module-level (`YUNET_STRIDES=(8,16,32)`, `YUNET_MIN_SIZES=(16,32,64)`, aspect 1.0, variance 1.0, fg index -1, model `iou` output unused — hard greedy NMS); deterministic decode (anchor order) + NMS (score DESC, ties index ASC, suppress IoU >= `nms_threshold`, `top_k` cap); model file never downloaded in tests
- Config (backward compatible): `faces.model_path="data/models/yunet.onnx"`, `faces.score_threshold=0.6`, `faces.nms_threshold=0.3`, `faces.top_k=5000`, `faces.det_size=640`
- Dependency added: `onnxruntime-gpu` ONLY (torch/ultralytics/insightface NOT; `opencv-python-headless` retained, `opencv-python` forbidden)
- `src/clipper/facedec.py` (PURE, no I/O, no numpy import): `anchor_grid`, `decode_yunet` (stable 2-class softmax score filter + box decode/clamp), `nms`, `normalize_box`, `_iou` — all deterministic, unit-testable with synthetic tensors
- `src/clipper/adapters/faces.py`: `FaceDetector` protocol (`detect(frame_bgr, t) -> list[FaceDetection]` normalized 0..1 / `close()` / `name()`); `YuNetOnnxDetector` (lazy `onnxruntime`; session `providers=["CUDAExecutionProvider","CPUExecutionProvider"]` + `provider_options=[{"device_id": <index from cfg.devices.vision>}, {}]`; loud `FaceDetectorError` if the CUDA EP is missing/unselected; resize to `det_size`, decode + NMS, normalized boxes; `name()="yunet-onnx-cuda"`); `Cv2CpuFaceDetector` (`name()="yunet-cpu"`, debug only); `FixtureFaceDetector` (canned detections per frame index, records calls); `get_face_detector(cfg, cpu=False)` factory; `parse_cuda_device_id` (GPU0-only rule)
- `scripts/install_models.sh` (manual ONLY, never in tests): OpenCV Zoo `face_detection_yunet_2023mar.onnx` → `data/models/yunet.onnx`, skip-if-present, prints target path
- `scripts/faces_preflight.py`: prints onnxruntime version + `get_available_providers()`, asserts CUDA EP, loads the model, runs one dummy 640x640 frame, prints detection count + device actually used; exit 0/1 (manual run — GPU0 must be free)
- Tests `tests/test_faces.py` (35, CPU-only — mocked onnxruntime/cv2, no model file, no GPU): `decode_yunet` synthetic tensors (zero-delta exact anchor box, score filtering, threshold-empty, box sanity + deterministic order, clamps, off-canvas drop, mismatched shapes, channel-first shapes); `nms` (suppression, disjoint kept, borderline IoU, strong overlap, `top_k`, determinism + tie-break, empty); `normalize_box` (known box, unit-square clamp, degenerate reject); `FixtureFaceDetector` (canned + copies + call recording); `YuNetOnnxDetector` loud failures (CUDA EP unavailable / onnxruntime missing / session loses CUDA, all via mock `onnxruntime`); CUDA `device_id` wiring + provider_options; missing model; `parse_cuda_device_id` GPU0-only; factory default-vs-cpu routing + cpu-without-model failure; config M6b defaults + M6a defaults intact; module import never pulls onnxruntime; `facedec.py` source dependency-free; PLUS optional oracle `test_gpu_oracle_yunet_onnx_matches_cv2` marked `@pytest.mark.gpu` (skipif model/CUDA missing; IoU >= 0.9 against `cv2.FaceDetectorYN`; EXCLUDED from `-m "not gpu"`)
- Acceptance: venv verified (`/home/rbilal/clipper/.venv/bin/python`); `python -m pytest -m "not gpu" -q` → **368 passed, 1 deselected** (the GPU oracle; ~7 min wall); `python -m ruff check .` → All checks passed
- Scope respected: NO tracking / layout / crop-path / stage / stage script; NO M7; no changes to M0–M6a logic, constants, contracts or defaults (additive config keys + new modules only); no model downloads or GPU in any CPU test

## M6c — Face Tracking and Track Statistics (Complete)
- Branch `m6c-tracking`; scope: PURE tracking + track statistics ONLY — **NO layout decision, NO crop path, NO stage, NO stage script** (later M6); M7 untouched; DRAFT ClipPlans unchanged; no new dependencies
- `docs/spec/m6.md` M6c section (decisions + constants + test list) and `docs/spec/contracts.md` `TrackSummary` mirror (in-memory contract, NOT a persisted artifact — no `schema_version`, mirroring `VisionSegmentInfo`/`ProxyInfo`)
- `src/clipper/tracking.py` (PURE — imports only `clipper.models`; no I/O, no detector imports, no numpy, no GPU): `associate(detections_by_frame, iou_threshold=0.3, max_gap_frames=3) -> list[FaceTrack]` (greedy IoU frame-to-frame vs each track's latest box; greedy by IoU DESC, ties track_id ASC / detection-index ASC; track IDs `0,1,2,…` in FIRST-APPEARANCE order; boxes `(t,x,y,w,h)` M6a-normalized; a track survives ≤ `max_gap_frames` consecutive misses, then is finalized — re-appearance starts a NEW track; empty frames count as gaps for all active tracks); `track_stats(track, frame_count, frame_w, frame_h) -> dict` (`persistence = frames_present/frame_count`, `mean_area = mean(w*h)`, `corner_affinity` = quadrant in which the box CENTROID spends ≥ 0.8 of the track's frames AND `mean_area < 0.04`, else `None`; also `mean_cx`; `frame_w`/`frame_h` API-symmetry only — boxes normalized); `summarize(tracks, frame_count, frame_w, frame_h) -> TrackSummary` (dominant = highest persistence, tie-break LARGER mean_area, then smallest tid; persistent = persistence ≥ 0.6; facecam candidate = corner_affinity set AND persistence ≥ 0.8 AND mean_area < 0.04; side_by_side_pair = two persistent tracks on opposite sides of cx=0.5 with mean_area within 2x — best pair by combined persistence, ties smaller tid-sum, else None; zero tracks → empty summary, no crash)
- `tracking.py` constants: `CORNER_AFFINITY_MIN_FRACTION=0.8`, `CORNER_AFFINITY_MAX_AREA=0.04`, `PERSISTENT_THRESHOLD=0.6`, `FACECAM_PERSISTENCE=0.8`, `FACECAM_MAX_AREA=0.04`, `SIDE_BY_SIDE_MAX_RATIO=2.0`
- Contract: new `TrackSummary` in `clipper.models` (`dominant_track_id: int|None`, `persistent_track_ids: list[int]`, `facecam_candidate_track_ids: list[int]`, `side_by_side_pair: tuple[int,int]|None` — JSON round-trip tested). Judgment note (no-drift rule): `associate` returns `FaceTrack` with `persistence=0.0` / `corner_affinity=None` (consumers fill them via `track_stats`) — matches the card's `track_stats -> dict` API
- Tests `tests/test_tracking.py` (34, CPU-only, synthetic `FaceDetection`s — no GPU/I/O/detector/model files): single face smooth → one track, persistence 1.0; two faces side by side → two tracks + side_by_side detected; small corner face → correct quadrant (tl/tr/bl/br) + facecam candidate; large centered face → `corner_affinity=None`, not facecam; short occlusion ≤ gap → same track; long gap → new track; empty input / all-empty frames → empty summary no crash; determinism + stable first-appearance track_id ordering; 80% quadrant boundary (≥ vs <); high/low IoU threshold effects; dominant-track selection; `TrackSummary` model round-trip
- Acceptance: venv verified (`/home/rbilal/clipper/.venv/bin/python`); `python -m pytest -m "not gpu" -k "tracking" -v` → 34 passed; `python -m pytest -m "not gpu" -v` → **402 passed, 1 deselected** (the M6b GPU oracle; ~7 min wall); `python -m ruff check .` → All checks passed (8 style fixes in `tracking.py`/`test_tracking.py`) — NOTE: the earlier PI run left ruff dirty + docs un-updated + work uncommitted; all fixed and committed here
- Scope respected: NO layout decision / crop path / stage / stage script; NO M7; no changes to M0–M6b logic, constants, contracts or defaults (additive `TrackSummary` model + new pure module only); no new dependencies; no GPU/model in any CPU test

## M6d — Layout Decision and Crop Track Generation (Complete)
- Branch `m6d-layouts-croppath`; scope: PURE layout decision + crop-track math ONLY — **NO stage, NO stage script, NO wiring, NO render** (later M6/M7); M7 untouched; DRAFT ClipPlans unchanged (`layout=center_crop`, `crop_path=[]`, `status=pending_qc`); no new dependencies
- `docs/spec/m6.md` M6d section (decisions + judgment notes + test list)
- `src/clipper/layouts.py` (PURE — imports only `math` + `clipper.models`; no I/O / GPU / numpy / adapters): `decide_layout(summary, source_w, source_h) -> Layout` (precedence facecam > split_screen > face_track > center_crop; **no face is a NORMAL state → center_crop**; decision is geometry-free per the judgment note); `build_tracks(layout, summary, source_w, source_h, native_fps, duration_s, shot_boundaries, tracks) -> list[RegionTrack]` (card signature + REQUIRED final `tracks: list[FaceTrack]` param — judgment note: `TrackSummary` carries only ids, so the M6c `associate` output is needed for the EMA/centering math). Every track emits exactly `round(duration_s * native_fps) + 1` CLIP-RELATIVE keys at `0.0, 1/fps, 2/fps, …`; coordinates are SOURCE pixels, integers, inside the source; `w`/`h` constant per track. center_crop = one STATIC `main`, largest EXACT 9:16 integer window (`k = min(source_w // 9, source_h // 16)`, `w=9k, h=16k`); face_track = full-height window `w = int(round(h * 9/16))` (clamped `<= source_w` — pathological portrait case), x follows the dominant centroid target clamped to `[0, source_w - w]` with EMA `alpha = 1 - exp(-dt / EMA_TAU_S)`, central 20% dead zone, velocity clamp `|dx| <= 0.30 * w * dt`, HARD RESET at shot boundaries (event at the boundary time, no cross-cut ramp); split_screen = STATIC `top`/`bottom` on pair members' mean centroids; facecam = STATIC exact-9:16 `main` + STATIC `face` square (`round(max(mean face w,h) * 4.0)`) on the first (smallest-id) candidate's centroid. Validation: non-positive dims/fps and negative duration → `ValueError`; missing track for a required layout → `ValueError` naming layout + track id
- `layouts.py` constants: `EMA_TAU_S = 0.8`, `DEAD_ZONE = 0.20`, `MAX_PAN_FRAC_PER_S = 0.30`, `ASPECT = 9/16` (exact in binary FP), `FACECAM_ENLARGE = 4.0` (new — judgment note)
- Tests `tests/test_layouts.py` (23, CPU-only, synthetic `FaceTrack`/`TrackSummary` — matching the spec's test list): no faces → center_crop static exact-9:16; single still face → static face_track; moving face → in-bounds motion; EMA smoothing; dead-zone jitter ⇒ zero motion; velocity clamp; shot-boundary hard reset; split_screen top/bottom; facecam main+face; key count `round(duration*fps)+1` and `t[0] == 0.0` in ALL layouts (incl. 24fps/1.7s); int coords in bounds; decide_layout precedence; ValueError guards; determinism (byte-identical `model_dump_json`)
- Acceptance: venv verified (`/home/rbilal/clipper/.venv/bin/python`); `python -m pytest tests/test_layouts.py -v` → **23 passed** (full suite deferred per operator — last full green run: 402 passed + 1 GPU oracle deselected at M6c; M6d adds 23 tests, no other files touched); `python -m ruff check .` → All checks passed
- Scope respected: NO stage / stage script / detector→tracks→layout wiring / M7; no changes to M0–M6c logic, constants, contracts or defaults (new pure module + tests + spec section + progress notes only); no new dependencies; no GPU/model in any CPU test

## Current Status (updated for M6d)
M0, M1, M2, M3a, M3b, M4a, M4b, M5a, M5b, M5C-0 (audit), M5C-1 (selection pure + gate config), M5C-2 (cuts pure + tests), M5C-3 (metadata contracts/prompt/pure validation + fallback), M5C-4 (selection stage + 14 stage tests), M5C-5 (clip_plans stage + select_job script, 19 tests), M6a (vision segments + contracts, 31 tests), M6b (YuNet ONNX face detector on GPU0, 35 tests incl. 1 GPU oracle), M6c (face tracking + track statistics pure, 34 tests), **M6d (layout decision + crop-track math pure, 23 tests)** complete. **M6d is GREEN** — 23 new CPU tests green + ruff clean (full-suite re-run deferred per operator; last full green run 402 passed, 1 GPU oracle deselected at M6c). Manual M6b GPU verification still pending: `bash scripts/install_models.sh` + `python scripts/faces_preflight.py` (GPU0 free, PI agent stopped). Next: M6 stage (per-clip `layout.json`/`face_detections.json` under `data/jobs/<job_id>/clips/<clip_id>/` + stage script wiring detector→tracks→layout) per `docs/spec/m6.md` decision 10. PI agent must be stopped before real LLM/GPU runs (docs/spec/llm.md; GPU0-only for video, docs/spec/m6.md decision 1).

<!-- Previous status (M6c) is preserved above as per-milestone entries; this section is the single latest state. -->
