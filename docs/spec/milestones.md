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
