---
name: clipper-project
description: Project knowledge and operating rules for the clipper clipping pipeline at /home/rbilal/clipper. Use for any clipper task, milestone M0-M10, spec/docs work, stage scripts, or debugging this project.
---

# Clipper project skill

## When to use
Any task involving:
- the clipper repo (`/home/rbilal/clipper`)
- milestones M0–M10 or their sub-stages (e.g. M5a/M5b/M5C-0…M5C-2)
- docs/spec files, contracts, prompts
- transcription, analysis, candidates, LLM scoring, selection, cuts, metadata/ClipPlans, face/layout, render, QC, posting
- debugging tests or `scripts/*_job.py` stage runners

## Project paths — do not invent paths
The project root is **`/home/rbilal/clipper`**. (It is NOT `/home/clipper` — that path does not exist.)

- Project root: `/home/rbilal/clipper`
- Docs root: `/home/rbilal/clipper/docs`
- Milestone/spec docs: `/home/rbilal/clipper/docs/spec/*.md` (milestones.md, m2.md, m3.md, m4.md, m5.md, m5c_status.md, m6.md, m7.md, m8.md, llm.md, contracts.md, scoring_prompt.md, pipeline.md)
- Source: `/home/rbilal/clipper/src/clipper` (contracts in `models/`, I/O in `adapters/`, stages in `stages/`)
- Stage runners: `/home/rbilal/clipper/scripts/`: `ingest_job.py`, `transcribe_job.py`, `analyze_job.py`, `candidates_job.py`, `score_job.py [--mock]`, `select_job.py [--no-llm|--mock]`, `layout_job.py [--clip --layout --score-threshold --cpu-faces]`, `render_job.py [--clip --cpu-render]`; MANUAL (GPU0 free + PI stopped): `bench_m6.py`, `bench_m7.py`, `faces_preflight.py`, `render_preflight.py`, `llm_preflight.py`; installers (manual, never in tests): `install_models.sh`, `install_fonts.sh`. `scripts/cuda_env.sh` must be sourced before `layout_job.py` and any GPU video run.
- Tests: `/home/rbilal/clipper/tests`
- DB: `data/clipper.db` (SQLite WAL); job artifacts under `data/jobs/<job_id>/`
- Venv: `/home/rbilal/clipper/.venv`
- Rules + progress (authoritative for "what is done now"): `/home/rbilal/clipper/AGENTS.md`

`config.py` uses a `CLIPPER_` env prefix and reads `.env`; `paths.data_dir` is a **CWD-relative** `./data`. Always `cd /home/rbilal/clipper` before running anything. Never use `~/clipper` (wrong).

## Startup ritual — every task
1. `cd /home/rbilal/clipper`
2. Read `AGENTS.md` — rules + current progress (authoritative for what is done now)
3. Read `docs/spec/milestones.md` for history; read the spec for the milestone in the task card (e.g. `docs/spec/m5.md`, `docs/spec/llm.md`)
4. Read only the files listed in the task card. Do not load the whole repo into context.
5. Activate the venv and verify:
   ```bash
   source .venv/bin/activate
   which python
   python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"
   ```
6. `git status --short`
7. Create the branch named in the task card (convention: `<milestone>-<slug>`, e.g. `m5c2-selection-stage-tests`)
8. If venv activation fails: write `BLOCKED.md` and stop. Never use system python.

## Milestone map
- M0–M4: scaffold/ingest/transcribe/analyze/candidates
- M5: LLM scoring — M5a runtime/client/contracts, M5b scoring stage, M5c selection+cuts+metadata
- M6: GPU face analysis, tracking, layout decision, crop path (fills `layout`/`crop_path` in draft ClipPlans). COMPLETE incl. M6f (`face_follow`) + M6g (`bridge_tracks`).
- M7: GPU render (NVDEC/NVENC), captions, hook overlay. COMPLETE (M7a–M7f) + M7g/M7h caption fixes. Residual: pasted bench_proof from `bench_m7.py`.
- M8: worker orchestration, one job at a time. **M8a DONE** (spec `docs/spec/m8.md`: 9 coarse stages in `worker_plan.STAGE_PLAN` — prep→transcribe→analyze→candidates→scoring→selection→clip_plans→layout→render; coarse skip only when all required fine stages are done + artifacts exist + `schema_version==1`; job states `queued→running→awaiting_qc` (≥1 clip) / `done` (zero clips, valid) / `failed`; retry ONLY via explicit `--retry` = `db.reset_failed_stages`; per-job `report.json` = `JobReport` schema_version=1 + `job_reports` table (migration v3); contracts in `models/worker_artifacts.py`; pure `worker_plan.py`; additive `db.py` helpers; M8 never touches Ollama 11434, LLM stages reuse `managed_ollama`). **M8b DONE** (`worker.py`: `StageRunner`/`FakeStageRunner`, `worker_lock` on `data/run/worker.lock`, `run_job` (skip/done/failed + `report.json` file + clean `WorkerError`), `run_next` once/loop; CLI `run JOB_ID [--retry]` / `run --next [--loop] [--poll S]` / `status JOB_ID` / `ls [--state]` / `report JOB_ID`; tests inject fakes into the `clipper.cli._stage_runners` hook). **M8c DONE** (`worker_stages.py`: one `CoarseRunner` per coarse stage wrapping EXACTLY ONE existing M1–M7 stage function with its adapters — no reimplementation; `STAGE_RUNNERS` keyed by the exact `STAGE_PLAN` names; `clipper.cli._stage_runners` now returns the REAL map; LLM routing per spec: `scoring` always under ONE `managed_ollama` (torn down on success AND failure), `clip_plans` only when ≥1 selected (zero selected → `client=None` + runtime never entered; managed `LLMError` → warn + `client=None`, job never fails), `selection`/`layout`/`render` NEVER enter it; `manage_runtime=false` untouched — main Ollama 11434 never modified; runners never self-lock/spawn/subprocess — the M8b loop is the only gate). **M8 COMPLETE (M8a+M8b+M8c) — `clipper run JOB_ID` drives the real M1–M7 pipeline.** No watch folder, no systemd until the card says so.
- M9: QC **dashboard** hosted on the Linux server (FastAPI/Jinja2/HTMX per config/docs) — the user operates the pipeline from a browser: toggles for face tracking, captions on/off, placement/margins, thresholds, explicit layout choice.
- M10: posting outbox

## Project direction & design paradigms (added 2026-11-25 — bind all future decisions to these)

The user reviewed real clips (job `7d187d98…`, TED-stage footage `test_media/ted/ted_x.mp4`) and set the direction:

1. **General content, not just one source.** Clips come from TED talks, shows/episodes, any horizontal-to-vertical reframing content. Face tracking must behave on low-confidence small faces too (e.g. YuNet top prob ~0.5 on stage footage never clears the old 0.6 bar).
2. **Face-stabilized zoom.** Even "zoom" (center_crop) clips should subtly follow the face so the person stays in frame; tracking is a continuous refinement, not either/or (full face_track vs static center_crop).
3. **Dashboard-first design.** M9 is a server-hosted dashboard controlling these knobs. Therefore EVERY user-facing behavior knob must be:
   - a **config default** (`cfg.`, read through `config.py`), and
   - overridable **per job / per clip** (precedent: `forced_layout` / `layout_source="manual"` in the M6e layout stage), and
   - plumbable as a pure function argument so the UI/stage API can pass overrides without new code paths.
   Do NOT hardcode behavior constants into stages/adapters. Tests stay CPU-only via fixtures.
4. **Spec-before-code.** Behavior/constant changes to completed milestones (M6d pan law, M7a caption format, thresholds) require a dated decision entry in the relevant `docs/spec/m#.md` (new decision number, old values recorded) BEFORE the code change; mirror contract changes in `docs/spec/contracts.md`; update `AGENTS.md` + `milestones.md`.
5. **Diagnosis evidence.** When a stage "fell back" or "found 0 faces", check the artifact metadata first (`layout.json`: `layout`, `face_track_count`, `layout_source`, `dominant_persistence`), then probe the raw detector scores on the actual frames (CPU ORT or `Cv2CpuFaceDetector` debug path) before touching thresholds. Note: `cv2.FaceDetectorYN` returns a `cv2.Mat`, NOT a `numpy.ndarray` — `isinstance(r, np.ndarray)` is the wrong emptiness check (returned -1 int = no faces).
6. **Karaoke (M7h lesson):** ASS format tags do NOT consume text — never duplicate the word around `\kf` (doubled subtitles, e.g. `Oh,Oh, there'sthere's`). `\c` color overrides do NOT recolor `\kf` karaoke parts (unsung falls back to the style SecondaryColour — that's why the old inline-`\c` form rendered RED, not yellow). Correct form: emit each word ONCE as `{\kf<cs>}<word>`; karaoke colors live in the Caption STYLE: `PrimaryColour=&HFFFFFF&` (sung white), `SecondaryColour=&H00FFFF&` (unsung yellow).
7. **Render assay rule (M7g/M7h pattern):** before/after ANY caption or ASS change, burn the ASS onto a solid `testsrc`/grey 1080x1920 frame with the real ffmpeg+libass and measure/eyeball (the mock-subprocess tests assert command strings, not pixels — both M7g and M7h bugs were invisible to them).
8. **Known-good facts (2026-11-25):** YuNet model `data/models/yunet.onnx` is byte-identical to official OpenCV Zoo (md5 `4ae92eeb150c82ce15ac80738b3b8167`). ffmpeg here is 4.4.2: `scale_cuda` has NO `format` AVOption — do not add `:format=nv12` back. ASS events MUST start with uppercase `Dialogue:` (libass silently drops `dialogue:` lines — this was the M7 caption bug).

## Session pattern for behavior changes (use for any new knob/fix)
1. Startup ritual (below) → `git status` → named branch.
2. If touching a completed milestone's contract/constants: spec decision entry FIRST, record old values (M6f/M6g/M7h all followed this).
3. Pure-logic changes (`layouts.py`, `captions.py`, `tracking.py`) stay I/O-free and fixture-tested; caption/ASS changes additionally get the render assay above.
4. New user-facing knobs: `config.py` default + `config.yaml` entry + (when per-clip relevant) the override parameter threaded through the stage's public function (existing pattern: `forced_layout`, `frame_reader`, `clip_id`).
5. Acceptance commands pass (venv, paste outputs), ruff clean, AGENTS.md + milestones.md updated, committed on the branch. STOP at the milestone boundary.

**Stale-map warning:** the old "Future Milestones" block in `AGENTS.md` (M6=quality gate, M7=cut refinement, M8=reframe, M9=captions, M10=QC+outbox) is a SUPERSEDED plan — selection/cut refinement was absorbed into M5c, and M9 is a DASHBOARD (see direction rules above), not "QC+outbox". `docs/spec/milestones.md` mixes old and new entries; trust the milestone specs + the task card over any conflicting list. `config.yaml` has `faces:`, `captions:`, `render:` blocks (1080x1920, crf 14, `h264_nvenc`).

**Status at 2026-11-25 (historical, superseded by the Status section below):** M0–M5c, M6 (a–e), M7a–M7f DONE (first real GPU render + bench proof exist). Open work agreed with the user: (a) caption `Dialogue:` case fix [M7 bug fix — DONE, M7g], (b) face-stabilized center crop (follow mode — DONE, M6f), (c) dashboard/config surface groundwork [now covered by M8a/M8b + M9].

## Status (updated after M7 verification + M7h/M6g fixes + M8a + M8b + M8c)
- **M0–M8 COMPLETE** (M7 incl. M6f `face_follow`, M6g track bridging, M7g `Dialogue:` case fix, M7h karaoke double-copy fix; M8a contracts/plan/DB helpers; M8b worker loop + CLI; M8c real M1–M7 stage wiring via `worker_stages.py`) — **646 CPU tests green, `pytest -m "not gpu"` → 646 passed + 1 GPU oracle deselected (suite wall 9.7s on this host since the 2026-11-27 M0 fix), ruff clean**. Full history: `AGENTS.md` tail.
- **M9 DESIGN DECISIONS RECORDED 2026-12-01** (authoritative: `docs/spec/m9.md` `## Decisions` — m9-1 … m9-11; contract mirror in `docs/spec/contracts.md`) — typed per-clip `caption_overrides`/`layout_overrides`, manual trims (hard 15–75s), fc-scan font family inventory, force-select without LLM, rerender dirty-set + `final.v<N>.mp4` history (retain 3), metadata regen via M5c path, stack **FastAPI+Jinja2+HTMX in `src/clipper/web/` (new deps approved: fastapi/uvicorn/jinja2 — not yet installed)**, 4-layer provenance resolver, two-tier preview, platform safe zones + burn-in modes. **SHIPPED through M9e (2026-12-02 — detail in AGENTS.md/specs):** M9a contracts/assets, M9b pure resolvers (`src/clipper/overrides.py`), M9c asset I/O + `job_overrides` persistence + `build_font_inventory` + installer wiring, M9d QC actions (`src/clipper/qc_actions.py` + `src/clipper/adapters/qc.py`; CLI `approve`/`reject`/`export JOB`/`force-select CAND [--reason]`, NO LLM/render, immutable artifacts, one worker lock per mutation), **M9e rerender + preview (`src/clipper/rerender.py`: `rerender_clip` dirty-set matrix (caption-only reuses mezzanine/audio; trim re-renders media; `--relayout` re-runs `run_layout(clip_id=...)`) + `final.v<N>.mp4` history (retain 3) + `final.mp4` copy + `plan_snapshot` before-state (m9-12) + `preview_frame` (m9-10 Tier 2 via additive `RenderAdapter.preview`); CLI `clipper rerender CLIP [--relayout] [--cpu-render]` / `clipper preview CLIP --at T`; m9-12 additive `RenderVersion.plan_snapshot: ClipPlan|None` (models + `model_rebuild` in `clipper.models`)). 935 passed (not gpu), ruff clean.** NEXT: M9 remainder — m9-6 per-clip metadata regen (batch=1, existing `managed_ollama` path, mocks in tests; never fails the job) + web per m9-7 (FastAPI+Jinja2+HTMX in `src/clipper/web/`, deps approved not installed). Then M10 (posting). M8c is committed on branch `m8c-real-wiring` (parents: M8b `757e3b6` + decisions doc `a8ea706`). Branch precedent: work commits land on long-lived feature tips then surface in `main` — when a card says "branch from main", branch from the actual latest tip and flag it.
- KNOWN pre-existing quirk (M0 code) — **FIXED 2026-11-27** (card `docs/spec/cards/m0-db-init-timeout.md`): fresh `Database(cfg)` init cost ~10s/build on this host (`init_db` held the WAL lock while `set_schema_version` committed from a second connection; both version writes hit the 5s sqlite busy timeout, swallowed by `except: pass`; version row never recorded). Migrations are now recorded on `init_db`'s own connection — fresh init ≈0.02s, version 3 recorded first pass. The `test_faces.py` `sys.modules` landmine (M5 test) was **FIXED the same day** (card `docs/spec/cards/m5-faces-import-hygiene-test.md` — see Environment gotchas).

## Layouts and tracking (M6f/M6g — do not regress)
- Layout precedence: `facecam > split_screen > face_track (≥0.6 persistence) > face_follow > center_crop` (`layouts.decide_layout`).
- `face_follow` (M6f): the face-stabilized zoom — same dynamic sendcmd+`crop` mezzanine path as `face_track` (`DYNAMIC_CROP_LAYOUTS` shares both), milder pan law `FOLLOW_EMA_TAU_S=0.5` / `FOLLOW_DEAD_ZONE=0.10` / `FOLLOW_MAX_PAN_FRAC_PER_S=0.40` (face_track keeps M6d: 0.8/0.20/0.30). Faceless `face_follow` → STATIC centered window, never raises. Knobs: `faces.face_follow_enabled` (default true), `faces.follow_persistence_min` (default 0.3) — both config + per-run, dashboard-ready.
- **Detection bar is global 0.6; per-run override is `scripts/layout_job.py --score-threshold`** (applied to `cfg.faces` before `get_face_detector`). The TED review footage needs `--score-threshold 0.5` (YuNet best score/frame ~0.50 there).
- Tracking flow: `associate(greedy-IoU, max_gap_frames=3)` → **`bridge_tracks` (M6g: merges segments with gap ≤ 1.0s and centroid dist ≤ 0.30 — spotty-but-real detections become one track, so `face_track` ≥ 0.6 fires where only `face_follow` did)** → `summarize` (fills `dominant_persistence`) → `build_tracks`.
- Diagnosing a bad crop/track: first read the per-clip `layout.json` (`layout`, `layout_source`, `face_track_count`, tracks), then probe raw detector scores on the actual `vision.mp4` frames, before touching any threshold.

## Pending manual GPU verifications (user or supervised session: GPU0 free + PI agent stopped)
1. `cd /home/rbilal/clipper && source .venv/bin/activate && source scripts/cuda_env.sh && python scripts/layout_job.py 7d187d989c2b453f838048b1aa2d1155 --score-threshold 0.5` → expect `layout: face_track` (bridged persistence ≥ 0.6) or `face_follow`, `face_track_count ≥ 1`
2. `python scripts/render_job.py 7d187d989c2b453f838048b1aa2d1155` → regenerates captions (old `data/jobs/…/clips/…/captions.ass` still render doubled/stale until re-rendered)
3. Bench proofs still owed: `python scripts/bench_m6.py <job_id>` (paste into AGENTS.md M6e entry) and `python scripts/bench_m7.py <job_id>` (paste into M7f entry — expected: `peak_gpu0_util > 0`, `nvenc_session_seen: true`, `encoder=h264_nvenc`)
- The user's review job is `7d187d989c2b453f838048b1aa2d1155` (TED-stage, `test_media/ted/ted_x.mp4`) — low-confidence faces (~0.5) are normal there.

## Hardware
- GPU0: RTX 2080 Super — media only (Whisper ASR, NVDEC/NVENC, M6/M7 video work)
- GPU1: Tesla P100-16GB — M5 runtime LLM (`gemma4:12b`)
- GPU2: Tesla P100-16GB — idle; may be used by the coding agent
- CPU: i5-9600KF 6c/6t, 32GB RAM
- P100s have no NVENC/NVDEC. Never assign video encode/decode to them.
- GPUs are often occupied by the user's Ollama (Qwen 3.8:27B). **The PI/coding agent must be stopped before real (non-mock) LLM or GPU-ASR runs** — it competes for VRAM. All in-skill work must be mock/CPU.

## Ollama topology — do not modify the user's service
- Coding agent LLM: Qwen 3.8:27B on the user's existing Ollama `127.0.0.1:11434`, multi-GPU. NEVER change that service, its env, or systemd config (`deploy/*.service` files are reference only — no systemd changes).
- M5 runtime: clipper spawns its own **ephemeral** `ollama serve` on `127.0.0.1:11435` with `CUDA_VISIBLE_DEVICES=1`, reusing the existing model store; `managed_ollama(cfg)` always tears it down (process group, SIGTERM→SIGKILL→port release) on success AND failure. Never restart per clip; `num_ctx` is per request.
- M5 model/options: `gemma4:12b`, `num_ctx=16384`, `num_batch=256`, `temperature=0.1`, `num_predict=2048` (M5a fix: was 1024 — gemma4:12b is a thinking model; client also sends `think=false`, else hidden reasoning eats the whole budget and content comes back empty), `timeout_s=300`, `batch_size=5`, `max_batch_tokens=6000`, context ±10s.
- Tests use MockLLM / monkeypatched factories (`get_llm_client`, `get_adapter`, `get_scene_adapter`) — no network, no GPU, no nvidia-smi.

## Environment gotchas (learned the hard way — do not regress)
- ASR currently defaults to `device="cpu"`, `compute_type="int8"` (faster-whisper/ctranslate2 4.9; `cpu_threads` must be passed, `device_index` must be a real int on CPU). GPU0 cuda:0/float16 only when GPU0 is free.
- scenedetect must be installed `pip install scenedetect --no-deps`; cv2 comes from `opencv-python-headless`. **`opencv-python` (non-headless) is FORBIDDEN** (libGL.so.1 failure). `pip check` false-positive for this pair is expected.
- `adapters/scenes.py` does lazy `scenedetect`/`cv2` imports inside `detect_shots()` so imports stay light.
- `data.json`/artifacts are written in canonical form `model_dump_json(indent=2)` + newline; golden fixtures (`tests/fixtures/m4/candidates.json`) must stay byte-identical.
- **Always `source scripts/cuda_env.sh` before running `scripts/layout_job.py`** (the GPU/onnxruntime-gpu M6 layout job). The system has no CUDA toolkit/cuDNN (no sudo); `cuda_env.sh` points `LD_LIBRARY_PATH` at the user-local cuBLAS 13 + cuDNN 9 pip libs and pins `CUDA_VISIBLE_DEVICES` to GPU0 (2080 Super, cc 7.5 — the only GPU onnxruntime 1.32's CUDA EP supports). Without it the run fails to find cuBLAS/cuDNN. Correct form: `source scripts/cuda_env.sh && python scripts/layout_job.py <job_id>`.

- Full suite is ~10s wall on this host since the 2026-11-27 M0 db-init fix (646 tests; before the fix it was ~26 min because every fresh `Database` init paid two 5s sqlite busy timeouts). If a run suddenly takes minutes, something re-introduced slow fixture I/O.
- **NEVER pop a parent package from `sys.modules` in tests** (2026-11-27 rule): the old `test_faces.py` hygiene test popped `clipper.adapters` and re-imported it mid-suite; the fresh package object lost its `.media`/`.llm`/… attribute bindings, so every later string-targeted `monkeypatch.setattr("clipper.adapters.<sub>.<attr>", …)` failed (16 failures on the first M8c full run). Rule: tests may pop/restore only the LEAF modules they assert on; prefer patching the exact module objects a module bound at import (e.g. `worker_stages.media_adapters.get_adapter`). Full details: `docs/spec/cards/m5-faces-import-hygiene-test.md`.

## Acceptance commands — always in venv
```bash
cd /home/rbilal/clipper
source .venv/bin/activate
which python
python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"
python -m pytest -m "not gpu" -v
python -m ruff check .
```
Paste the outputs as proof. Do not claim done without them.

## Architecture invariants — non-negotiable
- All data contracts live ONLY in `src/clipper/models/`, all with `schema_version=1`; re-exported from `clipper.models`.
- Every stage: typed input → JSON artifact in `data/jobs/<job_id>/` → typed output. Skip only if DB stage `done` AND artifact exists AND `schema_version == 1`. Job goes `queued→running→queued` on success; failure marks stage+job `failed` and raises clean `StageError` (naming the missing artifact AND the milestone/script that produces it).
- All GPU/ffmpeg/network/model I/O goes through `adapters/`. Pure logic in top-level modules (`candidates.py`, `scoring.py`, `selection.py`, `cuts.py`, `sentences.py`, `audio_analysis.py`) — no I/O, unit-testable without GPU/LLM.
- Every adapter has a CPU-only Fixture/Mock. Tests never require GPU, CUDA, live Ollama, model downloads, or root.
- SQLite WAL, stdlib sqlite3, no ORM. One job at a time in v1.
- `Zero clips/candidates is a valid success` at every stage.
- No new dependencies without explicit approval. No sudo, no systemd changes, no model downloads in tests.
- Do not delete user work or revert commits without asking.

## M5 rules (do not drift)
- LLM proposes, code decides. Axes 0–10: hook, payoff, self_contained, emotion, clarity.
- `weighted = 0.30*hook + 0.25*payoff + 0.20*self_contained + 0.15*emotion + 0.10*clarity` (single source: `scoring.weighted_score`)
- Gate (from `cfg.clips.gate.*`): weighted ≥ 7.2, hook ≥ 6, self_contained ≥ 7.
- Target 3, max 5 (`cfg.clips.target_count`/`max_count`). Dedupe temporal IoU > 0.30; backfill from next eligible.
- Cut order (module constants in `cuts.py`): clamp suggested ±8s (`SUGGEST_SLACK`) → sentence snap (1.0) → word padding (−0.120/+0.250) → silence snap (≤0.300) → shot snap (≤0.400) → validate 15–75s LAST; `CutError` carries a specific reason.
- Selection and cuts are pure Python and NEVER call the LLM.
- Metadata is one optional LLM call per batch of selected clips (selected-clip transcripts only, never the full video). Retry then deterministic fallback + warning — metadata failure must NEVER fail the job.
- Batch failure after `cfg.llm.max_retries` rejects those candidates (`reject_reason`), job continues.
- M5 writes DRAFT ClipPlans: `layout=center_crop`, `crop_path=[]`, `status=pending_qc`; `clip_id = f"{job_id}-clip{n:02d}"`. M6 fills layout/crop_path.
- Scoring prompt is `prompts.py::SCORING_SYSTEM` (canonical) mirrored verbatim in `docs/spec/scoring_prompt.md` — change both together only per spec decision.
- `scripts/score_job.py --mock` / `select_job.py --mock` are deterministic wiring tests only — never a real judgement.

## M6/M7 rules (complete — changes need spec decisions; see Session pattern above)
- **Running `scripts/layout_job.py` requires `source scripts/cuda_env.sh` first** (see Environment gotchas). Do not run the layout job for a real (non-mock) inference without it.
- Production path uses GPU0 (2080 Super) for all video decode/encode; NVDEC decode, NVENC encode (mezzanine + final), GPU crop/scale/composite where supported. No libx264 production path; CPU fallback only behind an explicit debug flag.
- Face detection via onnxruntime-gpu (YuNet/SCRFD-class) at 30fps minimum (native if lower). 3fps crop paths are forbidden. Use a 30fps vision proxy or source, not the M1 15fps proxy. `crop_path` at native video fps.
- Must include benchmark proof: decode fps, inference fps, render fps, GPU utilization, NVENC session in nvidia-smi, render time per clip.

## No-drift rules
- Follow the task card exactly; do not implement the next milestone; do not change constants, contracts, prompts, or config defaults of complete milestones.
- If a spec is ambiguous: choose the simplest boring option, document it in the milestone spec, note it in the report/AGENTS.md. (Examples of open judgment notes live in `docs/spec/m5c_status.md` — do not "fix" them without a spec decision.)
- If blocked after 3 attempts: write `BLOCKED.md` (what you tried, evidence) and stop.
- If a harness/PI generation timeout threatens: split the task into smaller chunks (the M5c pattern: audit → selection pure → cuts pure → metadata → stages → tests) instead of retrying one giant prompt.

## Task card template (expect incoming tasks in this shape)
```
Task: <milestone> — <scope>.
Continue "clipper". <previous milestone> is complete.
Read first: AGENTS.md, docs/spec/<relevant>.md, <listed files>.
Create branch <branch>.
Do NOT implement <next milestone>.
VENV — MANDATORY:
  source .venv/bin/activate
  which python
  python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"
Dependencies: <allowed / none>.
BUILD: <numbered steps>.
TESTS: <CPU-only list>.
ACCEPTANCE:
  source .venv/bin/activate
  python -m pytest -m "not gpu" -v
  python -m ruff check .
Update AGENTS.md progress. Paste outputs. STOP after <milestone>.
```

## Definition of done
- Acceptance commands pass in the venv (test count + ruff pasted as proof)
- New/changed constants recorded in the milestone spec; contracts mirrored in `docs/spec/contracts.md` where applicable
- `AGENTS.md` progress updated + `docs/spec/milestones.md` status line updated
- **This skill (`SKILL.md`) updated if anything long-lived changed** (status, scripts, layouts, known-good facts) — and note: if `AGENTS.md`'s tail is dated later than this skill, `AGENTS.md` + the specs win over this skill; fix this skill during the session
- Scope respected (no next-milestone work); work committed on the task's branch
- Zero GPU/network requirements in any new test
