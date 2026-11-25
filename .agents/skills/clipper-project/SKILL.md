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
- Milestone/spec docs: `/home/rbilal/clipper/docs/spec/*.md` (milestones.md, m2.md, m3.md, m4.md, m5.md, llm.md, contracts.md, scoring_prompt.md, m5c_status.md)
- Source: `/home/rbilal/clipper/src/clipper` (contracts in `models/`, I/O in `adapters/`, stages in `stages/`)
- Stage runners: `/home/rbilal/clipper/scripts/` (analyze_job.py, candidates_job.py, transcribe_job.py, score_job.py, select_job.py, layout_job.py, llm_preflight.py; `scripts/cuda_env.sh` must be sourced before `layout_job.py`)
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
- M6: GPU face analysis, layout decision, crop path (fills `layout`/`crop_path` in draft ClipPlans)
- M7: GPU render (NVDEC/NVENC), captions, hook overlay
- M8: worker orchestration (one job at a time in v1)
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
5. **Diagnosis evidence.** When a stage "fell back" or "found 0 faces", check the artifact metadata first (`layout.json`: `layout`, `face_track_count`, `layout_source`), then probe the raw detector scores on the actual frames (CPU ORT or `Cv2CpuFaceDetector` debug path) before touching thresholds. Note: `cv2.FaceDetectorYN` returns a `cv2.Mat`, NOT a `numpy.ndarray` — `isinstance(r, np.ndarray)` is the wrong emptiness check (returned -1 int = no faces).
6. **Known-good facts (2026-11-25):** YuNet model `data/models/yunet.onnx` is byte-identical to official OpenCV Zoo (md5 `4ae92eeb150c82ce15ac80738b3b8167`). ffmpeg here is 4.4.2: `scale_cuda` has NO `format` AVOption — do not add `:format=nv12` back. ASS events MUST start with uppercase `Dialogue:` (libass silently drops `dialogue:` lines — this was the M7 caption bug).

## Session pattern for behavior changes (use for any new knob/fix)
1. Startup ritual (below) → `git status` → named branch.
2. If touching a completed milestone's contract/constants: spec decision entry FIRST, record old values.
3. Pure-logic changes (`layouts.py`, `captions.py`, `tracking.py`) stay I/O-free and fixture-tested.
4. New user-facing knobs: `config.py` default + `config.yaml` entry + (when per-clip relevant) the override parameter threaded through the stage's public function (existing pattern: `forced_layout`, `frame_reader`, `clip_id`).
5. Acceptance commands pass (venv, paste outputs), ruff clean, AGENTS.md + milestones.md updated, committed on the branch. STOP at the milestone boundary.

**Stale-map warning:** the old "Future Milestones" block in `AGENTS.md` (M6=quality gate, M7=cut refinement, M8=reframe, M9=captions, M10=QC+outbox) is a SUPERSEDED plan — selection/cut refinement was absorbed into M5c, and M9 is a DASHBOARD (see direction rules above), not "QC+outbox". `docs/spec/milestones.md` mixes old and new entries; trust the milestone specs + the task card over any conflicting list. `config.yaml` has `faces:`, `captions:`, `render:` blocks (1080x1920, crf 14, `h264_nvenc`).

**Status at 2026-11-25:** M0–M5c, M6 (a–e), M7a–M7f DONE (first real GPU render + bench proof exist). Open work agreed with the user: (a) caption `Dialogue:` case fix [M7 bug fix], (b) face-stabilized center crop (follow mode — M6 spec delta), (c) dashboard/config surface groundwork (per-clip `options.json` + CLI flags).

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
- M5 model/options: `gemma4:12b`, `num_ctx=16384`, `num_batch=256`, `temperature=0.1`, `num_predict=1024`, `timeout_s=300`, `batch_size=5`, `max_batch_tokens=6000`, context ±10s.
- Tests use MockLLM / monkeypatched factories (`get_llm_client`, `get_adapter`, `get_scene_adapter`) — no network, no GPU, no nvidia-smi.

## Environment gotchas (learned the hard way — do not regress)
- ASR currently defaults to `device="cpu"`, `compute_type="int8"` (faster-whisper/ctranslate2 4.9; `cpu_threads` must be passed, `device_index` must be a real int on CPU). GPU0 cuda:0/float16 only when GPU0 is free.
- scenedetect must be installed `pip install scenedetect --no-deps`; cv2 comes from `opencv-python-headless`. **`opencv-python` (non-headless) is FORBIDDEN** (libGL.so.1 failure). `pip check` false-positive for this pair is expected.
- `adapters/scenes.py` does lazy `scenedetect`/`cv2` imports inside `detect_shots()` so imports stay light.
- `data.json`/artifacts are written in canonical form `model_dump_json(indent=2)` + newline; golden fixtures (`tests/fixtures/m4/candidates.json`) must stay byte-identical.
- **Always `source scripts/cuda_env.sh` before running `scripts/layout_job.py`** (the GPU/onnxruntime-gpu M6 layout job). The system has no CUDA toolkit/cuDNN (no sudo); `cuda_env.sh` points `LD_LIBRARY_PATH` at the user-local cuBLAS 13 + cuDNN 9 pip libs and pins `CUDA_VISIBLE_DEVICES` to GPU0 (2080 Super, cc 7.5 — the only GPU onnxruntime 1.32's CUDA EP supports). Without it the run fails to find cuBLAS/cuDNN. Correct form: `source scripts/cuda_env.sh && python scripts/layout_job.py <job_id>`.

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
- Scope respected (no next-milestone work); work committed on the task's branch
- Zero GPU/network requirements in any new test
