"""M8c — the REAL M1–M7 stage runners for the M8b worker loop.

See `docs/spec/m8.md` (M8c section) and `AGENTS.md`. Each runner in
`STAGE_RUNNERS` wraps exactly ONE EXISTING M1–M7 stage function with
the adapters that function already accepts — no stage logic is
reimplemented, and no M0–M7 logic/constant/contract/config default
changes (M8 decision 1).

Verified mapping (repo-confirmed signatures, M8c):

- `prep`        -> `stages.ingest.run_ingest` (M1; hash-idempotent)
- `transcribe`  -> `stages.transcribe.run_transcribe` (M2/M2b/M2c)
- `analyze`     -> `stages.analyze.run_analysis` (M3a/M3b)
- `candidates`  -> `stages.candidates.run_candidates` (M4a/M4b)
- `scoring`     -> `stages.score.run_scoring` (M5b, ONE managed Ollama)
- `selection`   -> `stages.select.run_selection_stage` (M5C-4, NO LLM)
- `clip_plans`  -> `stages.select.run_clip_plans_stage` (M5C-5)
- `layout`      -> `stages.layout.run_layout` (M6e)
- `render`      -> `stages.render.run_render` (M7d)

M8 rules honored here (docs/spec/m8.md):
- LLM stages (scoring; clip_plans only when >= 1 clip is selected)
  use the existing ephemeral `llm_runtime.managed_ollama` runtime,
  exactly as `scripts/score_job.py` / `scripts/select_job.py` already do.
  The child is ALWAYS torn down on success AND failure. With
  `manage_runtime=False` `managed_ollama` yields the existing
  base_url unchanged (no spawn) — the user's main Ollama
  (127.0.0.1:11434) is never touched either way (M8 decision 9).
- The selection stage NEVER enters `managed_ollama` (M5 rule: selection
  is pure Python and never calls the LLM).
- GPU work lives inside the injected adapters (GPU0 per SKILL.md); the
  process inherits its environment from the launching shell and these
  runners never mutate the process environment (e.g. fake CUDA setup).

Deliberately NOT done here (M8c task card):
- no watch folder, no systemd, no M9 dashboard, no M10 posting,
  no concurrency changes (the M8b worker lock in `clipper.worker`
  remains the ONLY gate — a runner neither acquires that lock nor
  starts another job);
- no new dependencies; tests are CPU-only (adapters, managed Ollama
  and stage functions are injected/monkeypatched).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable

from . import db
from .adapters import asr as asr_adapters
from .adapters import faces as face_adapters
from .adapters import llm as llm_adapters
from .adapters import media as media_adapters
from .adapters import render as render_adapters
from .adapters import scenes as scene_adapters
from .adapters import vision as vision_adapters
from .config import Config
from .llm_runtime import managed_ollama
from .stages import (
    StageError,
    analyze,
    candidates,
    ingest,
    layout,
    render,
    score,
    select,
    transcribe,
)
from .worker_plan import STAGE_PLAN

__all__ = [
    "CoarseRunner",
    "STAGE_RUNNERS",
]

RunnerFunc = Callable[[str, Config, sqlite3.Connection], None]
log = logging.getLogger(__name__)


class CoarseRunner:
    """One coarse stage from `STAGE_PLAN` bound to its wrapper function.

    `clipper.worker.run_job` calls `runner.run(job_id, cfg, conn)` for
    each coarse stage in plan order (the M8b `StageRunner` protocol).
    The runner owns only thin wiring: the existing stage function does
    the fine-grained stage bookkeeping itself.
    """

    def __init__(self, name: str, func: RunnerFunc) -> None:
        self.name = name
        self._func = func

    def run(self, job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
        """Execute the coarse stage for one job (raises on failure)."""
        self._func(job_id, cfg, conn)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CoarseRunner({self.name!r})"


def _require_job(conn: sqlite3.Connection, job_id: str) -> None:
    """Clean single-line error when `job_id` is unknown (runner guard)."""
    if db.get_job(conn, job_id) is None:
        raise StageError(
            f"job {job_id!r} not found — cannot run the M1-M7 pipeline "
            "for an unknown job"
        )


def _selected_clip_count(cfg: Config, job_id: str) -> int:
    """Count `selected == true` decisions in `selection.json` (M5C-4).

    Read-only: the `clip_plans` stage itself re-validates the artifact
    (schema_version, stage order). Zero selected is a valid pipeline
    state at this point.
    """
    artifact_path = cfg.resolved_data_dir / "jobs" / job_id / "selection.json"
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageError(
            f"clip_plans runner: cannot read {artifact_path} — the "
            f"selection stage must run first ({exc})"
        ) from exc
    decisions = data.get("decisions") or []
    return sum(1 for d in decisions if d.get("selected"))


# ---------------------------------------------------------------------------
# One wrapper per coarse stage (in STAGE_PLAN order), real wiring only
# ---------------------------------------------------------------------------


def _prep(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`prep` = M1 ingest (probe + proxy + audio inside `run_ingest`).

    `run_ingest` is idempotent by source content hash and returns
    `(job_id, is_new_job, should_retry)`: the SAME job on reuse (same
    hash -> same job). If it ever returned a DIFFERENT job id we refuse
    to adopt it: clean error, the worker marks the job failed.
    """
    job = db.get_job(conn, job_id)
    if job is None:
        raise StageError(
            f"job {job_id!r} not found — cannot run the M1-M7 pipeline "
            "for an unknown job"
        )
    returned_job_id, _is_new, _should_retry = ingest.run_ingest(
        source=job["source_path"],
        client=job["client"],
        niche=job["niche"],
        cfg=cfg,
        conn=conn,
        media=media_adapters.get_adapter(),
    )
    if returned_job_id != job_id:
        raise StageError(
            f"prep runner: run_ingest returned job {returned_job_id!r} "
            f"for job {job_id!r} — refusing to adopt a different job"
        )


def _transcribe(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`transcribe` = M2/M2b/M2c. The ASR adapter is injected per the
    M2b decision (never constructed inside the stage). Note: the real
    `run_transcribe(job_id, cfg, asr)` signature takes no `conn`."""
    _require_job(conn, job_id)
    transcribe.run_transcribe(job_id, cfg, asr=asr_adapters.get_adapter(cfg))


def _analyze(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`analyze` = M3a/M3b shots/silence/energy (media + scenes adapters
    injected as the stage takes them)."""
    _require_job(conn, job_id)
    analyze.run_analysis(
        job_id,
        cfg,
        conn,
        media_adapters.get_adapter(),
        scene_adapters.get_scene_adapter(),
    )


def _candidates(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`candidates` = M4a/M4b deterministic candidate generation (no
    adapters)."""
    _require_job(conn, job_id)
    candidates.run_candidates(job_id, cfg, conn)


def _scoring(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`scoring` = M5b LLM batch scoring under ONE managed ephemeral
    Ollama (torn down on success AND failure); the client is built
    from the yielded base_url — same shape as
    `scripts/score_job.py`."""
    _require_job(conn, job_id)
    with managed_ollama(cfg) as base_url:
        score.run_scoring(
            job_id, cfg, conn, llm_adapters.get_llm_client(cfg, base_url)
        )


def _selection(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`selection` = M5C-4 selection stage — NO LLM (pure,
    deterministic). Must NEVER enter `managed_ollama` (M5 rules)."""
    _require_job(conn, job_id)
    select.run_selection_stage(job_id, cfg, conn)


def _clip_plans(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`clip_plans` = M5C-5 metadata + `clips` rows.

    - zero selected clips: `client=None` (deterministic fallback
      metadata) and `managed_ollama` is NEVER entered;
    - >= 1 selected: ONE managed ephemeral Ollama for the single
      metadata call, ALWAYS torn down. A managed-Ollama startup/
      client failure (`LLMError`) must NOT fail the job (M5C-5 rule,
      same as `scripts/select_job.py`): warn and re-run the stage with
      `client=None` (deterministic fallback metadata). Metadata-call
      failures INSIDE the stage already fall back on their own
      (M5C-5 — untouched here).
    """
    _require_job(conn, job_id)
    if _selected_clip_count(cfg, job_id) == 0:
        select.run_clip_plans_stage(job_id, cfg, conn, None)
        return
    try:
        with managed_ollama(cfg) as base_url:
            select.run_clip_plans_stage(
                job_id, cfg, conn, llm_adapters.get_llm_client(cfg, base_url)
            )
    except llm_adapters.LLMError as exc:
        log.warning(
            "managed Ollama unavailable (%s) — running clip_plans with "
            "client=None (deterministic fallback metadata); the job "
            "does NOT fail",
            exc,
        )
        select.run_clip_plans_stage(job_id, cfg, conn, None)


def _layout(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`layout` = M6e per-clip face tracking + layout decision. The
    vision adapter and face detector are injected (GPU0 per SKILL.md;
    the stage closes the detector in its own `finally`). NO LLM."""
    _require_job(conn, job_id)
    layout.run_layout(
        job_id,
        cfg,
        conn,
        vision_adapters.get_vision_adapter(cfg),
        face_adapters.get_face_detector(cfg),
    )


def _render(job_id: str, cfg: Config, conn: sqlite3.Connection) -> None:
    """`render` = M7d per-clip NVDEC -> NVENC render. The RenderAdapter
    is injected (GPU0 production path; `cpu_render` stays the
    stage-level explicit debug flag). NO LLM."""
    _require_job(conn, job_id)
    render.run_render(
        job_id, cfg, conn, render_adapters.get_render_adapter(cfg)
    )


# One runner per coarse stage, keyed by the EXACT `STAGE_PLAN` names
# and in plan order (validated at M8b by `worker._runners_for`).
_RUNNER_FUNCS: dict[str, RunnerFunc] = {
    "prep": _prep,
    "transcribe": _transcribe,
    "analyze": _analyze,
    "candidates": _candidates,
    "scoring": _scoring,
    "selection": _selection,
    "clip_plans": _clip_plans,
    "layout": _layout,
    "render": _render,
}

STAGE_RUNNERS: dict[str, CoarseRunner] = {
    spec.name: CoarseRunner(spec.name, _RUNNER_FUNCS[spec.name])
    for spec in STAGE_PLAN
}
