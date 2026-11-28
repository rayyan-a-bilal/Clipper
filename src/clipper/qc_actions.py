"""M9d — QC selection actions and export (docs/spec/m9.md m9-4, m9-5).

Service layer for the CLI `approve` / `reject` / `export` / `force-select`
commands. Rules (m9-4 / m9-5, enforced here — see the M9d judgment notes in
`docs/spec/m9.md`):

- **One worker lock.** Every mutation holds the EXISTING M8 worker lock
  (`worker.worker_lock`) for its full duration. Never acquires a second
  lock, never calls `worker.run_job` — the M8b loop stays the only runner.
  Contended lock → the clean `WorkerError` refusal (m9-5 behaviour).
- **No LLM anywhere.** Force-select metadata is the existing deterministic
  `metadata.fallback_metadata` (m9-4). No Ollama, no client, no retry.
- **No render.** Export copies EXISTING `final.mp4` / `thumbnail.jpg`; a
  missing `final.mp4` is a concrete error naming the clip — no rerender.
- **Immutable artifacts.** `selection.json`, `scores.json`, `candidates.json`
  and `words.json` are never written (m9-5). The only artifact write is the
  `clip_plans.json` append/status update.
- **All file I/O through `adapters/qc.py`.** Pure logic reuses the existing
  surfaces: `overrides.effective_cut` (m9-2 validation), `metadata.*`,
  `cuts` constants, `clipper.models`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import clipper.adapters.qc as qc
from clipper import worker
from clipper.config import Config
from clipper.cuts import PAD_END, PAD_START, CutError
from clipper.db import get_job
from clipper.metadata import fallback_metadata
from clipper.models import Candidate, ClipPlan, Layout, SentencesArtifact
from clipper.overrides import effective_cut

__all__ = [
    "QCActionsError",
    "approve",
    "force_select",
    "reject",
    "export_job",
]


class QCActionsError(Exception):
    """QC action failed (unknown candidate/clip/job, invalid cut, missing
    render output). `str(error)` always carries the concrete detail."""


# ---------------------------------------------------------------------------
# Small shared helpers (no I/O of their own — file access stays in the
# adapter; table access goes through the caller's connection only)
# ---------------------------------------------------------------------------

def _job_dir(cfg: Config, job_id: str) -> Path:
    return cfg.resolved_data_dir / "jobs" / job_id


def _candidate_from_conn(conn, cand_id: str) -> tuple[str, Candidate]:
    """Resolve `(job_id, Candidate)` from the `candidates` table.

    The job_id is READ FROM THE TABLE (card rule — never string-parsed out
    of `cand_id`)."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT job_id, candidate_json FROM candidates WHERE cand_id = ?",
        (cand_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise QCActionsError(
            f"unknown candidate {cand_id!r}: no row in the candidates table "
            "(M4b candidates stage, scripts/candidates_job.py)"
        )
    try:
        candidate = Candidate.model_validate_json(row[1])
    except Exception as exc:
        raise QCActionsError(f"invalid candidate row for {cand_id!r}: {exc}") from exc
    return row[0], candidate


def _clip_from_conn(conn, clip_id: str) -> tuple[str, ClipPlan]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT job_id, plan_json FROM clips WHERE clip_id = ?", (clip_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise QCActionsError(
            f"unknown clip {clip_id!r}: no row in the clips table (a "
            "completed clip_plans stage or `clipper force-select` creates one)"
        )
    try:
        plan = ClipPlan.model_validate_json(row[1])
    except Exception as exc:
        raise QCActionsError(f"invalid clip row for {clip_id!r}: {exc}") from exc
    return row[0], plan


def _clip_window_transcript(
    sentences: SentencesArtifact, start: float, end: float
) -> str:
    """Transcript of the CLIP WINDOW only (sentences overlapping `[start,
    end]`) — same rule as the M5C-5 `_clip_transcript`, never the full
    video transcript."""
    in_window = [s for s in sentences.sentences if s.start < end and s.end > start]
    return " ".join(s.text for s in in_window)


def _next_clip_id(conn, job_id: str, existing: list[ClipPlan]) -> str:
    """Next free `f"{job_id}-clipNN"` (card format; NN continues from the
    lowest free two-digit slot — same shape as M5C-5's `-clip{n:02d}`)."""
    pattern = re.compile(rf"^{re.escape(job_id)}-clip(\d{{2}})$")
    used: set[int] = set()
    for plan in existing:
        match = pattern.fullmatch(plan.clip_id)
        if match:
            used.add(int(match.group(1)))
    cursor = conn.cursor()
    cursor.execute("SELECT clip_id FROM clips WHERE job_id = ?", (job_id,))
    for (clip_id,) in cursor.fetchall():
        match = pattern.fullmatch(clip_id)
        if match:
            used.add(int(match.group(1)))
    slot = 1
    while slot in used:
        slot += 1
    return f"{job_id}-clip{slot:02d}"


def _upsert_clip_row(conn, job_id: str, plan: ClipPlan) -> None:
    now = datetime.now().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO clips (clip_id, job_id, plan_json, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(clip_id) DO UPDATE SET
            job_id = excluded.job_id,
            plan_json = excluded.plan_json,
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        (plan.clip_id, job_id, plan.model_dump_json(), plan.status, now, now),
    )
    conn.commit()


def _set_clip_status(clip_id: str, new_status: str, cfg: Config, conn) -> ClipPlan:
    """approve/reject shared body — ONLY the status changes (m9-5: "set
    ClipPlan.status + clips.status only"), on BOTH the plan (persisted in
    clip_plans.json) and the clips row. Other clips are untouched."""
    with worker.worker_lock(cfg):
        job_id, plan = _clip_from_conn(conn, clip_id)
        updated = plan.model_copy(update={"status": new_status})
        job_dir = _job_dir(cfg, job_id)
        artifact = qc.read_clip_plans(job_dir)
        for index, existing in enumerate(artifact.plans):
            if existing.clip_id == clip_id:
                artifact.plans[index] = updated
                break
        else:
            raise QCActionsError(
                f"clip {clip_id!r} is not in {_job_dir(cfg, job_id)}/"
                f"{qc.CLIP_PLANS_FILENAME} — cannot {new_status} it"
            )
        qc.write_clip_plans(job_dir, artifact)
        _upsert_clip_row(conn, job_id, updated)
    return updated


# ---------------------------------------------------------------------------
# m9-4 — force select (NO LLM, NO gate editing)
# ---------------------------------------------------------------------------

def force_select(
    cand_id: str, reason: str | None, cfg: Config, conn
) -> ClipPlan:
    """Force-select a candidate as a new clip WITHOUT any LLM call (m9-4).

    - job_id resolved from the `candidates` table (never string-parsed)
    - cut = candidate bounds + the EXISTING padding constants
      (`cuts.PAD_START`/`cuts.PAD_END`, outward, start clamped at 0),
      validated on the existing cuts surface via
      `overrides.effective_cut` (start < end; 15.0s–75.0s)
      — `too short`/`too long` raise the concrete `CutError`; NO clamping
      (M9b-J1, M9d-J1)
    - metadata = the existing deterministic `metadata.fallback_metadata`
    - plan: `forced=True`, `force_reason=<reason>`,
      `status="pending_qc"`, `layout=center_crop`, `crop_path=[]`
      (layout/render belong to M9e)
    - `clip_id` = next free `f"{job_id}-clipNN"`; appended to
      `clip_plans.json` + upserted into the `clips` row
    - `selection.json` / `scores.json` / `candidates.json` / `words.json`
      are NEVER written (immutable, m9-5)
    """
    with worker.worker_lock(cfg):
        job_id, candidate = _candidate_from_conn(conn, cand_id)
        job = get_job(conn, job_id)
        if job is None:
            raise QCActionsError(
                f"unknown job {job_id!r} for candidate {cand_id!r}"
            )
        job_dir = _job_dir(cfg, job_id)

        # Cut: candidate bounds + existing padding constants (outward;
        # start never negative). 15–75s is enforced, concrete error.
        start = max(0.0, candidate.start - PAD_START)
        end = candidate.end + PAD_END

        # Metadata FIRST (deterministic fallback — the only metadata path)
        try:
            sentences = qc.read_sentences(job_dir)
        except ValueError as exc:
            raise QCActionsError(str(exc)) from exc
        transcript = _clip_window_transcript(sentences, start, end)
        meta = fallback_metadata(candidate.cand_id, transcript, job.get("niche") or "")

        artifact = qc.read_clip_plans(job_dir)
        clip_id = _next_clip_id(conn, job_id, artifact.plans)
        plan = ClipPlan(
            clip_id=clip_id,
            job_id=job_id,
            start=start,
            end=end,
            layout=Layout.center_crop,
            crop_path=[],
            weighted_score=candidate.cheap_score,
            title=meta.title,
            hook_overlay=meta.hook_overlay,
            description=meta.description,
            hashtags=meta.hashtags,
            status="pending_qc",
            forced=True,
            force_reason=reason,
        )
        try:
            effective_cut(plan)  # start < end + 15–75s, concrete CutError
        except CutError as exc:
            raise QCActionsError(f"force-select {cand_id}: {exc}") from exc

        # Persist AFTER validation — a rejected cut writes NOTHING.
        artifact = artifact.model_copy(update={"plans": [*artifact.plans, plan]})
        qc.write_clip_plans(job_dir, artifact)
        _upsert_clip_row(conn, job_id, plan)
    return plan


# ---------------------------------------------------------------------------
# m9-5 (approve/reject half) — status only
# ---------------------------------------------------------------------------

def approve(clip_id: str, cfg: Config, conn) -> ClipPlan:
    """Approve one clip: `ClipPlan.status` + `clips.status` → `approved`
    and nothing else changes."""
    return _set_clip_status(clip_id, "approved", cfg, conn)


def reject(clip_id: str, cfg: Config, conn) -> ClipPlan:
    """Reject one clip: `ClipPlan.status` + `clips.status` → `rejected`
    and nothing else changes."""
    return _set_clip_status(clip_id, "rejected", cfg, conn)


# ---------------------------------------------------------------------------
# m9-5 (export half) — approved clips only, copy-only, NO rerender
# ---------------------------------------------------------------------------

def export_job(job_id: str, cfg: Config, conn) -> list[Path]:
    """Copy the approved clips' render outputs into
    `data/exports/<job_id>/<clip_id>/`:

    - `final.mp4`   — copied from `data/jobs/<job_id>/clips/<clip_id>/`
    - `thumbnail.jpg` — copied alongside
    - `metadata.json` — the plan's metadata (title/hook/description/
      hashtags + identity), freshly written from the approved plan

    Rules: APPROVED clips only (zero approved → empty list, valid); a
    missing `final.mp4` (or `thumbnail.jpg`) is a concrete error naming
    the clip — export is read-only, it NEVER re-renders (m9-5); the
    source files are byte-identical after the copy."""
    with worker.worker_lock(cfg):
        if get_job(conn, job_id) is None:
            raise QCActionsError(f"unknown job {job_id!r}")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT clip_id, plan_json FROM clips WHERE job_id = ? "
            "AND status = 'approved' ORDER BY clip_id",
            (job_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        job_dir = _job_dir(cfg, job_id)
        out_dir = cfg.resolved_data_dir / "exports" / job_id
        written: list[Path] = []
        for clip_id, plan_json in rows:
            plan = ClipPlan.model_validate_json(plan_json)
            clip_dir = job_dir / "clips" / clip_id
            for source_name in (qc.FINAL_FILENAME, qc.THUMBNAIL_FILENAME):
                source = clip_dir / source_name
                if not qc.has_file(source):
                    raise QCActionsError(
                        f"cannot export {clip_id}: {source} is missing — "
                        "the clip was never rendered (or the output was "
                        "removed); export is read-only and does not "
                        "re-render"
                    )
            for source_name in (qc.FINAL_FILENAME, qc.THUMBNAIL_FILENAME):
                target = qc.copy_file(
                    clip_dir / source_name, out_dir / clip_id / source_name
                )
                written.append(target)
            metadata = {
                "clip_id": plan.clip_id,
                "job_id": job_id,
                "title": plan.title,
                "hook_overlay": plan.hook_overlay,
                "description": plan.description,
                "hashtags": plan.hashtags,
                "start": plan.start,
                "end": plan.end,
                "status": plan.status,
            }
            metadata_path = qc.write_text_file(
                out_dir / clip_id / "metadata.json",
                json.dumps(metadata, indent=2) + "\n",
            )
            written.append(metadata_path)
    return written
