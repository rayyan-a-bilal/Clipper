"""M8b — worker loop: one job at a time, coarse stages from the M8a plan.

See `docs/spec/m8.md` (fixed M8 decisions). The worker orchestrates the
EXISTING M1–M7 stage functions via `StageRunner` objects — it never
reimplements stage logic. M8b ships the loop with injected runners only;
the real M1–M7 wiring lands in M8c (per the M8b task card).

Guarantees (M8 decisions 2/5/6/7/8):
- one job at a time (worker file lock `data/run/worker.lock` +
  `run_next` serializing iterations), never two jobs concurrently,
- job `queued/failed -> running`, then `awaiting_qc` (>= 1 clip) /
  `done` (zero clips) / `failed` (stage error: job stops, report written,
  clean `WorkerError` raised),
- retry ONLY via `retry=True` = `db.reset_failed_stages` (failed fine
  stages back to `pending`; done stages and artifacts are never touched),
- per-job `report.json` (`data/jobs/<job_id>/report.json`) + the same
  canonical JSON in the `job_reports` table (`db.write_job_report`,
  M8a judgment note M8a-J5).

CPU-safe: pure orchestration over the M8a plan; no adapters, no GPU,
no network, no LLM. Stage work (and any failure) belongs to the runners.
"""

from __future__ import annotations

import fcntl
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from . import db, worker_plan
from .config import Config
from .models import JobReport, StageReport
from .worker_plan import StageSpec

__all__ = [
    "FakeStageRunner",
    "StageRunner",
    "WorkerError",
    "lock_path",
    "run_job",
    "run_next",
    "worker_lock",
]


class WorkerError(Exception):
    """Clean worker error (unknown job, lock contended, stage failure)."""


class StageRunner(Protocol):
    """Runs one coarse stage end to end (an existing M1–M7 stage function).

    A runner owns fine-grained stage bookkeeping for its coarse stage
    (the existing stages already do this). It signals failure by
    raising; the worker marks the job failed, writes the report, and
    raises a clean `WorkerError`.
    """

    def run(
        self, job_id: str, cfg: Config, conn: sqlite3.Connection
    ) -> None:
        """Execute the coarse stage for one job."""
        ...


class FakeStageRunner:
    """CPU test double for `StageRunner`.

    Records call order in `calls` and (optionally) raises `error` on
    every call. Zero I/O, zero GPU, zero LLM.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.error = error

    def run(
        self, job_id: str, cfg: Config, conn: sqlite3.Connection
    ) -> None:
        self.calls.append(job_id)
        if self.error is not None:
            raise self.error


def _now_iso() -> str:
    return datetime.now().isoformat()


def lock_path(cfg: Config) -> Path:
    """The worker file lock: `<data_dir>/run/worker.lock`."""
    return cfg.resolved_data_dir / "run" / "worker.lock"


@contextmanager
def worker_lock(cfg: Config) -> Iterator[object]:
    """Exclusive, non-blocking `flock` on the worker lock file.

    Raises `WorkerError` when another worker already holds it — that
    is how "never two jobs concurrently" is enforced even across
    processes (and it makes the contention CPU-testable).
    """
    path = lock_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise WorkerError(
            f"worker lock held by another worker: {path} ({exc}); "
            "one job at a time (M8 decision 2)"
        ) from exc
    try:
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _runners_for(runners: dict[str, object]) -> dict[str, StageRunner]:
    missing = [spec.name for spec in worker_plan.STAGE_PLAN if spec.name not in runners]
    if missing:
        raise WorkerError(f"no runner registered for coarse stage(s): {missing}")
    return runners  # type: ignore[return-value]


def _clip_count(conn: sqlite3.Connection, job_id: str) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM clips WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def _staged_artifacts(
    conn: sqlite3.Connection, job_id: str, spec: StageSpec
) -> list[str]:
    """The artifact paths recorded on this coarse stage's fine rows."""
    cursor = conn.cursor()
    paths: list[str] = []
    for name in spec.required:
        cursor.execute(
            "SELECT artifact_path FROM stages WHERE job_id = ? AND name = ?",
            (job_id, name),
        )
        row = cursor.fetchone()
        if row and row[0]:
            paths.append(row[0])
    return paths


def _mark_fine_failed(
    conn: sqlite3.Connection, job_id: str, spec: StageSpec, message: str
) -> None:
    """Mark the current coarse stage's first non-done fine stage `failed`.

    Never overwrites a fine stage the runner already marked `failed`
    (the stage owns its own bookkeeping; M8 decision 6) and never
    touches done stages.
    """
    cursor = conn.cursor()
    for name in spec.required:
        cursor.execute(
            "SELECT state FROM stages WHERE job_id = ? AND name = ?",
            (job_id, name),
        )
        row = cursor.fetchone()
        state = row[0] if row else None
        if state in worker_plan.DONE_STATES:
            continue
        if state != "failed":
            db.set_stage_state(conn, job_id, name, "failed", error=message)
        return


def _persist_report(
    cfg: Config, conn: sqlite3.Connection, job_id: str, report: JobReport
) -> Path:
    """Write `report.json` (M8 decision 8 / M8a-J5) AND the DB row."""
    report_json = db.write_job_report(conn, job_id, report)
    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "report.json"
    path.write_text(report_json + "\n", encoding="utf-8")
    return path


def run_job(
    job_id: str,
    cfg: Config,
    conn: sqlite3.Connection,
    runners: dict[str, StageRunner],
    retry: bool = False,
) -> JobReport:
    """Run one job start to finish (M8 decisions 2–8).

    Steps: acquire the worker lock; load the job (clean `WorkerError`
    when missing); `reset_failed_stages` when `retry`; job ->
    `running`; iterate `worker_plan.STAGE_PLAN` — skip a coarse stage
    only when `worker_plan.is_coarse_done`, else run its runner; on
    failure mark the current fine stage + job `failed`, write the
    report, and raise a clean `WorkerError` (later stages never start);
    on success: job -> `awaiting_qc` (>= 1 clip) or `done` (zero clips);
    write `report.json` + DB report; release the lock (always).
    """
    resolved = _runners_for(runners)
    with worker_lock(cfg):
        if db.get_job(conn, job_id) is None:
            raise WorkerError(
                f"unknown job {job_id!r} (no jobs row in the database)"
            )
        if retry:
            db.reset_failed_stages(conn, job_id)
        db.set_job_state(conn, job_id, "running")
        started_at = _now_iso()
        stage_reports: list[StageReport] = []
        try:
            for spec in worker_plan.STAGE_PLAN:
                if worker_plan.is_coarse_done(conn, job_id, spec):
                    stage_reports.append(
                        StageReport(
                            name=spec.name,
                            required=list(spec.required),
                            state="skipped",
                            artifact_paths=_staged_artifacts(conn, job_id, spec),
                        )
                    )
                    continue
                entry = StageReport(
                    name=spec.name,
                    required=list(spec.required),
                    state="running",
                    started_at=_now_iso(),
                )
                stage_reports.append(entry)
                resolved[spec.name].run(job_id, cfg, conn)
                entry.state = "done"
                entry.finished_at = _now_iso()
                entry.artifact_paths = _staged_artifacts(conn, job_id, spec)
        except Exception as exc:
            for entry in reversed(stage_reports):
                if entry.state == "running":
                    spec = next(
                        s for s in worker_plan.STAGE_PLAN if s.name == entry.name
                    )
                    entry.state = "failed"
                    entry.finished_at = _now_iso()
                    entry.error = str(exc)
                    _mark_fine_failed(conn, job_id, spec, str(exc))
                    db.set_job_state(conn, job_id, "failed")
                    failed_report = JobReport(
                        job_id=job_id,
                        state="failed",
                        clip_count=_clip_count(conn, job_id),
                        stages=stage_reports,
                        started_at=started_at,
                        finished_at=_now_iso(),
                        error=str(exc),
                    )
                    _persist_report(cfg, conn, job_id, failed_report)
                    raise WorkerError(
                        f"job {job_id}: coarse stage {entry.name!r} failed: "
                        f"{exc}"
                    ) from exc
            raise  # unreachable: the raised WorkerError above exits
        clip_count = _clip_count(conn, job_id)
        final_state = "awaiting_qc" if clip_count > 0 else "done"
        db.set_job_state(conn, job_id, final_state)
        report = JobReport(
            job_id=job_id,
            state=final_state,
            clip_count=clip_count,
            stages=stage_reports,
            started_at=started_at,
            finished_at=_now_iso(),
        )
        _persist_report(cfg, conn, job_id, report)
        return report


def run_next(
    cfg: Config,
    conn: sqlite3.Connection,
    runners: dict[str, StageRunner],
    loop: bool = False,
    poll_s: float = 2.0,
) -> JobReport | None:
    """Process the next `queued` job (one job at a time, M8 decision 2).

    - without `loop`: run the oldest queued job once; `None` when no job
      is queued.
    - with `loop`: poll every `poll_s` seconds until stopped
      (`KeyboardInterrupt` exits cleanly). Each iteration runs through
      `run_job`, which holds the worker lock for the whole job — two
      jobs are never started concurrently.
    """
    _runners_for(runners)
    if not loop:
        job = db.next_queued_job(conn)
        if job is None:
            return None
        return run_job(job["job_id"], cfg, conn, runners, retry=False)

    last: JobReport | None = None
    try:
        while True:
            job = db.next_queued_job(conn)
            if job is not None:
                last = run_job(job["job_id"], cfg, conn, runners, retry=False)
            time.sleep(poll_s)
    except KeyboardInterrupt:
        pass
    return last
