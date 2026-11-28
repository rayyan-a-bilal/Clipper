"""
CLI interface for clipper.
"""

import json
from pathlib import Path

import typer

from . import qc_actions, worker
from .adapters.media import get_adapter
from .config import Config
from .db import Database
from .db import list_jobs as list_jobs_db
from .devices import DeviceManager
from .stages.ingest import run_ingest
from .worker_plan import STAGE_PLAN, is_coarse_done

app = typer.Typer()

def get_config() -> Config:
    """Get the configuration."""
    return Config.load_config()

def get_db(config: Config) -> Database:
    """Get the database connection."""
    return Database(config)

def get_devices(config: Config) -> DeviceManager:
    """Get device manager."""
    return DeviceManager(config)

@app.command()
def doctor():
    """
    Check system for clipper requirements.
    """
    typer.echo("Checking clipper system requirements...")

    # Load configuration
    try:
        config = get_config()
        typer.echo("✓ Configuration loaded successfully")
    except Exception as e:
        typer.echo(f"✗ Failed to load configuration: {e}")
        raise typer.Exit(1) from e

    # Check data directory
    try:
        if not config.resolved_data_dir.exists():
            config.resolved_data_dir.mkdir(parents=True, exist_ok=True)
        typer.echo("✓ Data directory is accessible")
    except Exception as e:
        typer.echo(f"✗ Data directory inaccessible: {e}")
        raise typer.Exit(1) from e

    # Check database
    try:
        db = get_db(config)
        version = db.get_schema_version()
        typer.echo(f"✓ Database initialized (schema version {version})")
    except Exception as e:
        typer.echo(f"✗ Failed to connect to database: {e}")
        raise typer.Exit(1) from e

    # Check devices
    try:
        devices = get_devices(config)
        device_info = devices.get_all_devices()

        if "nvidia" in device_info and "gpus" in device_info["nvidia"]:
            gpus = device_info["nvidia"]["gpus"]
            typer.echo(f"✓ NVIDIA GPUs detected: {len(gpus)} GPU(s)")
            for gpu in gpus:
                label = f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']} MB)"
            typer.echo(f"  - {label}")
        else:
            typer.echo(
                "⚠ No NVIDIA GPUs available (this is okay for CPU-only processing)"
            )

        if "ffmpeg" in device_info and device_info["ffmpeg"].get("ffmpeg_available"):
            typer.echo("✓ FFmpeg detected")
        else:
            typer.echo("⚠ FFmpeg not found (will be required for video processing)")

        if "ffmpeg" in device_info and device_info["ffmpeg"].get("error"):
            typer.echo(f"  Error: {device_info['ffmpeg']['error']}")

    except Exception as e:
        typer.echo(f"✗ Device detection error: {e}")
        raise typer.Exit(1) from e

    # Check fonts directory
    fonts_dir = Path("./assets/fonts")
    if fonts_dir.exists():
        typer.echo("✓ Fonts directory found")
    else:
        typer.echo("⚠ Fonts directory not found")

    # Check Python version
    import sys
    typer.echo(f"✓ Python {sys.version}")

    typer.echo("System check completed successfully!")

@app.command()
def ingest(
    source_path: str = typer.Option(..., help="Path to source video file"),
    client: str = typer.Option(..., help="Client name"),
    niche: str = typer.Option(..., help="Niche/topic")
):
    """
    Ingest a new video for processing.
    """
    try:
        # Load configuration
        config = get_config()

        # Get database connection
        db = get_db(config)
        conn = db.get_connection()

        # Get media adapter
        media = get_adapter()

        # Run ingestion pipeline
        job_id, is_new_job, should_retry = run_ingest(
            source=source_path,
            client=client,
            niche=niche,
            cfg=config,
            conn=conn,
            media=media
        )

        # Print results
        typer.echo(f"Job ID: {job_id}")
        if is_new_job:
            typer.echo("Created new job")
        elif should_retry:
            typer.echo("Retrying failed job")
        else:
            typer.echo("Using existing job (already completed)")

        # Print job details
        job = db.get_job(job_id)
        if job:
            typer.echo(f"Source: {job['source_path']}")
            typer.echo(f"Client: {job['client']}")
            typer.echo(f"Niche: {job['niche']}")

        conn.close()

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

def db_get_job(config: Config, job_id: str) -> dict | None:
    """One job row (CLI helper)."""
    db = get_db(config)
    conn = db.get_connection()
    try:
        return db.get_job(job_id)
    finally:
        conn.close()


def _stage_runners(cfg: Config) -> dict[str, worker.StageRunner]:
    """The REAL M1–M7 stage wiring for `clipper run` (M8c).

    Returns `clipper.worker_stages.STAGE_RUNNERS` — one real runner per
    coarse stage, keyed by the exact `worker_plan.STAGE_PLAN` names
    (validated by `worker._runners_for`). Tests may still monkeypatch
    this hook to inject fake runners.
    """
    from . import worker_stages

    return worker_stages.STAGE_RUNNERS


def _print_report_summary(job_id: str, cfg: Config) -> None:
    """`clipper report JOB_ID` — path + summary (M8b task card)."""
    path: Path = cfg.resolved_data_dir / "jobs" / job_id / "report.json"
    payload: str | None
    if path.exists():
        payload = path.read_text(encoding="utf-8")
    else:
        db = get_db(cfg)
        conn = db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT report_json FROM job_reports WHERE job_id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        payload = row[0] if row else None
    if payload is None:
        typer.echo(f"No report found for job {job_id}")
        raise typer.Exit(1)
    data = json.loads(payload)
    if path.exists():
        typer.echo(f"report: {path}")
    else:
        typer.echo(
            "report: (database only — no report.json file; job_reports row present)"
        )
    typer.echo(f"job: {data['job_id']}")
    typer.echo(f"state: {data['state']}")
    typer.echo(f"clips: {data['clip_count']}")
    for stage in data["stages"]:
        line = f"stage {stage['name']}: {stage['state']}"
        if stage["error"]:
            line += f"  error: {stage['error']}"
        typer.echo(line)
    if data["error"]:
        typer.echo(f"error: {data['error']}")


@app.command()
def run(
    job_id: str | None = typer.Argument(None, help="Job ID to process"),
    next_job: bool = typer.Option(
        False, "--next", help="Process the oldest queued job"
    ),
    loop: bool = typer.Option(
        False, "--loop", help="Keep polling for queued jobs (with --next)"
    ),
    poll: float = typer.Option(
        2.0, "--poll", help="Poll interval in seconds (with --loop)"
    ),
    retry: bool = typer.Option(
        False, "--retry", help="Reset failed stages first (with JOB_ID)"
    ),
):
    """
    Run one job (JOB_ID) or the oldest queued job (--next [--loop]).
    """
    if bool(job_id) == next_job:
        typer.echo("error: pass exactly one of JOB_ID or --next")
        raise typer.Exit(1)
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        runners = _stage_runners(config)
        if next_job:
            report = worker.run_next(config, conn, runners, loop=loop, poll_s=poll)
            if report is None:
                typer.echo("No queued jobs")
                raise typer.Exit(0)
            job_id = report.job_id
        else:
            assert job_id is not None
            report = worker.run_job(job_id, config, conn, runners, retry=retry)
        typer.echo(f"Job ID: {job_id}")
        typer.echo(f"state: {report.state}")
        typer.echo(f"clips: {report.clip_count}")
        typer.echo(
            f"report: {config.resolved_data_dir / 'jobs' / job_id / 'report.json'}"
        )
        conn.close()
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to check status of"),
):
    """
    Show a job's state + coarse/fine stage states.
    """
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        try:
            job = db.get_job(job_id)
            if job is None:
                typer.echo(f"Unknown job: {job_id}")
                raise typer.Exit(1)
            typer.echo(f"Job: {job_id}")
            typer.echo(f"State: {job['state']}")
            typer.echo("Coarse stages:")
            cursor = conn.cursor()
            for spec in STAGE_PLAN:
                done = is_coarse_done(conn, job_id, spec)
                fine_bits = []
                for name in spec.required:
                    cursor.execute(
                        "SELECT state FROM stages WHERE job_id = ? AND name = ?",
                        (job_id, name),
                    )
                    row = cursor.fetchone()
                    fine_bits.append(f"{name}={row[0] if row else '-'}")
                typer.echo(
                    f"  {spec.name:<12} {'done' if done else 'pending':<8} "
                    f"({', '.join(fine_bits)})"
                )
        finally:
            conn.close()
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


@app.command()
def ls(
    state: str | None = typer.Option(None, "--state", help="Filter jobs by state"),
):
    """
    List all jobs in the system (oldest first).
    """
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        try:
            jobs = list_jobs_db(conn, state=state)
        finally:
            conn.close()
        if not jobs:
            note = f" (state={state})" if state else ""
            typer.echo(f"No jobs{note}")
            raise typer.Exit(0)
        typer.echo(f"{'JOB ID':<36} {'STATE':<12} {'CREATED':<27} SOURCE")
        for row in jobs:
            typer.echo(
                f"{row['job_id']:<36} {row['state']:<12} "
                f"{row['created_at']:<27} {row['source_path']}"
            )
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

@app.command()
def approve(
    clip_id: str = typer.Argument(..., help="Clip ID to approve"),
):
    """
    Approve a clip for final export (M9d, m9-5).
    """
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        try:
            plan = qc_actions.approve(clip_id, config, conn)
        finally:
            conn.close()
        typer.echo(f"{clip_id}: {plan.status}")
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


@app.command()
def reject(
    clip_id: str = typer.Argument(..., help="Clip ID to reject"),
):
    """
    Reject a clip (M9d, m9-5).
    """
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        try:
            plan = qc_actions.reject(clip_id, config, conn)
        finally:
            conn.close()
        typer.echo(f"{clip_id}: {plan.status}")
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

@app.command()
def rerender(
    clip_id: str = typer.Option(..., help="Clip ID to re-render"),
):
    """
    Re-render a clip.
    """
    typer.echo("Not implemented until M10")
    raise typer.Exit(0)

@app.command()
def export(
    job_id: str = typer.Argument(..., help="Job ID whose approved clips to export"),
):
    """
    Export a job's approved clips to data/exports/<job_id>/ (M9d, m9-5).
    """
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        try:
            paths = qc_actions.export_job(job_id, config, conn)
        finally:
            conn.close()
        out_dir = config.resolved_data_dir / "exports" / job_id
        if not paths:
            typer.echo(f"No approved clips to export for job {job_id}")
        else:
            typer.echo(f"Exported {len(paths)} file(s) to {out_dir}")
            for path in paths:
                typer.echo(f"  {path}")
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


@app.command()
def force_select(
    cand_id: str = typer.Argument(
        ..., help="Candidate ID to force-select (no LLM)"
    ),
    reason: str | None = typer.Option(
        None, "--reason", help="Free-text reason captured on the plan"
    ),
):
    """
    Force-select a candidate as a clip WITHOUT any LLM call (M9d, m9-4).
    """
    try:
        config = get_config()
        db = get_db(config)
        conn = db.get_connection()
        try:
            plan = qc_actions.force_select(cand_id, reason, config, conn)
        finally:
            conn.close()
        typer.echo(f"{plan.clip_id}: forced selection from candidate {cand_id}")
        typer.echo(f"status: {plan.status}")
        if plan.force_reason:
            typer.echo(f"reason: {plan.force_reason}")
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

@app.command()
def report(
    job_id: str = typer.Argument(..., help="Job ID whose report to show"),
):
    """
    Show a job's report.json (path + summary).
    """
    try:
        config = get_config()
        if db_get_job(config, job_id) is None:
            typer.echo(f"Unknown job: {job_id}")
            raise typer.Exit(1)
        _print_report_summary(job_id, config)
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

if __name__ == "__main__":
    app()
