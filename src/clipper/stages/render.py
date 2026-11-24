"""
M7d render stage (docs/spec/m7.md) — center_crop render, end to end.

``render`` stage (ALL I/O via adapters ONLY — the ``RenderAdapter`` is
injected, mirroring the M6e layout stage):

- required per-job artifacts (every input is ``schema_version == 1``; a
  clean ``StageError`` names the missing artifact AND the milestone/script
  that produces it):
    * ``clip_plans.json`` — M5C-5 clip_plans stage (``scripts/select_job.py``)
    * ``words.json``      — M2b transcribe stage (``scripts/transcribe_job.py``)
    * ``probe.json``      — M1 ingest stage       (``python -m clipper ingest``)
  plus, per clip, ``clips/<clip_id>/layout.json`` — M6e layout stage
  (``scripts/layout_job.py``).
- for each ClipPlan (or only ``clip_id`` when given) under
  ``data/jobs/<job_id>/clips/<clip_id>/``:
    1. ``render.render_mezzanine(...)`` → ``mezzanine.mp4`` (1080x1920,
       native fps from ``probe.json``).
    2. ``render.extract_audio(...)``    → ``audio.m4a``.
    3. ``captions.words_to_ass(...)``   → ``captions.ass`` (pure, M7a).
    4. ``render.compose_final(...)``    → ``final.mp4`` (ASS burn +
       loudnorm + NVENC).
    5. ``render.make_thumbnail(...)``   → ``thumbnail.jpg``.
    6. write ``render.json`` (``RenderArtifact``, canonical
       ``model_dump_json(indent=2)`` + newline; the stage fills
       ``clip_id`` + ``thumbnail_path`` on the PARTIAL artifact the
       adapter returns).
  The ClipPlan's ``status`` is updated to ``"rendered"`` (all other
  fields intact); ``clip_plans.json`` is rewritten in full (in order) and
  the ``clips`` rows are upserted.
- skip rule: DB stage state == ``done`` AND, for EVERY scoped clip,
  ``final.mp4`` exists AND ``render.json`` validates (``RenderArtifact``,
  ``schema_version == 1``, matching ``clip_id``).
- job goes ``queued -> running -> queued`` on success; on failure the
  stage + job are marked ``failed`` and a clean ``StageError`` is raised.
- ZERO plans is a VALID success — no render work at all.
- the ``center_crop`` static path renders correctly with ``faces == 0``
  (decision 8) — the stage never requires face tracks to be non-empty.
- ``cpu_render=True`` is the EXPLICIT debug path (decision 2) and is
  logged LOUDLY at the stage entry point.
"""

import json
import logging
import time
from pathlib import Path

from ..adapters.render import CPU_RENDER_WARNING
from ..captions import words_to_ass
from ..config import Config
from ..models import (
    ClipLayoutArtifact,
    ClipPlan,
    ClipPlansArtifact,
    ProbeResult,
    RenderArtifact,
    WordsArtifact,
)
from . import StageError
from .analyze import (
    _finish_stage,
    _now,
    _set_job_state,
    _start_stage,
)

logger = logging.getLogger("clipper.stages.render")

_SCHEMA_VERSION = 1
_STAGE_NAME = "render"

# Per-clip output file names (docs/spec/m7.md decision 4).
_MEZZANINE_FILE = "mezzanine.mp4"
_AUDIO_FILE = "audio.m4a"
_CAPTIONS_FILE = "captions.ass"
_FINAL_FILE = "final.mp4"
_THUMBNAIL_FILE = "thumbnail.jpg"
_RENDER_FILE = "render.json"

_REQUIRED_INPUTS = (
    (
        "clip_plans.json",
        ClipPlansArtifact,
        "produced by the M5C-5 clip_plans stage (scripts/select_job.py)",
    ),
    (
        "words.json",
        WordsArtifact,
        "produced by the M2b transcribe stage (scripts/transcribe_job.py)",
    ),
    (
        "probe.json",
        ProbeResult,
        "produced by the M1 ingest stage (python -m clipper ingest)",
    ),
)


def _load_inputs(job_dir: Path) -> tuple[ClipPlansArtifact, WordsArtifact, ProbeResult]:
    """Load + schema-validate the ``render`` inputs (clean ``StageError``
    naming the missing/offending artifact AND its producer milestone)."""
    payload: dict[str, object] = {}
    for name, model, hint in _REQUIRED_INPUTS:
        path = job_dir / name
        if not path.exists():
            raise StageError(
                f"Missing required input {path} — {hint}; run that stage first"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise StageError(f"Failed to read {name}: {e}") from e

        version = data.get("schema_version") if isinstance(data, dict) else None
        if version != _SCHEMA_VERSION:
            raise StageError(
                f"{name} has schema_version={version!r} "
                f"(expected {_SCHEMA_VERSION}) — {hint}"
            )
        try:
            payload[name] = model.model_validate(data)
        except Exception as e:
            raise StageError(f"{name} is not a valid artifact: {e}") from e
    return (  # type: ignore[return-value]
        payload["clip_plans.json"],
        payload["words.json"],
        payload["probe.json"],
    )


def _write_artifact(path: Path, artifact) -> None:
    """Canonical artifact form: ``model_dump_json(indent=2)`` + newline."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2) + "\n")
    except OSError as e:
        raise StageError(f"Failed to write artifact {path}: {e}") from e


def _load_layout(job_dir: Path, clip_id: str) -> ClipLayoutArtifact:
    """Read + validate ``clips/<clip_id>/layout.json`` (clean
    ``StageError`` naming the artifact AND the M6e producer)."""
    path = job_dir / "clips" / clip_id / "layout.json"
    if not path.is_file():
        raise StageError(
            f"Missing required input {path} — produced by the M6e layout "
            f"stage (scripts/layout_job.py); run that stage first"
        )
    try:
        artifact = ClipLayoutArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as e:
        raise StageError(
            f"{path} is not a valid layout artifact: "
            f"{e} — produced by the M6e layout stage (scripts/layout_job.py)"
        ) from e
    if artifact.schema_version != _SCHEMA_VERSION:
        raise StageError(
            f"{path} has schema_version={artifact.schema_version!r} "
            f"(expected {_SCHEMA_VERSION})"
        )
    if artifact.clip_id != clip_id:
        raise StageError(
            f"{path} belongs to clip {artifact.clip_id!r}, "
            f"not {clip_id!r}"
        )
    return artifact


def _render_record(job_dir: Path, clip_id: str) -> dict:
    """Per-clip summary record read back from a VALID ``render.json``."""
    path = job_dir / "clips" / clip_id / _RENDER_FILE
    artifact = RenderArtifact.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return _artifact_record(clip_id, artifact)


def _artifact_record(clip_id: str, artifact: RenderArtifact) -> dict:
    return {
        "clip_id": clip_id,
        "final_path": artifact.final_path,
        "duration_s": artifact.duration_s,
        "width": artifact.width,
        "height": artifact.height,
        "fps": artifact.fps,
        "encoder": artifact.encoder,
        "hwaccel": artifact.hwaccel,
        "render_seconds": artifact.render_seconds,
    }


def _stage_state(conn, job_id: str) -> str | None:
    row = conn.execute(
        "SELECT state FROM stages WHERE job_id = ? AND name = ?",
        (job_id, _STAGE_NAME),
    ).fetchone()
    return row[0] if row else None


def _all_renders_valid(job_dir: Path, clip_ids: list[str]) -> bool:
    """M7d skip rule: EVERY scoped clip has a ``final.mp4`` AND a valid
    ``render.json`` (``RenderArtifact``, ``schema_version == 1``, matching
    ``clip_id``)."""
    for clip_id in clip_ids:
        clip_dir = job_dir / "clips" / clip_id
        if not (clip_dir / _FINAL_FILE).is_file():
            return False
        path = clip_dir / _RENDER_FILE
        if not path.is_file():
            return False
        try:
            artifact = RenderArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            return False
        if artifact.schema_version != _SCHEMA_VERSION:
            return False
        if artifact.clip_id != clip_id:
            return False
        if not Path(artifact.final_path).is_file():
            return False
    return True


def _upsert_clips(conn, job_id: str, plans: list[ClipPlan]) -> None:
    """Upsert one ``clips`` row per UPDATED plan (clip_id is the PK)."""
    now = _now()
    for p in plans:
        conn.execute(
            """
            INSERT INTO clips
                (clip_id, job_id, plan_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                plan_json = excluded.plan_json,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (p.clip_id, job_id, p.model_dump_json(), p.status, now, now),
        )
    conn.commit()


def _process_clip(
    plan: ClipPlan,
    job_dir: Path,
    cfg: Config,
    words: list,
    probe: ProbeResult,
    source_path: str,
    render,
) -> tuple[ClipPlan, dict]:
    """One clip: layout → mezzanine → audio → ASS → final → thumbnail →
    ``render.json`` (all render I/O through the injected adapter)."""
    started = time.monotonic()
    clip_dir = job_dir / "clips" / plan.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    layout_art = _load_layout(job_dir, plan.clip_id)

    mezzanine = render.render_mezzanine(
        source_path,
        plan,
        layout_art.layout,
        str(clip_dir / _MEZZANINE_FILE),
        cfg,
        native_fps=probe.fps,
    )
    audio = render.extract_audio(
        source_path, plan.start, plan.end, str(clip_dir / _AUDIO_FILE)
    )

    captions_path = clip_dir / _CAPTIONS_FILE
    try:
        captions_path.write_text(words_to_ass(words, plan, cfg), encoding="utf-8")
    except OSError as e:
        raise StageError(f"Failed to write {captions_path}: {e}") from e

    final_path = str(clip_dir / _FINAL_FILE)
    partial = render.compose_final(
        mezzanine.path, audio.path, str(captions_path), final_path, cfg
    )
    timestamp = render.make_thumbnail(
        final_path, str(clip_dir / _THUMBNAIL_FILE)
    )

    artifact = partial.model_copy(
        update={"clip_id": plan.clip_id, "thumbnail_path": timestamp}
    )
    _write_artifact(clip_dir / _RENDER_FILE, artifact)

    updated_plan = plan.model_copy(update={"status": "rendered"})
    record = _artifact_record(plan.clip_id, artifact)
    record["layout"] = layout_art.layout.value
    record["face_track_count"] = layout_art.face_track_count
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return updated_plan, record


def run_render(
    job_id: str,
    cfg: Config,
    conn,
    render,
    clip_id: str | None = None,
    cpu_render: bool = False,
) -> dict:
    """Run the ``render`` stage for a job (M7d).

    Args:
        job_id: The job ID (must exist in the DB, with a
            ``clip_plans.json`` from the M5C-5 stage and per-clip
            ``layout.json`` from the M6e stage).
        cfg: Configuration object (data_dir, render.*).
        conn: sqlite3 connection over the clipper database.
        render: A ``RenderAdapter`` (``FFmpegRenderAdapter`` in
            production, ``FixtureRenderAdapter`` in tests). Injected —
            never constructed here.
        clip_id: Restrict the run to ONE clip (``None`` = all plans).
        cpu_render: ``True`` = EXPLICIT CPU/DEBUG render path (decision 2)
            — logged LOUDLY.

    Returns:
        Summary dict: job_id, skipped, clips (per-clip records: clip_id,
        final_path, duration_s, width, height, fps, encoder, hwaccel,
        render_seconds, layout, face_track_count, elapsed_seconds).

    Raises:
        StageError: Clean error on any failure (missing job/inputs,
            schema mismatch, render failure, write failure). The stage
            and job are marked failed.
    """
    if cpu_render:
        # Loud (docs/spec/m7.md decision 2): the debug path must announce
        # itself even when the adapter was already constructed upstream.
        logger.warning(CPU_RENDER_WARNING)

    row = conn.execute(
        "SELECT source_path FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")
    source_path = str(row[0] or "")

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    summary: dict = {"job_id": job_id, "skipped": False, "clips": []}
    try:
        plans_artifact, words_artifact, probe = _load_inputs(job_dir)
        plans = plans_artifact.plans
        if clip_id is not None:
            scope = [p for p in plans if p.clip_id == clip_id]
            if not scope:
                raise StageError(
                    f"Clip {clip_id} not found in clip_plans.json "
                    f"for job {job_id}"
                )
        else:
            scope = list(plans)

        if (
            _stage_state(conn, job_id) == "done"
            and _all_renders_valid(job_dir, [p.clip_id for p in scope])
        ):
            summary["skipped"] = True
            summary["clips"] = [
                _render_record(job_dir, p.clip_id) for p in scope
            ]
            return summary

        if scope:
            _set_job_state(conn, job_id, "running")
            _start_stage(conn, job_id, _STAGE_NAME)
        else:
            # ZERO clips: VALID success with no render work at all —
            # still mark the stage done so re-runs skip.
            _start_stage(conn, job_id, _STAGE_NAME)

        updates: dict[str, ClipPlan] = {}
        records: list[dict] = []
        for plan in scope:
            updated_plan, record = _process_clip(
                plan, job_dir, cfg, words_artifact.words, probe,
                source_path, render,
            )
            updates[plan.clip_id] = updated_plan
            records.append(record)
        logger.info(
            "render: %d clip(s) processed: %s",
            len(records),
            ", ".join(
                f"{r['clip_id']} ({r['duration_s']}s, {r['encoder']})"
                for r in records
            ),
        )

        if scope:
            new_plans = [updates.get(p.clip_id, p) for p in plans]
            _write_artifact(
                job_dir / "clip_plans.json",
                ClipPlansArtifact(
                    schema_version=_SCHEMA_VERSION, plans=new_plans
                ),
            )
            _upsert_clips(conn, job_id, list(updates.values()))

        _finish_stage(conn, job_id, _STAGE_NAME, str(job_dir / "clips"))
        _set_job_state(conn, job_id, "queued")
        summary["clips"] = records
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"render stage failed: {e}"
        )
        try:
            # Guarantee a stage row exists so the failure is recordable
            # (the error may have happened before _start_stage ran).
            _start_stage(conn, job_id, _STAGE_NAME)
        except Exception:
            pass
        try:
            _finish_stage(conn, job_id, _STAGE_NAME, error=str(clean))
        except Exception:
            pass
        try:
            _set_job_state(conn, job_id, "failed")
        except Exception:
            pass
        if clean is e:
            raise
        raise clean from e


__all__ = [
    "run_render",
]
