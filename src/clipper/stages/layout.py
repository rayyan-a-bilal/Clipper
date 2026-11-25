"""
M6e layout stage (docs/spec/m6.md) — per-clip layout + crop path.

``layout`` stage (per-clip face pipeline; ALL I/O via adapters ONLY):

- required per-job artifacts (every input is ``schema_version == 1``; a
  clean ``StageError`` names the missing artifact AND the milestone/script
  that produces it):
    * ``clip_plans.json`` — M5C-5 ``clip_plans`` stage (``scripts/select_job.py``)
    * ``probe.json``      — M1 ingest stage       (``python -m clipper ingest``)
    * ``shots.json``      — M3b analysis stage    (``scripts/analyze_job.py``)
- for each DRAFT ClipPlan (or only ``clip_id`` when given) under
  ``data/jobs/<job_id>/clips/<clip_id>/``:
    1. ``vision.make_segment(...)`` → ``vision.mp4`` (960x540, no pad,
       fps = ``select_vision_fps(native_fps, cfg.faces.vision_fps_cap)``).
    2. decode EVERY frame (``frame_reader`` or the default lazy-cv2 reader)
       and run ``detector.detect(frame, t)`` per frame (``t`` =
       clip-relative seconds, ``t = frame_index / fps``).
    3. ``tracking.associate`` → ``bridge_tracks`` (M6g) → ``FaceTrack``
       list; ``tracking.summarize`` → ``TrackSummary`` (all pure, M6c/M6g).
    4. ``forced_layout`` (``layout_source="manual"``) or
       ``layouts.decide_layout(summary, w, h)`` (``layout_source="auto"``).
    5. ``layouts.build_tracks(..., tracks)`` — the M6d 8-arg form
       (``tracks`` REQUIRED as last parameter per the M6d judgment note).
    6. write ``layout.json`` (``ClipLayoutArtifact``, canonical
       ``model_dump_json(indent=2)`` + newline); update the ClipPlan's
       ``layout`` + ``crop_path`` (``main`` track keys — the ``top`` track
       for ``split_screen``) leaving ALL other fields intact.
- rewrites ``clip_plans.json`` (in full, in order) and upserts the updated
  ``clips`` rows.
- skip rule: DB stage state == ``done`` AND ALL scoped per-clip
  ``layout.json`` files exist and validate (``ClipLayoutArtifact``,
  ``schema_version == 1``).
- job goes ``queued -> running -> queued`` on success; on failure the stage
  + job are marked ``failed`` and a clean ``StageError`` is raised.
- ``detector.close()`` is ALWAYS called in ``finally`` (success AND failure).
- ZERO plans is a VALID success — no vision/detector work at all (the
  detector is never invoked, only closed).
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from ..adapters.vision import select_vision_fps
from ..config import Config
from ..layouts import build_tracks, decide_layout
from ..models import (
    ClipLayoutArtifact,
    ClipPlan,
    ClipPlansArtifact,
    FaceDetection,
    Layout,
    ProbeResult,
    ShotsArtifact,
)
from ..tracking import associate, bridge_tracks, summarize
from . import StageError
from .analyze import (
    _finish_stage,
    _now,
    _set_job_state,
    _start_stage,
)

logger = logging.getLogger("clipper.stages.layout")

_SCHEMA_VERSION = 1
_STAGE_NAME = "layout"
_SEGMENT_NAME = "vision.mp4"
_LAYOUT_FILE = "layout.json"

# A frame reader yields ``(frame_bgr, t)`` for EVERY frame of the segment
# (``t`` = clip-relative seconds). The default implementation decodes with a
# LAZILY imported cv2 ``VideoCapture``; tests inject a fake reader instead
# (CPU-only, zero GPU/ffmpeg — M6e judgment note in docs/spec/m6.md).
FrameReader = Callable[[str, float], Iterator[tuple[object, float]]]

_REQUIRED_INPUTS = (
    (
        "clip_plans.json",
        ClipPlansArtifact,
        "produced by the M5C-5 clip_plans stage (scripts/select_job.py)",
    ),
    (
        "probe.json",
        ProbeResult,
        "produced by the M1 ingest stage (python -m clipper ingest)",
    ),
    (
        "shots.json",
        ShotsArtifact,
        "produced by the M3b analysis stage (scripts/analyze_job.py)",
    ),
)


def default_frame_reader(path: str, fps: float) -> Iterator[tuple[object, float]]:
    """Decode EVERY frame of the vision segment (lazy cv2 import).

    Yields ``(frame_bgr, t)`` with ``t`` = frame index / ``fps``
    (clip-relative seconds). Tests never call this.
    """
    import cv2  # lazy import — keeps the module import light + tests CPU-only

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise StageError(f"Cannot open vision segment for decode: {path}")
    if fps <= 0.0:
        fps = 30.0  # defensive; make_segment always reports a positive fps
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame, index / fps
            index += 1
    finally:
        capture.release()


def _load_inputs(
    job_dir: Path,
) -> tuple[ClipPlansArtifact, ProbeResult, ShotsArtifact]:
    """Load + schema-validate the ``layout`` inputs (clean ``StageError``
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
        payload["probe.json"],
        payload["shots.json"],
    )


def _write_artifact(path: Path, artifact) -> None:
    """Canonical artifact form: ``model_dump_json(indent=2)`` + newline."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2) + "\n")
    except OSError as e:
        raise StageError(f"Failed to write artifact {path}: {e}") from e


def _clip_record(job_dir: Path, clip_id: str) -> dict:
    """Per-clip summary record read back from a VALID ``layout.json``."""
    path = job_dir / "clips" / clip_id / _LAYOUT_FILE
    artifact = ClipLayoutArtifact.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    role = _crop_role(artifact.layout)
    crop = next((rt for rt in artifact.tracks if rt.role == role), None)
    return {
        "clip_id": clip_id,
        "layout": artifact.layout.value,
        "face_track_count": artifact.face_track_count,
        "detector": artifact.detector,
        "keyframe_count": len(crop.keys) if crop else 0,
        "layout_source": artifact.layout_source,
        "elapsed_seconds": None,
    }


def _stage_state(conn, job_id: str) -> str | None:
    row = conn.execute(
        "SELECT state FROM stages WHERE job_id = ? AND name = ?",
        (job_id, _STAGE_NAME),
    ).fetchone()
    return row[0] if row else None


def _all_layouts_valid(job_dir: Path, clip_ids: list[str]) -> bool:
    """M6e skip rule: EVERY scoped clip has a valid ``layout.json``."""
    for clip_id in clip_ids:
        path = job_dir / "clips" / clip_id / _LAYOUT_FILE
        if not path.is_file():
            return False
        try:
            artifact = ClipLayoutArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            return False
        if artifact.schema_version != _SCHEMA_VERSION:
            return False
        if artifact.clip_id != clip_id:
            return False
    return True


def _crop_role(layout: Layout) -> str:
    """The track whose keys become ``ClipPlan.crop_path`` (M6e card):
    ``main`` for every layout EXCEPT split_screen (there ``top``)."""
    return "top" if layout is Layout.split_screen else "main"


def _shot_boundaries_in_window(
    shots: list, start: float, end: float,
) -> list[float]:
    """Shot boundaries clipped to the clip window, CLIP-RELATIVE
    (``build_tracks`` expects boundaries in ``(0, duration_s)``)."""
    return sorted(
        {round(s.start - start, 6) for s in shots if start < s.start < end}
    )


def _process_clip(
    plan: ClipPlan,
    job_dir: Path,
    cfg: Config,
    probe: ProbeResult,
    shots: list,
    source_path: str,
    vision,
    detector,
    forced_layout: Layout | None,
    frame_reader: FrameReader | None,
) -> tuple[ClipPlan, dict]:
    """One clip: segment → detect every frame → tracks → layout → artifact."""
    started = time.monotonic()
    clip_dir = job_dir / "clips" / plan.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    fps = select_vision_fps(probe.fps, cfg.faces.vision_fps_cap)
    segment = vision.make_segment(
        source_path,
        str(clip_dir / _SEGMENT_NAME),
        plan.start,
        plan.end,
        fps,
        int(cfg.faces.vision_width),
        int(cfg.faces.vision_height),
        clip_id=plan.clip_id,
    )

    reader = frame_reader if frame_reader is not None else default_frame_reader
    detections_by_frame: list[list[FaceDetection]] = [
        list(detector.detect(frame, t))
        for frame, t in reader(segment.path, segment.fps)
    ]

    tracks = bridge_tracks(associate(detections_by_frame))  # M6g
    summary = summarize(
        tracks, len(detections_by_frame), segment.width, segment.height
    )
    if forced_layout is not None:
        layout = forced_layout
    else:
        layout = decide_layout(
            summary,
            probe.width,
            probe.height,
            face_follow_enabled=bool(cfg.faces.face_follow_enabled),
            follow_persistence_min=float(cfg.faces.follow_persistence_min),
        )

    region_tracks = build_tracks(
        layout,
        summary,
        probe.width,
        probe.height,
        probe.fps,
        plan.end - plan.start,
        _shot_boundaries_in_window(shots, plan.start, plan.end),
        tracks,
    )
    role = _crop_role(layout)
    crop = next((rt for rt in region_tracks if rt.role == role), None)
    if crop is None:
        raise StageError(
            f"{plan.clip_id}: layout {layout.value!r} produced no "
            f"'{role}' track for crop_path"
        )
    updated_plan = plan.model_copy(
        update={"layout": layout, "crop_path": list(crop.keys)}
    )

    artifact = ClipLayoutArtifact(
        schema_version=_SCHEMA_VERSION,
        clip_id=plan.clip_id,
        layout=layout,
        source_width=probe.width,
        source_height=probe.height,
        native_fps=probe.fps,
        tracks=list(region_tracks),
        face_track_count=len(tracks),
        detector=detector.name(),
        layout_source="manual" if forced_layout is not None else "auto",
    )
    _write_artifact(clip_dir / _LAYOUT_FILE, artifact)

    record = {
        "clip_id": plan.clip_id,
        "layout": layout.value,
        "face_track_count": len(tracks),
        "detector": detector.name(),
        "keyframe_count": len(crop.keys),
        "layout_source": artifact.layout_source,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return updated_plan, record


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


def run_layout(
    job_id: str,
    cfg: Config,
    conn,
    vision,
    detector,
    clip_id: str | None = None,
    forced_layout: Layout | None = None,
    frame_reader: FrameReader | None = None,
) -> dict:
    """Run the ``layout`` stage for a job (M6e).

    Args:
        job_id: The job ID (must exist in the DB, with a ``clip_plans.json``
            from the M5C-5 stage).
        cfg: Configuration object (data_dir, faces.*).
        conn: sqlite3 connection over the clipper database.
        vision: A ``VisionAdapter`` (``FFmpegVisionAdapter`` in production,
            ``FixtureVisionAdapter`` in tests).
        detector: A ``FaceDetector`` (``YuNetOnnxDetector`` /
            ``Cv2CpuFaceDetector`` in production, ``FixtureFaceDetector`` in
            tests). ALWAYS closed in ``finally``.
        clip_id: Restrict the run to ONE clip (``None`` = all plans).
        forced_layout: Force a layout for all scoped clips
            (``layout_source="manual"``); ``None`` = auto decision.
        frame_reader: ``Callable[(path, fps) -> iterator[(frame, t)]``;
            ``None`` = default lazy-cv2 ``default_frame_reader``. Tests
            inject a fake reader (CPU-only; M6e judgment note).

    Returns:
        Summary dict: job_id, skipped, clips (per-clip records: clip_id,
        layout, face_track_count, detector, keyframe_count, layout_source,
        elapsed_seconds).

    Raises:
        StageError: Clean error on any failure (missing job/inputs, schema
            mismatch, decode/detection failure, write failure). The stage
            and job are marked failed; the detector is still closed.
    """
    row = conn.execute(
        "SELECT source_path FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")
    source_path = str(row[0] or "")

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    summary: dict = {"job_id": job_id, "skipped": False, "clips": []}
    try:
        plans_artifact, probe, shots = _load_inputs(job_dir)
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
            and _all_layouts_valid(job_dir, [p.clip_id for p in scope])
        ):
            summary["skipped"] = True
            summary["clips"] = [
                _clip_record(job_dir, p.clip_id) for p in scope
            ]
            return summary

        if scope:
            _set_job_state(conn, job_id, "running")
            _start_stage(conn, job_id, _STAGE_NAME)
        else:
            # ZERO clips: VALID success with no detector work at all —
            # still mark the stage done so re-runs skip.
            _start_stage(conn, job_id, _STAGE_NAME)

        updates: dict[str, ClipPlan] = {}
        records: list[dict] = []
        for plan in scope:
            updated_plan, record = _process_clip(
                plan, job_dir, cfg, probe, shots.shots,
                source_path, vision, detector, forced_layout, frame_reader,
            )
            updates[plan.clip_id] = updated_plan
            records.append(record)
        logger.info(
            "layout: %d clip(s) processed: %s",
            len(records),
            ", ".join(f"{r['clip_id']}={r['layout']}" for r in records),
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
            f"layout stage failed: {e}"
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
    finally:
        # M6e card: detector.close() ALWAYS, on success AND on failure.
        try:
            detector.close()
        except Exception as close_error:
            logger.warning(
                "layout: detector.close() raised (ignored): %s", close_error
            )


__all__ = [
    "FrameReader",
    "default_frame_reader",
    "run_layout",
]
