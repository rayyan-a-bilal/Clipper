"""
M6e layout-stage tests — CPU only (FixtureVisionAdapter +
FixtureFaceDetector + injected frame reader + temp DB; NO GPU/ffmpeg/model).

Covers (stage level): per-clip ``layout.json`` + ``clip_plans.json`` +
``clips`` row updates; no faces → static ``center_crop``; forced layout →
``layout_source="manual"``; ``--clip`` limits work to ONE clip; second run
skips the completed stage; ``detector.close()`` on success AND on
exception; missing inputs fail cleanly; zero clips succeeds WITHOUT
invoking the detector; ``crop_path`` starts at t=0.0 and is clip-relative;
determinism with the fixture detector.
"""

import json

import pytest

from clipper.adapters.faces import FixtureFaceDetector
from clipper.adapters.vision import FixtureVisionAdapter
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.models import (
    ClipLayoutArtifact,
    ClipPlan,
    ClipPlansArtifact,
    Layout,
    ProbeResult,
    Shot,
    ShotsArtifact,
)
from clipper.stages import StageError
from clipper.stages.layout import run_layout

JOB_ID = "job-m6e"

# Clip window: [10.0, 14.0) @ 30fps -> 120 frames, 121 crop keys (t=0..4).
START, END = 10.0, 14.0
N_FRAMES = 120
# One large, stationary face (area 0.09 >= 0.04 facecam cap) on every frame
# -> exactly ONE persistent track -> auto layout = face_track.
# (`t` is required by the contract; the fixture ignores it.)
_FACE = {"t": 0.0, "x": 0.35, "y": 0.35, "w": 0.30, "h": 0.30, "score": 0.9}


def _probe() -> ProbeResult:
    return ProbeResult(
        duration_s=200.0, width=1920, height=1080, fps=30.0,
        vcodec="h264", acodec="aac", has_audio=True, size_bytes=1000,
        schema_version=1,
    )


def _plan(clip_id: str, start: float, end: float) -> ClipPlan:
    return ClipPlan(
        clip_id=clip_id,
        job_id=JOB_ID,
        start=start,
        end=end,
        layout=Layout.center_crop,
        crop_path=[],
        weighted_score=8.5,
        title="Title",
        hook_overlay="Hook",
        description="Desc",
        hashtags=[],
        status="pending_qc",
    )


class _RecordingVision:
    """Counts make_segment calls (fixture does the actual (stub) work)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._impl = FixtureVisionAdapter()

    @property
    def impl(self) -> FixtureVisionAdapter:
        return self._impl

    def make_segment(self, source_path, dst_path, start_s, end_s,
                     fps, width, height, clip_id="unknown"):
        self.calls.append((clip_id, dst_path))
        return self._impl.make_segment(
            source_path, dst_path, start_s, end_s, fps, width, height,
            clip_id=clip_id,
        )


class _ClosingDetector:
    """Wraps a FixtureFaceDetector and records close() calls."""

    def __init__(self, fixture: FixtureFaceDetector) -> None:
        self._d = fixture
        self.close_calls = 0

    def detect(self, frame, t):
        return self._d.detect(frame, t)

    def name(self) -> str:
        return self._d.name()

    def close(self) -> None:
        self.close_calls += 1
        self._d.close()


def _frame_reader(n_frames: int = N_FRAMES):
    """CPU-only fake decoder: yields N (frame, t) pairs, t = i/fps."""

    def reader(path: str, fps: float):
        return iter(("FRAME", i / fps) for i in range(n_frames))

    return reader


def _setup(
    tmp_path,
    plans: list[ClipPlan] | None = None,
    with_probe: bool = True,
    with_plans: bool = True,
) -> Config:
    """Temp data dir + DB job + required artifacts on disk."""
    if plans is None:
        plans = [_plan(f"{JOB_ID}-clip01", START, END)]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(data_dir)))
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")

    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    if with_probe:
        (job_dir / "probe.json").write_text(
            _probe().model_dump_json(indent=2) + "\n", encoding="utf-8",
        )
    (job_dir / "shots.json").write_text(
        ShotsArtifact(shots=[Shot(start=0.0, end=200.0)]).model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    if with_plans:
        (job_dir / "clip_plans.json").write_text(
            ClipPlansArtifact(plans=plans).model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    return cfg


def _run(cfg: Config, **kwargs) -> dict:
    conn = Database(cfg).get_connection()
    try:
        return run_layout(JOB_ID, cfg, conn, **kwargs)
    finally:
        conn.close()


def _detections_all_frames(face: dict, n: int = N_FRAMES) -> dict[int, list]:
    return {i: [dict(face)] for i in range(n)}


def _job_dir(cfg: Config):
    return cfg.resolved_data_dir / "jobs" / JOB_ID


def _clip_row(cfg: Config, clip_id: str) -> dict:
    conn = Database(cfg).get_connection()
    try:
        row = conn.execute(
            "SELECT plan_json FROM clips WHERE clip_id = ?", (clip_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "clips row missing"
    return json.loads(row[0])


def test_writes_layout_artifact_and_updates_plans(tmp_path):
    cfg = _setup(tmp_path)
    vision = _RecordingVision()
    detector = _ClosingDetector(FixtureFaceDetector(
        detections=_detections_all_frames(_FACE)))
    summary = _run(
        cfg, vision=vision, detector=detector,
        frame_reader=_frame_reader(),
    )

    assert summary["skipped"] is False
    rec = summary["clips"][0]
    assert rec["clip_id"] == f"{JOB_ID}-clip01"
    assert rec["layout"] == "face_track"
    assert rec["face_track_count"] == 1
    assert rec["detector"] == "fixture-face-detector"
    assert rec["keyframe_count"] == 121
    assert rec["layout_source"] == "auto"

    # per-clip artifact (schema_version=1, canonical content)
    job_dir = _job_dir(cfg)
    art_path = job_dir / "clips" / f"{JOB_ID}-clip01" / "layout.json"
    assert art_path.is_file()
    art = ClipLayoutArtifact.model_validate(
        json.loads(art_path.read_text(encoding="utf-8"))
    )
    assert art.schema_version == 1
    assert art.layout is Layout.face_track
    assert (art.source_width, art.source_height) == (1920, 1080)
    assert art.native_fps == 30.0
    assert art.tracks[0].role == "main"
    assert art.tracks[0].keys[0].t == 0.0

    # clip_plans.json rewritten in full, layout + crop_path filled
    plans = json.loads(
        (job_dir / "clip_plans.json").read_text(encoding="utf-8")
    )["plans"]
    assert len(plans) == 1
    assert plans[0]["layout"] == "face_track"
    assert len(plans[0]["crop_path"]) == 121
    assert plans[0]["crop_path"][0]["t"] == 0.0
    # every other field intact
    assert plans[0]["start"] == START and plans[0]["end"] == END
    assert plans[0]["status"] == "pending_qc"
    assert plans[0]["weighted_score"] == 8.5

    # clips row upserted with the same update
    row = _clip_row(cfg, f"{JOB_ID}-clip01")
    assert row["layout"] == "face_track"
    assert len(row["crop_path"]) == 121
    assert row["status"] == "pending_qc"

    # DB bookkeeping: stage done, job back to queued
    conn = Database(cfg).get_connection()
    try:
        state = conn.execute(
            "SELECT state FROM stages WHERE job_id=? AND name='layout'",
            (JOB_ID,),
        ).fetchone()
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
    finally:
        conn.close()
    assert state[0] == "done"
    assert job_state[0] == "queued"

    assert detector.close_calls == 1


def test_no_faces_center_crop_static(tmp_path):
    cfg = _setup(tmp_path)
    vision = _RecordingVision()
    detector = _ClosingDetector(FixtureFaceDetector(detections={}))
    summary = _run(
        cfg, vision=vision, detector=detector,
        frame_reader=_frame_reader(),
    )
    assert summary["clips"][0]["layout"] == "center_crop"
    assert summary["clips"][0]["face_track_count"] == 0

    art = ClipLayoutArtifact.model_validate(
        json.loads(
            (
                _job_dir(cfg) / "clips" / f"{JOB_ID}-clip01" / "layout.json"
            ).read_text(encoding="utf-8")
        )
    )
    keys = art.tracks[0].keys
    assert len(keys) == 121
    # static: every key has IDENTICAL geometry (only t advances along grid)
    geom = (keys[0].x, keys[0].y, keys[0].w, keys[0].h)
    assert all((k.x, k.y, k.w, k.h) == geom for k in keys)
    k0 = keys[0]
    assert k0.y + k0.h <= 1080 and k0.x + k0.w <= 1920
    # exact 9:16 aspect, fitted + centered in the source (M6d geometry)
    assert k0.h * 9 == k0.w * 16
    assert abs(k0.x - (1920 - k0.w) / 2) <= 1
    assert abs(k0.y - (1080 - k0.h) / 2) <= 1


def test_forced_layout_marks_manual(tmp_path):
    # Faces are present (auto would choose face_track), but the layout is
    # forced to center_crop -> recorded as layout_source="manual".
    cfg = _setup(tmp_path)
    vision = _RecordingVision()
    detector = _ClosingDetector(FixtureFaceDetector(
        detections=_detections_all_frames(_FACE)))
    summary = _run(
        cfg, vision=vision, detector=detector,
        frame_reader=_frame_reader(), forced_layout=Layout.center_crop,
    )
    assert summary["clips"][0]["layout"] == "center_crop"
    art = ClipLayoutArtifact.model_validate(
        json.loads(
            (
                _job_dir(cfg) / "clips" / f"{JOB_ID}-clip01" / "layout.json"
            ).read_text(encoding="utf-8")
        )
    )
    assert art.layout is Layout.center_crop
    assert art.layout_source == "manual"


def test_clip_filter_limits_work(tmp_path):
    plans = [
        _plan(f"{JOB_ID}-clip01", START, END),
        _plan(f"{JOB_ID}-clip02", 50.0, 53.0),
    ]
    cfg = _setup(tmp_path, plans=plans)
    vision = _RecordingVision()
    detector = FixtureFaceDetector(detections=_detections_all_frames(_FACE))
    _run(
        cfg, vision=vision, detector=detector,
        frame_reader=_frame_reader(), clip_id=f"{JOB_ID}-clip02",
    )
    # only the scoped clip was rendered + written
    assert [c[0] for c in vision.calls] == [f"{JOB_ID}-clip02"]
    clips_root = _job_dir(cfg) / "clips"
    assert (clips_root / f"{JOB_ID}-clip02" / "layout.json").is_file()
    assert not (clips_root / f"{JOB_ID}-clip01" / "layout.json").exists()

    # clip_plans.json rewritten in FULL: clip02 updated, clip01 intact
    plans_data = json.loads(
        (_job_dir(cfg) / "clip_plans.json").read_text(encoding="utf-8")
    )["plans"]
    by_id = {p["clip_id"]: p for p in plans_data}
    assert by_id[f"{JOB_ID}-clip02"]["layout"] == "face_track"
    assert by_id[f"{JOB_ID}-clip01"]["layout"] == "center_crop"
    assert by_id[f"{JOB_ID}-clip01"]["crop_path"] == []

    with pytest.raises(StageError, match="not found in clip_plans.json"):
        _run(
            cfg, vision=vision, detector=detector,
            frame_reader=_frame_reader(), clip_id=f"{JOB_ID}-clip99",
        )


def test_second_run_skips_completed_stage(tmp_path):
    cfg = _setup(tmp_path)
    vision = _RecordingVision()
    detector = _ClosingDetector(FixtureFaceDetector(
        detections=_detections_all_frames(_FACE)))
    first = _run(cfg, vision=vision, detector=detector,
                 frame_reader=_frame_reader())
    assert first["skipped"] is False
    assert len(vision.calls) == 1

    second = _run(cfg, vision=vision, detector=detector,
                  frame_reader=_frame_reader())
    assert second["skipped"] is True
    assert len(vision.calls) == 1  # NO re-render on skip
    # skip summary still reports the per-clip record
    assert second["clips"][0]["clip_id"] == f"{JOB_ID}-clip01"
    assert second["clips"][0]["layout"] == "face_track"


def test_detector_close_called_on_success_and_exception(tmp_path):
    det_ok = _ClosingDetector(FixtureFaceDetector(
        detections=_detections_all_frames(_FACE)))
    cfg = _setup(tmp_path)
    _run(cfg, vision=_RecordingVision(), detector=det_ok,
         frame_reader=_frame_reader())
    assert det_ok.close_calls == 1

    # exception path (FRESH job, since the first run already completed the
    # stage and would be skipped): no faces + forced face_track -> build
    # fails cleanly, detector still closed
    cfg_bad = _setup(tmp_path / "x")
    det_bad = _ClosingDetector(FixtureFaceDetector(detections={}))
    with pytest.raises(StageError):
        _run(cfg_bad, vision=_RecordingVision(), detector=det_bad,
             frame_reader=_frame_reader(),
             forced_layout=Layout.face_track)
    assert det_bad.close_calls == 1  # closed DESPITE the failure


def test_missing_inputs_fail_cleanly(tmp_path):
    # (a) missing clip_plans.json
    cfg = _setup(tmp_path, with_plans=False)
    detector = _ClosingDetector(FixtureFaceDetector(detections={}))
    with pytest.raises(StageError, match="clip_plans.json"):
        _run(cfg, vision=_RecordingVision(), detector=detector,
             frame_reader=_frame_reader())
    assert detector.close_calls == 1

    # (b) missing probe.json -> clean error + job/stage marked failed
    cfg2 = _setup(tmp_path / "b", with_probe=False)
    detector2 = _ClosingDetector(FixtureFaceDetector(detections={}))
    with pytest.raises(StageError, match="probe.json"):
        _run(cfg2, vision=_RecordingVision(), detector=detector2,
             frame_reader=_frame_reader())

    conn = Database(cfg2).get_connection()
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
        stage_state = conn.execute(
            "SELECT state FROM stages WHERE job_id=? AND name='layout'",
            (JOB_ID,),
        ).fetchone()
    finally:
        conn.close()
    assert job_state[0] == "failed"
    assert stage_state[0] == "failed"
    assert detector2.close_calls == 1


def test_zero_clips_succeeds_without_detector_work(tmp_path):
    cfg = _setup(tmp_path, plans=[])

    def _explode(*a, **k):  # pragma: no cover — must never be reached
        raise AssertionError("detector must NOT be invoked for zero clips")

    class _NoWorkDetector:
        def detect(self, frame, t):
            _explode()

        def name(self) -> str:
            return "no-work"

        def close(self) -> None:
            return None

    detector = _NoWorkDetector()
    summary = _run(
        cfg, vision=_RecordingVision(), detector=detector,
        frame_reader=_frame_reader(),
    )
    assert summary["skipped"] is False
    assert summary["clips"] == []

    conn = Database(cfg).get_connection()
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
        stage_state = conn.execute(
            "SELECT state FROM stages WHERE job_id=? AND name='layout'",
            (JOB_ID,),
        ).fetchone()
    finally:
        conn.close()
    assert job_state[0] == "queued"
    assert stage_state[0] == "done"


def test_crop_path_starts_at_zero_clip_relative(tmp_path):
    cfg = _setup(tmp_path)
    vision = _RecordingVision()
    detector = _ClosingDetector(FixtureFaceDetector(
        detections=_detections_all_frames(_FACE)))
    _run(cfg, vision=vision, detector=detector, frame_reader=_frame_reader())

    plans = json.loads(
        (_job_dir(cfg) / "clip_plans.json").read_text(encoding="utf-8")
    )["plans"]
    keys = plans[0]["crop_path"]
    assert keys[0]["t"] == 0.0
    duration = END - START
    for k in keys:
        assert 0.0 <= k["t"] <= duration
        assert isinstance(k["x"], int) and 0 <= k["x"] <= 1920
        assert isinstance(k["y"], int) and 0 <= k["y"] <= 1080
    assert abs(keys[-1]["t"] - duration) < 1e-9


def test_determinism_with_fixture_detector(tmp_path):
    def _one_run(base: object) -> tuple[str, str]:
        cfg = _setup(base)
        vision = _RecordingVision()
        detector = _ClosingDetector(FixtureFaceDetector(
            detections=_detections_all_frames(_FACE)))
        _run(cfg, vision=vision, detector=detector,
             frame_reader=_frame_reader())
        job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
        layout = (
            job_dir / "clips" / f"{JOB_ID}-clip01" / "layout.json"
        ).read_bytes()
        plans = (job_dir / "clip_plans.json").read_bytes()
        return layout, plans

    first = _one_run(tmp_path / "run1")
    second = _one_run(tmp_path / "run2")
    assert first[0] == second[0]  # byte-identical layout.json
    assert first[1] == second[1]  # byte-identical clip_plans.json
