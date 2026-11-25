"""
M7d render-stage tests — CPU only (FixtureRenderAdapter + temp DB; NO
GPU / real ffmpeg / fonts / network).

Covers (stage level): per-clip ``mezzanine.mp4`` / ``audio.m4a`` /
``captions.ass`` / ``final.mp4`` / ``thumbnail.jpg`` / ``render.json``
creation; ``clip_plans.json`` status + ``clips`` row updates;
``center_crop`` static path with ``faces == 0``; ``--clip`` limits work
to ONE clip; second run skips the completed stage; missing
``layout.json`` / ``words.json`` fail cleanly; zero clips succeeds
WITHOUT any render work; ``cpu_render=True`` logs a LOUD warning;
determinism with the fixture adapter (byte-identical artifacts).
"""

import json
import os
from pathlib import Path

import pytest

from clipper.adapters.render import FixtureRenderAdapter
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.models import (
    ClipPlan,
    ClipPlansArtifact,
    CropKey,
    Layout,
    ProbeResult,
    RenderArtifact,
    Word,
    WordsArtifact,
)
from clipper.stages import StageError
from clipper.stages.render import run_render

JOB_ID = "job-m7d"

# Clip window [10.0, 14.0) -> 4s clip.
START, END = 10.0, 14.0


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
        hook_overlay="Hook text",
        description="Desc",
        hashtags=[],
        status="pending_qc",
    )


def _words() -> list[Word]:
    """Words overlapping (and not overlapping) the clip window."""
    return [
        Word(start=8.0, end=9.5, text="before", prob=0.9),   # outside
        Word(start=9.5, end=10.5, text="honest,", prob=0.92),
        Word(start=10.5, end=11.2, text="I", prob=0.95),
        Word(start=11.2, end=12.1, text="don't", prob=0.9),
        Word(start=12.1, end=13.2, text="care", prob=0.88),
        Word(start=13.2, end=13.9, text="about", prob=0.91),
        Word(start=14.0, end=15.5, text="after", prob=0.9),   # outside
    ]


def _words_artifact() -> WordsArtifact:
    return WordsArtifact(schema_version=1, words=_words())


def _layout_artifact(
    clip_id: str,
    layout: str = "center_crop",
    face_track_count: int = 0,
    with_tracks: bool = True,
) -> dict:
    """A VALID M6e layout artifact (schema_version=1). Variants:
    `center_crop` (static exact-9:16 main), `face_track` (moving
    full-height window) and the M7f multi-region `split_screen`
    (`top`+`bottom`) / `facecam` (`main`+`face`). `with_tracks=False`
    omits the `tracks` list entirely (adapter-level failure case)."""
    x0 = (1920 - 608) // 2
    if layout == "face_track":
        keys = [
            CropKey(t=i / 30.0, x=x0 + i, y=0, w=608, h=1080)
            for i in range(25)
        ]
        tracks = [{"role": "main", "keys": [k.model_dump() for k in keys]}]
    elif layout == "split_screen":
        half = 1920 // 2  # left/right halves of the 1920x1080 source
        top = CropKey(t=0.0, x=96, y=0, w=half, h=1080)
        bottom = CropKey(t=0.0, x=960 + 96, y=0, w=half, h=1080)
        tracks = [
            {"role": "top", "keys": [top.model_dump()]},
            {"role": "bottom", "keys": [bottom.model_dump()]},
        ]
    elif layout == "facecam":
        mkeys = [CropKey(t=0.0, x=(1920 - 606) // 2, y=0, w=606, h=1080)]
        fkeys = [CropKey(t=0.0, x=100, y=60, w=200, h=200)]
        tracks = [
            {"role": "main", "keys": [k.model_dump() for k in mkeys]},
            {"role": "face", "keys": [k.model_dump() for k in fkeys]},
        ]
    else:  # exact 9:16 fitted into 1920x1080: h=1080 -> w=607.5 -> even 606
        keys = [
            CropKey(t=round(i * (END - START) / 30.0, 3), x=(1920 - 606) // 2,
                    y=0, w=606, h=1080)
            for i in range(25)
        ]
        tracks = [{"role": "main", "keys": [k.model_dump() for k in keys]}]
    artifact = {
        "schema_version": 1,
        "clip_id": clip_id,
        "layout": layout,
        "source_width": 1920,
        "source_height": 1080,
        "native_fps": 30.0,
        "face_track_count": face_track_count,
        "detector": "fixture-face-detector",
        "layout_source": "auto",
    }
    if with_tracks:
        artifact["tracks"] = tracks
    return artifact


class _RecordingRender:
    """Counts per-clip render work (fixture does the placeholder work)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.impl = FixtureRenderAdapter()

    def render_mezzanine(self, source, plan, layout, dst, cfg,
                         native_fps=None, tracks=None):
        self.calls.append(("mezzanine", plan.clip_id))
        return self.impl.render_mezzanine(
            source, plan, layout, dst, cfg,
            native_fps=native_fps, tracks=tracks)

    def extract_audio(self, source, start_s, end_s, dst):
        self.calls.append(("audio", dst))
        return self.impl.extract_audio(source, start_s, end_s, dst)

    def compose_final(self, mezzanine, audio, ass_path, dst, cfg):
        self.calls.append(("compose", dst))
        return self.impl.compose_final(
            mezzanine, audio, ass_path, dst, cfg)

    def make_thumbnail(self, final, dst):
        self.calls.append(("thumbnail", dst))
        return self.impl.make_thumbnail(final, dst)


def _setup(
    tmp_path,
    plans: list[ClipPlan] | None = None,
    with_probe: bool = True,
    with_plans: bool = True,
    with_words: bool = True,
    with_layouts: bool = True,
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
    if with_words:
        (job_dir / "words.json").write_text(
            _words_artifact().model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    if with_plans:
        (job_dir / "clip_plans.json").write_text(
            ClipPlansArtifact(plans=plans).model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    if with_layouts:
        for p in plans:
            clip_dir = job_dir / "clips" / p.clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            (clip_dir / "layout.json").write_text(
                json.dumps(_layout_artifact(p.clip_id), indent=2) + "\n",
                encoding="utf-8",
            )
    return cfg


def _setup_face_track(tmp_path) -> tuple[Config, ClipPlan]:
    """One face_track clip: M6e-style layout.json + a ClipPlan whose
    `crop_path` holds the main-track keys (t=0 first, M6e rule)."""
    clip_id = f"{JOB_ID}-clip01"
    plan = _plan(clip_id, START, END)
    plan.layout = Layout.face_track
    keys = [
        CropKey(t=i / 30.0, x=(1920 - 608) // 2 + i, y=0, w=608, h=1080)
        for i in range(25)
    ]
    plan.crop_path = keys
    cfg = _setup(tmp_path, plans=[plan], with_layouts=False)
    clip_dir = _job_dir(cfg) / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    artifact = _layout_artifact(clip_id, layout="face_track",
                                face_track_count=1)
    (clip_dir / "layout.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8",
    )
    return cfg, plan


def _run(cfg: Config, render=None, **kwargs) -> dict:
    if render is None:
        render = _RecordingRender()
    conn = Database(cfg).get_connection()
    try:
        return run_render(JOB_ID, cfg, conn, render, **kwargs)
    finally:
        conn.close()


def _job_dir(cfg: Config) -> "object":
    return cfg.resolved_data_dir / "jobs" / JOB_ID


def _clip_row(cfg: Config, clip_id: str) -> dict:
    conn = Database(cfg).get_connection()
    try:
        row = conn.execute(
            "SELECT plan_json, status FROM clips WHERE clip_id = ?",
            (clip_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "clips row missing"
    return {"plan": json.loads(row[0]), "status": row[1]}


def _stage_and_job_state(cfg: Config) -> tuple[str, str]:
    conn = Database(cfg).get_connection()
    try:
        stage = conn.execute(
            "SELECT state FROM stages WHERE job_id=? AND name='render'",
            (JOB_ID,),
        ).fetchone()
        job = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
    finally:
        conn.close()
    return str(stage[0]), str(job[0])


def _clip_files(cfg: Config, clip_id: str) -> set[str]:
    clip_dir = _job_dir(cfg) / "clips" / clip_id
    return {p.name for p in clip_dir.iterdir()} if clip_dir.is_dir() else set()


def test_renders_all_outputs_center_crop_faces_zero(tmp_path):
    cfg = _setup(tmp_path)
    render = _RecordingRender()
    summary = _run(cfg, render)
    cid = f"{JOB_ID}-clip01"

    assert summary["skipped"] is False
    rec = summary["clips"][0]
    assert rec["clip_id"] == cid
    assert rec["layout"] == "center_crop"
    assert rec["face_track_count"] == 0  # faces == 0 path (decision 8)

    # ALL SIX per-clip outputs exist (decision 4; + layout.json M6e input)
    files = _clip_files(cfg, cid)
    assert files >= {
        "mezzanine.mp4", "audio.m4a", "captions.ass",
        "final.mp4", "thumbnail.jpg", "render.json",
    }

    # render.json is a VALID RenderArtifact with clip_id + thumbnail filled
    art = RenderArtifact.model_validate_json(
        (_job_dir(cfg) / "clips" / cid / "render.json"
         ).read_text(encoding="utf-8")
    )
    assert art.schema_version == 1
    assert art.clip_id == cid
    assert art.encoder == "h264_nvenc"
    assert art.hwaccel == "cuda"
    assert (art.width, art.height) == (1080, 1920)  # cfg render geometry
    assert art.duration_s == pytest.approx(15.0)  # fixture compose_final
    assert art.render_seconds >= 0.0
    assert art.mezzanine_path.endswith("mezzanine.mp4")
    assert art.audio_path.endswith("audio.m4a")
    assert art.captions_path.endswith("captions.ass")
    assert art.final_path.endswith("final.mp4")
    assert art.thumbnail_path.endswith("thumbnail.jpg")
    for label, path in (
        ("mezzanine", art.mezzanine_path),
        ("audio", art.audio_path),
        ("captions", art.captions_path),
        ("final", art.final_path),
        ("thumbnail", art.thumbnail_path),
    ):
        assert os.path.exists(path), f"{label} file missing"

    # captions.ass: deterministic M7a doc, clip content, karaoke + hook
    ass = (_job_dir(cfg) / "clips" / cid / "captions.ass"
           ).read_text(encoding="utf-8")
    assert "[Script Info]" in ass and "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "honest," in ass
    assert "I" in ass and "don't" in ass and "care" in ass and "about" in ass
    assert "\\kf" in ass and "&H00FFFF&" in ass and "&HFFFFFF&" in ass
    assert "Hook text" in ass
    # pre-cut and post-cut words excluded
    assert "before" not in ass
    assert "after" not in ass

    # clip_plans.json rewritten (IN FULL) with status=rendered
    plans = json.loads(
        (_job_dir(cfg) / "clip_plans.json").read_text(encoding="utf-8")
    )["plans"]
    assert plans[0]["clip_id"] == cid
    assert plans[0]["status"] == "rendered"
    # every other field intact
    assert plans[0]["start"] == START and plans[0]["end"] == END
    assert plans[0]["layout"] == "center_crop"
    assert plans[0]["weighted_score"] == 8.5

    # clips row upserted with the same status
    row = _clip_row(cfg, cid)
    assert row["status"] == "rendered"
    assert row["plan"]["status"] == "rendered"

    # DB bookkeeping: stage done, job back to queued
    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "done"
    assert job_state == "queued"

    # adapter was used exactly once per step, in order
    assert [c[0] for c in render.calls] == [
        "mezzanine", "audio", "compose", "thumbnail",
    ]


def test_clip_filter_limits_work(tmp_path):
    plans = [
        _plan(f"{JOB_ID}-clip01", START, END),
        _plan(f"{JOB_ID}-clip02", 50.0, 53.0),
    ]
    cfg = _setup(tmp_path, plans=plans)
    render = _RecordingRender()
    _run(cfg, render, clip_id=f"{JOB_ID}-clip02")

    # ONLY clip02 was rendered
    assert [c[1] for c in render.calls if c[0] == "mezzanine"] == [
        f"{JOB_ID}-clip02"
    ]
    assert "layout.json" in _clip_files(cfg, f"{JOB_ID}-clip01")  # M6e input
    assert "final.mp4" not in _clip_files(cfg, f"{JOB_ID}-clip01")
    assert "final.mp4" in _clip_files(cfg, f"{JOB_ID}-clip02")

    # clip_plans.json rewritten in FULL: clip02 rendered, clip01 intact
    plans_data = {
        p["clip_id"]: p
        for p in json.loads(
            (_job_dir(cfg) / "clip_plans.json").read_text(encoding="utf-8")
        )["plans"]
    }
    assert plans_data[f"{JOB_ID}-clip02"]["status"] == "rendered"
    assert plans_data[f"{JOB_ID}-clip01"]["status"] == "pending_qc"

    # unknown clip fails cleanly
    with pytest.raises(StageError, match="not found in clip_plans.json"):
        _run(cfg, _RecordingRender(), clip_id=f"{JOB_ID}-clip99")


def test_second_run_skips_completed_stage(tmp_path):
    cfg = _setup(tmp_path)
    render = _RecordingRender()
    first = _run(cfg, render)
    assert first["skipped"] is False
    assert len([c for c in render.calls if c[0] == "mezzanine"]) == 1

    second = _run(cfg, render)
    assert second["skipped"] is True
    # NO re-render on skip
    assert len([c for c in render.calls if c[0] == "mezzanine"]) == 1
    # skip summary still reports the per-clip record (from render.json)
    rec = second["clips"][0]
    assert rec["clip_id"] == f"{JOB_ID}-clip01"
    assert rec["final_path"].endswith("final.mp4")
    assert rec["encoder"] == "h264_nvenc"

    # stage still done, job queued
    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "done"
    assert job_state == "queued"


def test_skip_requires_final_and_valid_render_json(tmp_path):
    cfg = _setup(tmp_path)
    _run(cfg)
    cid = f"{JOB_ID}-clip01"
    clip_dir = _job_dir(cfg) / "clips" / cid

    # corrupt render.json -> NO skip, re-render happens
    (clip_dir / "render.json").write_text('{"schema_version": 1}\n')
    render = _RecordingRender()
    again = _run(cfg, render)
    assert again["skipped"] is False
    assert len([c for c in render.calls if c[0] == "mezzanine"]) == 1


def test_missing_layout_json_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, with_layouts=False)
    with pytest.raises(StageError, match=r"layout\.json"):
        _run(cfg)

    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "failed"
    assert job_state == "failed"
    # NO render work happened (no per-clip outputs)
    files = _clip_files(cfg, f"{JOB_ID}-clip01")
    assert "final.mp4" not in files and "render.json" not in files


def test_missing_words_json_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, with_words=False)
    render = _RecordingRender()
    with pytest.raises(StageError, match=r"words\.json"):
        _run(cfg, render)

    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "failed"
    assert job_state == "failed"
    # nothing rendered (layout.json is the M6e input, not render output)
    files = _clip_files(cfg, f"{JOB_ID}-clip01")
    assert "final.mp4" not in files and "render.json" not in files


def test_missing_clip_plans_or_probe_fail_cleanly(tmp_path):
    cfg = _setup(tmp_path, with_plans=False)
    with pytest.raises(StageError, match=r"clip_plans\.json"):
        _run(cfg)

    cfg2 = _setup(tmp_path / "b", with_probe=False)
    with pytest.raises(StageError, match=r"probe\.json"):
        _run(cfg2)


def test_zero_clips_succeeds_without_render_work(tmp_path):
    cfg = _setup(tmp_path, plans=[])

    class _ExplodingRender:
        def _explode(self, *a, **k):
            raise AssertionError("render adapter must NOT be invoked")

        render_mezzanine = _explode
        extract_audio = _explode
        compose_final = _explode
        make_thumbnail = _explode

    render = _ExplodingRender()
    summary = _run(cfg, render)
    assert summary["skipped"] is False
    assert summary["clips"] == []

    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "done"
    assert job_state == "queued"

    # and the SECOND run skips (done + all zero renders valid)
    again = _run(cfg, render)
    assert again["skipped"] is True


def test_cpu_render_logs_loud_warning(tmp_path, caplog):
    cfg = _setup(tmp_path)
    render = _RecordingRender()
    with caplog.at_level("WARNING", logger="clipper.stages.render"):
        _run(cfg, render, cpu_render=True)
    warnings = [
        r.message for r in caplog.records
        if r.levelno >= 30 and "cpu_render" in r.message
    ]
    assert warnings, "cpu_render=True must log a LOUD warning"
    assert any("DEBUG" in w for w in warnings)


def test_no_warning_without_cpu_render(tmp_path, caplog):
    cfg = _setup(tmp_path)
    with caplog.at_level("WARNING", logger="clipper.stages.render"):
        _run(cfg)
    assert not [
        r for r in caplog.records if "cpu_render" in r.message
    ]


def test_face_track_layout_renders_cleanly(tmp_path):
    """M7e: the dynamic face_track path renders end-to-end (stage level)
    on the CPU-only fixture adapter: crop.cmd lands BESIDE the mezzanine,
    all six outputs exist, status updated."""
    cfg, plan = _setup_face_track(tmp_path)
    render = _RecordingRender()
    summary = _run(cfg, render)
    cid = f"{JOB_ID}-clip01"

    assert summary["skipped"] is False
    rec = summary["clips"][0]
    assert rec["layout"] == "face_track"
    assert rec["face_track_count"] == 1

    clip_dir = _job_dir(cfg) / "clips" / cid
    assert (clip_dir / "crop.cmd").is_file()  # write beside crop.cmd
    assert (clip_dir / "mezzanine.mp4").is_file()
    crop_cmd_lines = (
        (clip_dir / "crop.cmd").read_text(encoding="utf-8"
                                          ).splitlines())
    assert crop_cmd_lines[0] == f"0 crop x {plan.crop_path[0].x}, crop y 0;"

    files = _clip_files(cfg, cid)
    assert files >= {
        "mezzanine.mp4", "audio.m4a", "captions.ass",
        "final.mp4", "thumbnail.jpg", "render.json",
    }
    art = RenderArtifact.model_validate_json(
        (clip_dir / "render.json").read_text(encoding="utf-8")
    )
    assert art.clip_id == cid
    assert art.encoder == "h264_nvenc"

    row = _clip_row(cfg, cid)
    assert row["status"] == "rendered"
    assert row["plan"]["layout"] == "face_track"
    assert [k["t"] for k in row["plan"]["crop_path"]][0] == 0.0
    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "done"
    assert job_state == "queued"


def test_layout_without_region_tracks_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path)
    artifact = _layout_artifact(f"{JOB_ID}-clip01")
    artifact["layout"] = "split_screen"  # tracks still only `main`
    (
        _job_dir(cfg) / "clips" / f"{JOB_ID}-clip01" / "layout.json"
    ).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(StageError, match=r"split_screen"):
        _run(cfg)
    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "failed"
    assert job_state == "failed"


@pytest.mark.parametrize("layout", ["split_screen", "facecam"])
def test_multiregion_layout_renders_with_region_tracks(tmp_path, layout):
    """M7f stage level: split_screen/facecam layout.json (M6e
    `RegionTrack`s) render end-to-end (fixture adapter) — the stage
    forwards `layout_art.tracks` to the adapter."""
    cfg = _setup(tmp_path)
    clip_id = f"{JOB_ID}-clip01"
    artifact = _layout_artifact(clip_id, layout=layout)
    (
        _job_dir(cfg) / "clips" / clip_id / "layout.json"
    ).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    summary = _run(cfg)

    assert summary["skipped"] is False
    assert summary["clips"][0]["clip_id"] == clip_id
    files = _clip_files(cfg, clip_id)
    for expected in (
            "mezzanine.mp4", "audio.m4a", "captions.ass", "final.mp4",
            "thumbnail.jpg", "render.json"):
        assert expected in files


def test_render_error_marks_failed_and_job_failed(tmp_path):
    cfg = _setup(tmp_path)

    class _BrokenRender:
        def render_mezzanine(self, *a, **k):
            raise RuntimeError("nvenc exploded")

        def extract_audio(self, *a, **k):
            raise AssertionError("must not be reached")

        def compose_final(self, *a, **k):
            raise AssertionError("must not be reached")

        def make_thumbnail(self, *a, **k):
            raise AssertionError("must not be reached")

    with pytest.raises(StageError, match="nvenc exploded"):
        _run(cfg, _BrokenRender())

    stage_state, job_state = _stage_and_job_state(cfg)
    assert stage_state == "failed"
    assert job_state == "failed"

    # a retry now SUCCEEDS (stage reset, job re-runnable)
    ok = _run(cfg)
    assert ok["skipped"] is False
    assert "final.mp4" in _clip_files(cfg, f"{JOB_ID}-clip01")


def test_determinism_with_fixture_adapter(tmp_path):
    def _normalized(path: object, job_dir: object) -> bytes:
        # paths are absolute (machine-specific); normalize the data dir
        # before comparing — everything else must be byte-identical
        raw = Path(path).read_bytes().decode("utf-8")
        return raw.replace(str(job_dir), "<job_dir>").encode("utf-8")

    def _one_run(base: object) -> tuple[bytes, bytes, bytes]:
        cfg = _setup(base)
        _run(cfg)
        job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
        clip_dir = job_dir / "clips" / f"{JOB_ID}-clip01"
        return (
            _normalized(clip_dir / "captions.ass", job_dir),
            _normalized(clip_dir / "render.json", job_dir),
            _normalized(job_dir / "clip_plans.json", job_dir),
        )

    first = _one_run(tmp_path / "run1")
    second = _one_run(tmp_path / "run2")
    assert first[0] == second[0]  # identical captions.ass
    assert first[1] == second[1]  # identical render.json
    assert first[2] == second[2]  # identical clip_plans.json
