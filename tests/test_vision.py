"""
M6a vision-segment tests — CPU only, subprocess mocked (docs/spec/m6.md).

Covers: contract round-trips, fps selection rule, exact ffmpeg command
(NVDEC + NVENC, no padding), probe-based VisionSegmentInfo, failure
behavior (MediaError, never silent CPU fallback), the fixture adapter,
the factory, and the new backward-compatible config keys.
"""

import subprocess

import pytest
from pydantic import ValidationError

import clipper.adapters.vision as vision_mod
from clipper.adapters.media import MediaError
from clipper.adapters.vision import (
    FFmpegVisionAdapter,
    FixtureVisionAdapter,
    get_vision_adapter,
    select_vision_fps,
)
from clipper.config import Config, DevicesConfig, FacesConfig
from clipper.models import (
    ClipLayoutArtifact,
    CropKey,
    FaceDetection,
    Layout,
    ProbeResult,
    RegionTrack,
    VisionSegmentInfo,
)

CLIP_ID = "job-x-clip01"
SRC = "/tmp/source.mp4"
DST = "/tmp/out/clips/clip01/vision.mp4"
START, END = 12.0, 39.0


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_DEVICES_VISION", raising=False)
    monkeypatch.delenv("CLIPPER_FACES_VISION_FPS_CAP", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_probe(width=960, height=540, fps=30.0):
    return ProbeResult(
        duration_s=END - START,
        width=width,
        height=height,
        fps=fps,
        vcodec="h264",
        acodec=None,
        has_audio=False,
        size_bytes=12345,
    )


def _fake_media_result(width=960, height=540, fps=30.0, nb_frames="810"):
    class _FakeMedia:
        def probe(self, path):
            return _fake_probe(width=width, height=height, fps=fps)

        def _run_ffprobe(self, cmd, timeout=60):
            if nb_frames is None:
                return {"streams": [{"nb_frames": None}]}
            return {"streams": [{"nb_frames": nb_frames}]}

    return _FakeMedia()


def _fake_run_factory(monkeypatch, returncode=0, stdout=b"", stderr=b""):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr=stderr
        )

    monkeypatch.setattr(vision_mod.subprocess, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_vision_segment_info_round_trip():
    info = VisionSegmentInfo(
        clip_id=CLIP_ID,
        path="/data/jobs/job-x/clips/clip01/vision.mp4",
        width=960,
        height=540,
        fps=29.97,
        frame_count=542,
        source_start=START,
        source_end=END,
    )
    back = VisionSegmentInfo.model_validate_json(info.model_dump_json())
    assert back == info
    assert back.frame_count == 542
    assert back.source_start == 12.0
    assert back.source_end == 39.0


def test_face_detection_round_trip_normalized():
    det = FaceDetection(t=0.033, x=0.1, y=0.2, w=0.3, h=0.4, score=0.93)
    back = FaceDetection.model_validate_json(det.model_dump_json())
    assert back == det
    for v in (det.x, det.y, det.w, det.h, det.score):
        assert 0.0 <= v <= 1.0


def test_face_detection_requires_score():
    with pytest.raises(ValidationError):
        FaceDetection(t=0.0, x=0.0, y=0.0, w=0.1, h=0.1)


def test_region_track_round_trip_and_roles():
    track = RegionTrack(
        role="face",
        keys=[CropKey(t=0.0, x=100, y=200, w=64, h=64)],
    )
    back = RegionTrack.model_validate_json(track.model_dump_json())
    assert back == track
    for role in ("main", "top", "bottom", "face"):
        assert RegionTrack(role=role, keys=[]).role == role
    with pytest.raises(ValidationError):
        RegionTrack(role="center", keys=[])


def test_clip_layout_artifact_round_trip():
    art = ClipLayoutArtifact(
        clip_id=CLIP_ID,
        layout=Layout.center_crop,
        source_width=1920,
        source_height=1080,
        native_fps=60.0,
        tracks=[],
        face_track_count=0,
        detector="",
        layout_source="auto",
    )
    assert art.schema_version == 1
    back = ClipLayoutArtifact.model_validate_json(art.model_dump_json())
    assert back == art
    assert back.layout is Layout.center_crop
    assert back.tracks == []


def test_clip_layout_artifact_layout_source_restricted():
    with pytest.raises(ValidationError):
        ClipLayoutArtifact(
            clip_id=CLIP_ID,
            layout=Layout.center_crop,
            source_width=1920,
            source_height=1080,
            native_fps=30.0,
            tracks=[],
            face_track_count=0,
            detector="",
            layout_source="guessed",
        )


# ---------------------------------------------------------------------------
# fps selection (M6a decision 3)
# ---------------------------------------------------------------------------


def test_fps_selection_native_above_cap():
    assert select_vision_fps(60.0, 30.0) == 30.0


def test_fps_selection_native_below_cap_kept():
    assert select_vision_fps(24.0, 30.0) == 24.0


def test_fps_selection_unknown_native_uses_cap():
    assert select_vision_fps(0.0, 30.0) == 30.0
    assert select_vision_fps(0, 30.0) == 30.0


def test_fps_selection_default_cap_matches_spec():
    assert vision_mod.VISION_FPS_MAX_CAP == 30.0
    assert select_vision_fps(0.0) == 30.0
    assert select_vision_fps(120.0) == 30.0


# ---------------------------------------------------------------------------
# Config (backward compatible new keys)
# ---------------------------------------------------------------------------


def test_config_m6a_defaults():
    assert DevicesConfig().vision == "cuda:0"
    faces = FacesConfig()
    assert faces.vision_fps_cap == 30.0
    assert faces.vision_width == 960
    assert faces.vision_height == 540
    # pre-M6a keys unchanged and coexist (backward compatible)
    assert faces.detection_fps == 3.0
    assert faces.iou_threshold == 0.5
    assert DevicesConfig().ffmpeg is None


def test_config_overrides_load():
    faces = FacesConfig(vision_fps_cap=25.0, vision_width=640, vision_height=360)
    assert select_vision_fps(0.0, faces.vision_fps_cap) == 25.0
    assert select_vision_fps(24.0, faces.vision_fps_cap) == 24.0


# ---------------------------------------------------------------------------
# ffmpeg command shape (documented exactly in docs/spec/m6.md)
# ---------------------------------------------------------------------------


def _cmd():
    return FFmpegVisionAdapter().build_command(
        SRC, DST, START, END, 30.0, 960, 540
    )


def test_command_contains_nvdec_and_nvenc():
    cmd = _cmd()
    assert "-hwaccel" in cmd and "cuda" in cmd
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    assert "-preset" in cmd and "p4" in cmd
    assert "-cq" in cmd and "23" in cmd


def test_command_has_no_padding_filter():
    cmd = _cmd()
    assert not any("pad" in part for part in cmd)


def test_command_fit_then_even_dims_and_no_audio():
    cmd = _cmd()
    vf = cmd[cmd.index("-vf") + 1]
    assert "scale=960:540:force_original_aspect_ratio=decrease" in vf
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in vf
    assert "-an" in cmd


def test_command_seeks_before_input_with_range():
    cmd = _cmd()
    i_ss, i_to, i_in = cmd.index("-ss"), cmd.index("-to"), cmd.index("-i")
    assert i_ss < i_in
    assert i_to < i_in
    assert cmd[i_ss + 1] == f"{START:.3f}"
    assert cmd[i_to + 1] == f"{END:.3f}"
    assert cmd[-1] == DST


# ---------------------------------------------------------------------------
# make_segment success — subprocess mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def src_file(tmp_path):
    """A real (dummy) source file, since make_segment checks it exists."""
    p = tmp_path / "source.mp4"
    p.write_text("fake source")
    return str(p)


def test_make_segment_success_probes_result(monkeypatch, tmp_path, src_file):
    adapter = FFmpegVisionAdapter()
    adapter._media = _fake_media_result(nb_frames="810")
    calls = _fake_run_factory(monkeypatch, returncode=0)

    dst = str(tmp_path / "clips" / "clip01" / "vision.mp4")
    info = adapter.make_segment(
        src_file, dst, START, END, 30.0, 960, 540, clip_id=CLIP_ID
    )

    assert len(calls) == 1  # one ffmpeg run, no fallback/second attempt
    assert info == VisionSegmentInfo(
        clip_id=CLIP_ID,
        path=dst,
        width=960,
        height=540,
        fps=30.0,
        frame_count=810,
        source_start=START,
        source_end=END,
    )


def test_make_segment_frame_count_fallback_when_nb_frames_missing(
    monkeypatch, tmp_path, src_file
):
    adapter = FFmpegVisionAdapter()
    adapter._media = _fake_media_result(nb_frames=None)
    _fake_run_factory(monkeypatch, returncode=0)

    dst = str(tmp_path / "vision.mp4")
    info = adapter.make_segment(src_file, dst, START, END, 30.0, 960, 540)
    assert info.frame_count == int(round((END - START) * 30.0))


def test_make_segment_clamps_fps_to_m6_cap(monkeypatch, tmp_path, src_file):
    adapter = FFmpegVisionAdapter()
    adapter._media = _fake_media_result()
    calls = _fake_run_factory(monkeypatch, returncode=0)

    dst = str(tmp_path / "vision.mp4")
    adapter.make_segment(src_file, dst, START, END, fps=60.0, width=960, height=540)
    assert "-r" in calls[0]
    assert calls[0][calls[0].index("-r") + 1] == "30"


def test_make_segment_missing_source_raises(tmp_path):
    adapter = FFmpegVisionAdapter()
    with pytest.raises(MediaError):
        adapter.make_segment(
            str(tmp_path / "nope.mp4"), str(tmp_path / "v.mp4"),
            START, END, 30.0, 960, 540,
        )


# ---------------------------------------------------------------------------
# Failure — MediaError, never silent CPU fallback
# ---------------------------------------------------------------------------


def test_nvdec_nvenc_failure_raises_media_error_with_stderr_tail(monkeypatch, src_file):
    adapter = FFmpegVisionAdapter()
    stderr = b"... cuInit failed ... No NVENC session available ..."
    calls = _fake_run_factory(monkeypatch, returncode=1, stderr=stderr)

    with pytest.raises(MediaError) as exc_info:
        adapter.make_segment(src_file, DST, START, END, 30.0, 960, 540)

    msg = str(exc_info.value)
    assert "No NVENC session available" in msg  # stderr tail surfaced
    assert len(calls) == 1  # NO silent CPU fallback / retry
    assert "libx264" not in calls[0]


def test_ffmpeg_timeout_raises_media_error(monkeypatch, src_file):
    def time_out(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1800)

    monkeypatch.setattr(vision_mod.subprocess, "run", time_out)
    adapter = FFmpegVisionAdapter()
    with pytest.raises(MediaError, match="timed out"):
        adapter.make_segment(src_file, DST, START, END, 30.0, 960, 540)


def test_ffmpeg_missing_raises_media_error(monkeypatch, src_file):
    def not_found(cmd, **kwargs):
        raise FileNotFoundError(cmd)

    monkeypatch.setattr(vision_mod.subprocess, "run", not_found)
    adapter = FFmpegVisionAdapter()
    with pytest.raises(MediaError, match="FFmpeg not found"):
        adapter.make_segment(src_file, DST, START, END, 30.0, 960, 540)


# ---------------------------------------------------------------------------
# Fixture adapter + factory
# ---------------------------------------------------------------------------


def test_fixture_adapter_returns_valid_segment_info(tmp_path):
    adapter = FixtureVisionAdapter()
    dst = str(tmp_path / "clips" / "clip01" / "vision.mp4")
    (tmp_path / "clips" / "clip01").mkdir(parents=True)

    info = adapter.make_segment(
        SRC, dst, START, END, 30.0, 960, 540, clip_id=CLIP_ID
    )
    assert (tmp_path / "clips" / "clip01" / "vision.mp4").exists()
    assert info == VisionSegmentInfo(
        clip_id=CLIP_ID,
        path=dst,
        width=960,
        height=540,
        fps=30.0,
        frame_count=int(round((END - START) * 30.0)),
        source_start=START,
        source_end=END,
    )


def test_fixture_adapter_clamps_fps(tmp_path):
    adapter = FixtureVisionAdapter()
    dst = str(tmp_path / "v.mp4")
    info = adapter.make_segment(SRC, dst, 0.0, 1.0, fps=60.0, width=960, height=540)
    assert info.fps == 30.0


def _reset_factory(monkeypatch):
    monkeypatch.setattr(vision_mod, "_vision_adapter_instance", None)
    return None


def test_factory_default_is_ffmpeg_adapter(monkeypatch):
    _reset_factory(monkeypatch)
    assert isinstance(get_vision_adapter(Config()), FFmpegVisionAdapter)


def test_factory_cuda_device_is_ffmpeg_adapter(monkeypatch):
    _reset_factory(monkeypatch)
    cfg = Config(devices=DevicesConfig(vision="cuda:0"))
    assert isinstance(get_vision_adapter(cfg), FFmpegVisionAdapter)


def test_factory_fixture_device(monkeypatch):
    _reset_factory(monkeypatch)
    cfg = Config(devices=DevicesConfig(vision="fixture"))
    assert isinstance(get_vision_adapter(cfg), FixtureVisionAdapter)


def test_factory_caches_instance(monkeypatch):
    _reset_factory(monkeypatch)
    first = get_vision_adapter(Config())
    second = get_vision_adapter(Config())
    assert first is second


def test_adapters_implementation_satisfies_protocol():
    assert isinstance(FFmpegVisionAdapter(), vision_mod.VisionAdapter)
    assert isinstance(FixtureVisionAdapter(), vision_mod.VisionAdapter)


def test_models_reexported_from_package():
    import clipper.models as m

    assert m.VisionSegmentInfo is VisionSegmentInfo
    assert m.FaceDetection is FaceDetection
    assert m.RegionTrack is RegionTrack
    assert m.ClipLayoutArtifact is ClipLayoutArtifact
