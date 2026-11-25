"""M7b/M7c CPU-only tests: render adapter (NVDEC/NVENC mezzanine + audio
+ final compose + thumbnail).

All ffmpeg I/O is mocked via `subprocess.run` — no GPU, no real ffmpeg,
no network, no downloads (docs/spec/m7.md M7b/M7c).
"""

import subprocess
from pathlib import Path

import pytest

import clipper.adapters.render as render_mod
from clipper.adapters.media import MediaError
from clipper.adapters.render import (
    CPU_RENDER_WARNING,
    CPU_SCALE_FALLBACK_WARNING,
    FFmpegRenderAdapter,
    FixtureRenderAdapter,
    RenderAdapter,
    get_render_adapter,
    reset_render_adapter,
)
from clipper.config import Config
from clipper.models import (
    ClipAudioInfo,
    ClipPlan,
    CropKey,
    Layout,
    MezzanineInfo,
    ProbeResult,
    RenderArtifact,
)
from clipper.render_paths import CROP_CMD_FILENAME, crop_cmd_from_path

CFG = Config()

FILTERS_WITH_SCALE_CUDA = (
    " T. scale        V->V       Scale the size of the video\n"
    " ... ass         V->V       Render text subtitles on top of input\n"
    " T. scale_cuda   V->V       GPU scaler for hardware-accelerated streams\n"
    " T. hwupload_cuda  N->N     Upload to cuda\n"
)
FILTERS_WITHOUT_SCALE_CUDA = (
    " T. scale        V->V       Scale the size of the video\n"
    " ... ass         V->V       Render text subtitles on top of input\n"
)
FILTERS_NO_ASS = (  # M7c: an ffmpeg build WITHOUT the libass `ass` filter
    " T. scale        V->V       Scale the size of the video\n"
    " T. scale_cuda   V->V       GPU scaler\n"
)
# M7e: an ffmpeg build WITHOUT the `sendcmd` filter (or without `crop`)
FILTERS_NO_SENDCMD = (
    " T. scale        V->V       Scale the size of the video\n"
    " ... ass         V->V       Render text subtitles on top of input\n"
    " T. scale_cuda   V->V       GPU scaler for hardware-accelerated streams\n"
    " T. crop         V->V       Crop the input video\n"
)
FILTERS_NO_CROP = (
    " T. scale        V->V       Scale the size of the video\n"
    " ... ass         V->V       Render text subtitles on top of input\n"
    " T. scale_cuda   V->V       GPU scaler for hardware-accelerated streams\n"
    " T. sendcmd      V->N       Send commands to filters\n"
)


FILTERS_GPU_SENDCMD = (
    " T. scale        V->V       Scale the size of the video\n"
    " ... ass         V->V       Render text subtitles on top of input\n"
    " T. scale_cuda   V->V       GPU scaler for hardware-accelerated streams\n"
    " T. hwupload_cuda  N->N     Upload to cuda\n"
    " T. sendcmd      V->N       Send commands to filters\n"
    " T. crop         V->V       Crop the input video\n"
)
FILTERS_CPU_SENDCMD = (
    " T. scale        V->V       Scale the size of the video\n"
    " ... ass         V->V       Render text subtitles on top of input\n"
    " T. sendcmd      V->N       Send commands to filters\n"
    " T. crop         V->V       Crop the input video\n"
)


def make_plan(start: float = 10.0, end: float = 25.0) -> ClipPlan:
    return ClipPlan(
        clip_id="jobX-clip01",
        job_id="jobX",
        start=start,
        end=end,
        layout=Layout.center_crop,
        crop_path=[],
        weighted_score=8.0,
        title="t",
        hook_overlay="",
        description="",
        hashtags=[],
        status="pending_qc",
    )


def make_face_plan(n_keys: int = 5, fps: float = 30.0) -> ClipPlan:
    """An M7e face_track plan: source 1920x1080, full-height 9:16 window
    (w=608, h=1080) tracking a face drifting 656 -> 700 px (t=0 first)."""
    keys = [
        CropKey(
            t=round(i / fps, 6), x=656 + i * 11, y=0, w=608, h=1080,
        )
        for i in range(n_keys)
    ]
    return ClipPlan(
        clip_id="jobX-clip01",
        job_id="jobX",
        start=10.0,
        end=25.0,
        layout=Layout.face_track,
        crop_path=keys,
        weighted_score=8.0,
        title="t",
        hook_overlay="",
        description="",
        hashtags=[],
        status="pending_qc",
    )


class FakeFFmpeg:
    """Records ffmpeg invocations; serves canned -filters output."""

    def __init__(self, filters_out: str = FILTERS_WITH_SCALE_CUDA,
                 render_rc: int = 0, render_stderr: bytes = b""):
        self.calls: list[list[str]] = []
        self.filters_out = filters_out
        self.render_rc = render_rc
        self.render_stderr = render_stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if "-filters" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=self.filters_out.encode(), stderr=b"")
        return subprocess.CompletedProcess(
            cmd, self.render_rc, stdout=b"", stderr=self.render_stderr)


def install_fake(monkeypatch, fake: FakeFFmpeg) -> None:
    monkeypatch.setattr(subprocess, "run", fake)


def probe_result(width=1080, height=1920, fps=30.0, duration_s=15.02) -> ProbeResult:
    return ProbeResult(
        duration_s=duration_s, width=width, height=height, fps=fps,
        vcodec="h264", acodec=None, has_audio=False, size_bytes=1000)


@pytest.fixture(autouse=True)
def _fresh_adapter():
    reset_render_adapter()
    yield
    reset_render_adapter()


# -- mezzanine command: center_crop static path (GPU) --------------------

def test_center_crop_command_hwaccel_nvenc_no_libx264():
    cmd = FFmpegRenderAdapter().mezzanine_command(
        "src.mp4", make_plan(), Layout.center_crop, "out.mp4", CFG,
        native_fps=30.0, gpu_scale=True)
    assert cmd[0] == "ffmpeg"
    assert "-hwaccel" in cmd and cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "crop=w=trunc(ih*9/16/2)*2:h=ih:x=(iw-ow)/2:y=0" in vf
    assert "hwupload_cuda,scale_cuda=1080:1920" in vf
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "p5"
    assert "-rc" in cmd and cmd[cmd.index("-rc") + 1] == "vbr"
    assert "-cq" in cmd and cmd[cmd.index("-cq") + 1] == "16"
    assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "30"
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "10.000"
    assert "-to" in cmd and cmd[cmd.index("-to") + 1] == "25.000"
    # input BEFORE -vf (input-seek) and -an (no audio in mezzanine)
    assert cmd.index("-i") < cmd.index("-vf")
    assert "-an" in cmd
    assert cmd[-1] == "out.mp4"


def test_mezzanine_command_rejects_unknown_layouts():
    adapter = FFmpegRenderAdapter()
    for layout in ("bogus",):
        with pytest.raises(MediaError):
            adapter.mezzanine_command(
                "s", make_plan(), layout, "d", CFG, native_fps=30.0, gpu_scale=True)


def _region_tracks(layout: Layout) -> list:
    """Valid M6d-style tracks: split (top/bottom full-height 9:16) and
    facecam (main exact-9:16 + face 200x200 square), source 1920x1080."""
    from clipper.models import RegionTrack
    if layout is Layout.split_screen:
        return [
            RegionTrack(role="top", keys=[
                CropKey(t=0.0, x=152, y=0, w=608, h=1080)]),
            RegionTrack(role="bottom", keys=[
                CropKey(t=0.0, x=1160, y=0, w=608, h=1080)]),
        ]
    return [
        RegionTrack(role="main", keys=[
            CropKey(t=0.0, x=657, y=0, w=606, h=1080)]),
        RegionTrack(role="face", keys=[
            CropKey(t=0.0, x=100, y=80, w=200, h=200)]),
    ]


def test_split_screen_command_two_crops_vstack_nvenc_no_libx264():
    adapter = FFmpegRenderAdapter()
    cmd = adapter.mezzanine_command(
        "src.mp4", make_plan(), Layout.split_screen, "out/clip/mezzanine.mp4",
        CFG, native_fps=30.0, gpu_scale=True,
        tracks=_region_tracks(Layout.split_screen))
    assert cmd[0] == "ffmpeg"
    assert "-hwaccel" in cmd and cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    assert "-filter_complex" in cmd
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "[vout]"
    graph = cmd[cmd.index("-filter_complex") + 1]
    # two region crops + vstack into [vout]
    assert graph.count("crop=w=608:h=1080:") == 2
    assert "crop=w=608:h=1080:x=152:y=0" in graph
    assert "crop=w=608:h=1080:x=1160:y=0" in graph
    assert "[top][bottom]vstack=inputs=2[vout]" in graph
    # GPU scale per band (1080x960) — CUDA preferred
    assert graph.count("hwupload_cuda,scale_cuda=1080:960") == 2
    assert cmd.index("-i") < cmd.index("-filter_complex")


def test_split_screen_cpu_scale_fallback_keeps_nvenc_and_warns(
        tmp_path, monkeypatch, caplog):
    fake = FakeFFmpeg(filters_out=FILTERS_CPU_SENDCMD)  # no scale_cuda
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    with caplog.at_level("WARNING", logger="clipper.adapters.render"):
        adapter.render_mezzanine(
            str(src), make_plan(), Layout.split_screen,
            str(tmp_path / "clips" / "c1" / "m.mp4"), CFG,
            native_fps=30.0, tracks=_region_tracks(Layout.split_screen))
    cmd = fake.calls[-1]
    assert "h264_nvenc" in cmd  # encode STAYS NVENC
    assert "libx264" not in cmd
    assert "-hwaccel" not in cmd  # CPU path: no NVDEC either
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("scale=1080:960:flags=lanczos") == 2
    assert "scale_cuda" not in graph
    warnings = [r.message for r in caplog.records if "scale_cuda" in r.message]
    assert warnings, "CPU scale fallback must log a LOUD warning"


def test_facecam_cpu_scale_fallback_keeps_nvenc_and_warns(
        tmp_path, monkeypatch, caplog):
    fake = FakeFFmpeg(filters_out=FILTERS_CPU_SENDCMD)  # no scale_cuda
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    with caplog.at_level("WARNING", logger="clipper.adapters.render"):
        adapter.render_mezzanine(
            str(src), make_plan(), Layout.facecam,
            str(tmp_path / "clips" / "c1" / "m.mp4"), CFG,
            native_fps=30.0, tracks=_region_tracks(Layout.facecam))
    cmd = fake.calls[-1]
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    assert "-hwaccel" not in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1080:608:flags=lanczos" in graph
    assert "scale_cuda" not in graph
    assert "boxblur=8:1.5" in graph
    assert any("scale_cuda" in r.message for r in caplog.records)

def test_split_screen_requires_tracks_or_roles(tmp_path):
    adapter = FFmpegRenderAdapter()
    tracks = _region_tracks(Layout.split_screen)
    # missing tracks entirely
    with pytest.raises(MediaError, match="tracks"):
        adapter.mezzanine_command(
            "s", make_plan(), Layout.split_screen, str(tmp_path / "m.mp4"),
            CFG, native_fps=30.0, gpu_scale=True, tracks=None)
    # missing a required role (no `bottom`)
    with pytest.raises(MediaError, match="bottom"):
        adapter.mezzanine_command(
            "s", make_plan(), Layout.split_screen, str(tmp_path / "m.mp4"),
            CFG, native_fps=30.0, gpu_scale=True,
            tracks=tracks[:1])


def test_facecam_command_main_face_regions_nvenc_no_libx264():
    adapter = FFmpegRenderAdapter()
    cmd = adapter.mezzanine_command(
        "src.mp4", make_plan(), Layout.facecam, "out/clip/mezzanine.mp4",
        CFG, native_fps=30.0, gpu_scale=True, tracks=_region_tracks(Layout.facecam))
    assert cmd[0] == "ffmpeg"
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    assert "-filter_complex" in cmd
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "[vout]"
    graph = cmd[cmd.index("-filter_complex") + 1]
    # main + face crops (the face region is tracked as a square here)
    assert "crop=w=606:h=1080:x=657:y=0" in graph
    assert "crop=w=200:h=200:x=100:y=80" in graph
    # main band 1080x608 (top), face aspect-fit 1080x1080 (bottom 1080x1312)
    assert "scale_cuda=1080:608" in graph
    assert "scale_cuda=1080:1080" in graph
    # aspect differs -> blurred fill behind the face region
    assert "force_original_aspect_ratio=increase" in graph
    assert "crop=1080:1312" in graph
    assert "boxblur=8:1.5" in graph
    assert "overlay=(W-w)/2:(H-h)/2" in graph
    # vstack of the two bands into [vout] (1080x1920)
    assert "[main][bot]vstack=inputs=2[vout]" in graph


def test_facecam_exact_aspect_skips_blurred_fill():
    adapter = FFmpegRenderAdapter()
    from clipper.models import RegionTrack
    tracks = [
        RegionTrack(role="main", keys=[
            CropKey(t=0.0, x=0, y=0, w=606, h=1080)]),
        RegionTrack(role="face", keys=[
            CropKey(t=0.0, x=0, y=0, w=270, h=328)]),  # 270:328 == 1080:1312
    ]
    cmd = adapter.mezzanine_command(
        "s", make_plan(), Layout.facecam, "d", CFG,
        native_fps=30.0, gpu_scale=True, tracks=tracks)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "boxblur" not in graph
    assert "overlay=" not in graph
    assert "scale_cuda=1080:1312" in graph
    assert "[main][face]vstack=inputs=2[vout]" in graph


def test_facecam_requires_tracks_or_roles(tmp_path):
    adapter = FFmpegRenderAdapter()
    with pytest.raises(MediaError, match="tracks"):
        adapter.mezzanine_command(
            "s", make_plan(), Layout.facecam, str(tmp_path / "m.mp4"),
            CFG, native_fps=30.0, gpu_scale=True, tracks=None)
    tracks = _region_tracks(Layout.facecam)
    with pytest.raises(MediaError, match="face"):
        adapter.mezzanine_command(
            "s", make_plan(), Layout.facecam, str(tmp_path / "m.mp4"),
            CFG, native_fps=30.0, gpu_scale=True, tracks=tracks[:1])


def test_mezzanine_command_rejects_face_track_without_crop_path():
    adapter = FFmpegRenderAdapter()
    with pytest.raises(MediaError, match=r"crop_path"):
        adapter.mezzanine_command(
            "s", make_face_plan(n_keys=0),
            Layout.face_track, "d", CFG, native_fps=30.0, gpu_scale=True)


def test_mezzanine_command_rejects_nonpositive_native_fps():
    adapter = FFmpegRenderAdapter()
    with pytest.raises(MediaError, match="native fps"):
        adapter.mezzanine_command(
            "s", make_plan(), Layout.center_crop, "d", CFG,
            native_fps=0.0, gpu_scale=True)


# -- mezzanine: M7e face_track dynamic sendcmd crop -----------------------

def test_face_track_command_sendcmd_crop_nvenc_no_libx264():
    adapter = FFmpegRenderAdapter()
    plan = make_face_plan()
    cmd = adapter.mezzanine_command(
        "src.mp4", plan, Layout.face_track, "out/clip/mezzanine.mp4",
        CFG, native_fps=30.0, gpu_scale=True)
    assert cmd[0] == "ffmpeg"
    assert "-hwaccel" in cmd and cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "p5"
    assert "-cq" in cmd and cmd[cmd.index("-cq") + 1] == "16"
    assert "-an" in cmd
    assert cmd.index("-i") < cmd.index("-vf")
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("sendcmd=f=")
    crop_cmd_path = (
        Path("out/clip/mezzanine.mp4").resolve().parent / CROP_CMD_FILENAME)
    assert f"sendcmd=f={crop_cmd_path}," in vf
    first = plan.crop_path[0]
    assert f"crop=w={first.w}:h={first.h}:x={first.x}:y={first.y}" in vf
    assert "hwupload_cuda,scale_cuda=1080:1920" in vf
    # sendcmd BEFORE the crop filter (it drives the crop filter's x/y)
    assert vf.index("sendcmd") < vf.index("crop=")


def test_render_mezzanine_face_track_writes_crop_cmd_beside_mezzanine(
        tmp_path, monkeypatch):
    fake = FakeFFmpeg(filters_out=FILTERS_GPU_SENDCMD)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    clip_dir = tmp_path / "clips" / "jobX-clip01"
    dst = clip_dir / "mezzanine.mp4"
    plan = make_face_plan(n_keys=121)  # dense 4s @ 30fps key grid
    info = adapter.render_mezzanine(str(src), plan, Layout.face_track,
                                    str(dst), CFG, native_fps=30.0)
    assert isinstance(info, MezzanineInfo)
    # crop.cmd written BESIDE the mezzanine
    crop_cmd_file = clip_dir / CROP_CMD_FILENAME
    assert crop_cmd_file.is_file()
    content = crop_cmd_file.read_text(encoding="utf-8")
    assert content == crop_cmd_from_path(plan.crop_path, 30.0)
    # exactly one render invocation; vf drives the crop.cmd file
    render_calls = [c for c in fake.calls if "-filters" not in c]
    assert len(render_calls) == 1
    vf = render_calls[0][render_calls[0].index("-vf") + 1]
    assert f"sendcmd=f={crop_cmd_file.resolve()}" in vf


def test_face_track_cpu_scale_fallback_keeps_nvenc_and_warns(
        tmp_path, monkeypatch, caplog):
    fake = FakeFFmpeg(filters_out=FILTERS_CPU_SENDCMD)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with caplog.at_level("WARNING"):
        adapter.render_mezzanine(str(src), make_face_plan(), Layout.face_track,
                                 str(tmp_path / "m.mp4"), CFG,
                                 native_fps=30.0)
    assert any(CPU_SCALE_FALLBACK_WARNING in m for m in caplog.messages)
    render_calls = [c for c in fake.calls if "-filters" not in c]
    assert len(render_calls) == 1
    vf = render_calls[0][render_calls[0].index("-vf") + 1]
    assert vf.startswith("sendcmd=f=")
    assert "crop=w=608:h=1080:x=656:y=0" in vf
    assert "scale=1080:1920:flags=lanczos" in vf
    assert "scale_cuda" not in vf and "hwupload_cuda" not in vf
    assert "h264_nvenc" in render_calls[0]  # NVENC retained
    assert "libx264" not in render_calls[0]


def test_face_track_sendcmd_unavailable_raises_cleanly(
        tmp_path, monkeypatch):
    fake = FakeFFmpeg(filters_out=FILTERS_NO_SENDCMD)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(MediaError, match="sendcmd"):
        adapter.render_mezzanine(str(src), make_face_plan(), Layout.face_track,
                                 str(tmp_path / "m.mp4"), CFG, native_fps=30.0)
    assert [c for c in fake.calls if "-filters" not in c] == []  # no encode
    assert not (tmp_path / CROP_CMD_FILENAME).exists()


def test_face_track_crop_filter_unavailable_raises_cleanly(
        tmp_path, monkeypatch):
    fake = FakeFFmpeg(filters_out=FILTERS_NO_CROP)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(MediaError, match="crop"):
        adapter.render_mezzanine(str(src), make_face_plan(), Layout.face_track,
                                 str(tmp_path / "m.mp4"), CFG, native_fps=30.0)
    assert [c for c in fake.calls if "-filters" not in c] == []


def test_face_track_rejects_crop_path_not_starting_at_t0(
        tmp_path, monkeypatch):
    fake = FakeFFmpeg(filters_out=FILTERS_GPU_SENDCMD)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    plan = make_face_plan()
    plan.crop_path = plan.crop_path[1:]  # first key t != 0
    with pytest.raises(MediaError, match="t=0"):
        adapter.render_mezzanine(str(src), plan, Layout.face_track,
                                 str(tmp_path / "d.mp4"), CFG, native_fps=30.0)
    assert [c for c in fake.calls if "-filters" not in c] == []
    assert not (tmp_path / "d.mp4").exists()


# -- audio command --------------------------------------------------------

def test_audio_command_exact():
    cmd = FFmpegRenderAdapter().audio_command("src.mp4", 5.0, 15.0, "out.m4a")
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "5.000"
    assert "-to" in cmd and cmd[cmd.index("-to") + 1] == "15.000"
    assert cmd.index("-i") < cmd.index("-vn")
    assert "-vn" in cmd
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2"
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "48000"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"
    assert "-b:a" in cmd and cmd[cmd.index("-b:a") + 1] == "192k"
    assert cmd[-1] == "out.m4a"
    assert "libx264" not in cmd and "h264_nvenc" not in cmd


# -- render_mezzanine: scale_cuda preferred + probed contract -------------

def test_render_mezzanine_prefers_scale_cuda(tmp_path, monkeypatch):
    fake = FakeFFmpeg(filters_out=FILTERS_WITH_SCALE_CUDA)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "clips" / "jobX-clip01" / "mezzanine.mp4"
    info = adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                                    str(dst), CFG, native_fps=30.0)
    assert isinstance(info, MezzanineInfo)
    assert info.path == str(dst.resolve())
    assert (info.width, info.height, info.fps) == (1080, 1920, 30.0)
    assert info.schema_version == 1
    render_calls = [c for c in fake.calls if "-filters" not in c]
    assert len(render_calls) == 1
    vf = render_calls[0][render_calls[0].index("-vf") + 1]
    assert "scale_cuda=1080:1920" in vf


def test_render_mezzanine_cpu_scale_fallback_keeps_nvenc_and_warns(
        tmp_path, monkeypatch, caplog):
    fake = FakeFFmpeg(filters_out=FILTERS_WITHOUT_SCALE_CUDA)
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with caplog.at_level("WARNING"):
        adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                                 str(tmp_path / "m.mp4"), CFG, native_fps=30.0)
    assert any(CPU_SCALE_FALLBACK_WARNING in m for m in caplog.messages)
    render_calls = [c for c in fake.calls if "-filters" not in c]
    assert len(render_calls) == 1
    vf = render_calls[0][render_calls[0].index("-vf") + 1]
    assert "scale=1080:1920:flags=lanczos" in vf
    assert "scale_cuda" not in vf and "hwupload_cuda" not in vf


def test_render_mezzanine_probes_source_when_native_fps_given(tmp_path, monkeypatch):
    fake = FakeFFmpeg()
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    seen = []

    def fake_probe(p):
        seen.append(p)
        return probe_result(fps=29.97)
    monkeypatch.setattr(adapter._media, "probe", fake_probe)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                             str(tmp_path / "m.mp4"), CFG)
    assert seen[0] == str(src)  # source probed for native fps (then dst probed)
    render_calls = [c for c in fake.calls if "-vf" in c]
    assert render_calls[0][render_calls[0].index("-r") + 1] == "29.97"


def test_render_mezzanine_fails_cleanly_when_fps_unknown(tmp_path, monkeypatch):
    fake = FakeFFmpeg()
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result(fps=0.0))
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(MediaError, match="native fps"):
        adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                                 str(tmp_path / "m.mp4"), CFG)


# -- extract_audio: contract from probe -----------------------------------

def test_extract_audio_probes_contract(tmp_path, monkeypatch):
    fake = FakeFFmpeg()
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(
        adapter._media, "_run_ffprobe",
        lambda cmd: {
            "streams": [{"codec_type": "audio", "sample_rate": 48000,
                         "channels": 2}],
            "format": {"duration": "9.987"},
        })
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    info = adapter.extract_audio(str(src), 5.0, 15.0, str(tmp_path / "a.m4a"))
    assert isinstance(info, ClipAudioInfo)
    assert info.path == str((tmp_path / "a.m4a").resolve())
    assert (info.sample_rate, info.channels) == (48000, 2)
    assert info.duration_s == pytest.approx(9.987)
    render_calls = [c for c in fake.calls if "-vf" not in c]
    assert any("-c:a" in c and c[c.index("-c:a") + 1] == "aac" for c in render_calls)


def test_extract_audio_duration_fallback(tmp_path, monkeypatch):
    fake = FakeFFmpeg()
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "_run_ffprobe", lambda cmd: {"streams": []})
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    info = adapter.extract_audio(str(src), 5.0, 15.0, str(tmp_path / "a.m4a"))
    assert info.duration_s == pytest.approx(10.0)
    assert (info.sample_rate, info.channels) == (48000, 2)


# -- ffmpeg failure: clean errors, no fallback -----------------------------

def test_ffmpeg_render_failure_raises_clean_error(tmp_path, monkeypatch):
    fake = FakeFFmpeg(render_rc=1, render_stderr=b"nvenc init failed: no device")
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(MediaError) as ei:
        adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                                 str(tmp_path / "m.mp4"), CFG, native_fps=30.0)
    assert "no device" in str(ei.value)
    assert "NO silent encode fallback" in str(ei.value)


def test_ffmpeg_missing_raises(tmp_path, monkeypatch):
    def no_ffmpeg(cmd, **kwargs):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(subprocess, "run", no_ffmpeg)
    adapter = FFmpegRenderAdapter()
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(MediaError, match="FFmpeg not found"):
        adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                                 str(tmp_path / "m.mp4"), CFG, native_fps=30.0)


def test_ffmpeg_timeout_raises(tmp_path, monkeypatch):
    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(subprocess, "run", slow)
    adapter = FFmpegRenderAdapter(timeout=1)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(MediaError, match="timed out"):
        adapter.render_mezzanine(str(src), make_plan(), Layout.center_crop,
                                 str(tmp_path / "m.mp4"), CFG, native_fps=30.0)


def test_missing_source_raises_clean_error(monkeypatch):
    fake = FakeFFmpeg()
    install_fake(monkeypatch, fake)
    adapter = FFmpegRenderAdapter()
    with pytest.raises(MediaError, match="does not exist"):
        adapter.render_mezzanine("/nonexistent/x.mp4", make_plan(),
                                 Layout.center_crop, "out.mp4", CFG,
                                 native_fps=30.0)


# -- cpu_render: explicit debug path, LOUD warning --------------------------

def test_factory_cpu_render_loud_and_explicit(caplog):
    reset_render_adapter()
    with caplog.at_level("WARNING"):
        adapter = get_render_adapter(CFG, cpu_render=True)
    assert isinstance(adapter, FFmpegRenderAdapter)
    assert adapter.cpu_render is True
    assert any(CPU_RENDER_WARNING in m for m in caplog.messages)
    # debug path: NO NVDEC hwaccel, CPU scale, libx264 (DEBUG path only —
    # docs/spec/m7.md decision 2 forbids it as a PRODUCTION path)
    cmd = adapter.mezzanine_command(
        "s", make_plan(), Layout.center_crop, "d", CFG,
        native_fps=30.0, gpu_scale=True)
    assert "-hwaccel" not in cmd
    assert "h264_nvenc" not in cmd
    assert "scale=1080:1920:flags=lanczos" in cmd[cmd.index("-vf") + 1]
    assert "libx264" in cmd


def test_factory_defaults_to_gpu_ffmpeg():
    adapter = get_render_adapter(CFG)
    assert isinstance(adapter, FFmpegRenderAdapter)
    assert adapter.cpu_render is False
    # singleton caching
    assert get_render_adapter(CFG) is adapter


def test_factory_fixture_device():
    cfg = Config()
    cfg.devices.vision = "fixture"
    adapter = get_render_adapter(cfg)
    assert isinstance(adapter, FixtureRenderAdapter)


def test_factory_honours_cfg_cpu_render(tmp_path, caplog):
    cfg = Config()
    cfg.render.cpu_render = True
    if cfg.devices.vision in ("fixture", "mock"):
        cfg.devices.vision = "cuda:0"
    with caplog.at_level("WARNING"):
        adapter = get_render_adapter(cfg)
    assert isinstance(adapter, FFmpegRenderAdapter)
    assert adapter.cpu_render is True
    assert any(CPU_RENDER_WARNING in m for m in caplog.messages)


# -- FixtureRenderAdapter: valid contracts, placeholders --------------------

def test_fixture_mezzanine_contract(tmp_path):
    adapter = FixtureRenderAdapter()
    dst = tmp_path / "clips" / "jobX-clip01" / "mezzanine.mp4"
    info = adapter.render_mezzanine("/nonexistent/source.mp4", make_plan(),
                                    Layout.center_crop, str(dst), CFG)
    assert dst.exists()
    assert isinstance(info, MezzanineInfo)
    assert info.path == str(dst.resolve())
    assert (info.width, info.height) == (1080, 1920)
    assert info.fps > 0
    assert info.duration_s == pytest.approx(15.0)
    assert info.schema_version == 1


def test_fixture_mezzanine_honours_native_fps_and_rejects_bad_layout(tmp_path):
    adapter = FixtureRenderAdapter()
    info = adapter.render_mezzanine("s", make_plan(), Layout.center_crop,
                                    str(tmp_path / "m.mp4"), CFG, native_fps=24.0)
    assert info.fps == pytest.approx(24.0)
    with pytest.raises(MediaError):
        adapter.render_mezzanine("s", make_plan(), "bogus",
                                 str(tmp_path / "m4.mp4"), CFG)
    # M7f: split_screen/facecam ARE renderable, but they REQUIRE the
    # region tracks (clean MediaError without them, nothing written)
    for layout in (Layout.split_screen, Layout.facecam):
        with pytest.raises(MediaError, match="tracks"):
            adapter.render_mezzanine("s", make_plan(), layout,
                                     str(tmp_path / "m2.mp4"), CFG)
        assert not (tmp_path / "m2.mp4").exists()


def test_fixture_split_screen_contract_and_determinism(tmp_path):
    """M7f fixture adapter: split_screen renders the placeholder
    mezzanine with valid contracts; repeated calls are byte-identical
    (determinism)."""
    adapter = FixtureRenderAdapter()
    tracks = _region_tracks(Layout.split_screen)
    dst = tmp_path / "c1" / "m.mp4"
    info1 = adapter.render_mezzanine(
        "s", make_plan(), Layout.split_screen, str(dst), CFG,
        native_fps=30.0, tracks=tracks)
    body1 = dst.read_text(encoding="utf-8")
    json1 = info1.model_dump_json(indent=2)
    info2 = adapter.render_mezzanine(
        "s", make_plan(), Layout.split_screen, str(dst), CFG,
        native_fps=30.0, tracks=tracks)
    assert dst.is_file()
    assert isinstance(info1, MezzanineInfo)
    assert (info1.width, info1.height) == (1080, 1920)
    assert info1.duration_s == pytest.approx(15.0)
    # determinism: byte-identical placeholder + contract
    assert dst.read_text(encoding="utf-8") == body1
    assert info2.model_dump_json(indent=2) == json1


def test_fixture_facecam_contract_and_determinism(tmp_path):
    """M7f fixture adapter: facecam renders the placeholder mezzanine
    with valid contracts; repeated calls are byte-identical."""
    adapter = FixtureRenderAdapter()
    tracks = _region_tracks(Layout.facecam)
    dst = tmp_path / "c2" / "m.mp4"
    info1 = adapter.render_mezzanine(
        "s", make_plan(), Layout.facecam, str(dst), CFG,
        native_fps=30.0, tracks=tracks)
    body1 = dst.read_text(encoding="utf-8")
    json1 = info1.model_dump_json(indent=2)
    info2 = adapter.render_mezzanine(
        "s", make_plan(), Layout.facecam, str(dst), CFG,
        native_fps=30.0, tracks=tracks)
    assert dst.is_file()
    assert isinstance(info1, MezzanineInfo)
    assert (info1.width, info1.height) == (1080, 1920)
    assert dst.read_text(encoding="utf-8") == body1
    assert info2.model_dump_json(indent=2) == json1


def test_fixture_facecam_rejects_missing_face_role(tmp_path):
    adapter = FixtureRenderAdapter()
    tracks = _region_tracks(Layout.facecam)[:1]  # only `main`
    with pytest.raises(MediaError, match="face"):
        adapter.render_mezzanine(
            "s", make_plan(), Layout.facecam, str(tmp_path / "m9.mp4"),
            CFG, native_fps=30.0, tracks=tracks)
    assert not (tmp_path / "m9.mp4").exists()


def test_fixture_face_track_writes_crop_cmd_beside(tmp_path):
    adapter = FixtureRenderAdapter()
    clip_dir = tmp_path / "clips" / "jobX-clip01"
    dst = clip_dir / "mezzanine.mp4"
    plan = make_face_plan()
    info = adapter.render_mezzanine("s", plan, Layout.face_track,
                                    str(dst), CFG, native_fps=30.0)
    assert dst.exists()
    assert isinstance(info, MezzanineInfo)
    assert (info.width, info.height) == (1080, 1920)
    crop_cmd_file = clip_dir / CROP_CMD_FILENAME
    assert crop_cmd_file.is_file()
    assert crop_cmd_file.read_text(
        encoding="utf-8") == crop_cmd_from_path(plan.crop_path, 30.0)


def test_fixture_face_track_rejects_empty_crop_path(tmp_path):
    adapter = FixtureRenderAdapter()
    empty = make_face_plan(n_keys=0)
    with pytest.raises(MediaError, match="crop_path"):
        adapter.render_mezzanine("s", empty, Layout.face_track,
                                 str(tmp_path / "m9.mp4"), CFG)
    assert not (tmp_path / "m9.mp4").exists()


def test_fixture_audio_contract(tmp_path):
    adapter = FixtureRenderAdapter()
    dst = tmp_path / "audio.m4a"
    info = adapter.extract_audio("/nonexistent/source.mp4", 5.0, 15.0, str(dst))
    assert dst.exists()
    assert isinstance(info, ClipAudioInfo)
    assert info.path == str(dst.resolve())
    assert (info.sample_rate, info.channels) == (48000, 2)
    assert info.duration_s == pytest.approx(10.0)
    assert info.schema_version == 1


# -- M7c: final compose (ASS burn + loudnorm + NVENC) -----------------------

def _touch(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("placeholder")
    return str(p)


def _slice(cmd: list[str], token: str, n: int) -> list[str]:
    idx = cmd.index(token)
    return cmd[idx: idx + n]


def test_compose_command_ass_loudnorm_nvenc_no_libx264(tmp_path):
    cmd = FFmpegRenderAdapter().compose_command(
        "mezz.mp4", "audio.m4a", "captions.ass", "final.mp4", CFG,
    )
    joined = " ".join(cmd)
    assert "ass=captions.ass:fontsdir=" in joined
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in joined
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd
    # exact card shape: inputs, filters, NVENC final, AAC 160k, faststart
    assert cmd[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert ["-i", "mezz.mp4", "-i", "audio.m4a"] == cmd[5:9]
    assert "-vf" in cmd and "-af" in cmd
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert _slice(cmd, "-movflags", 2) == ["-movflags", "+faststart"]
    assert _slice(cmd, "-c:a", 4) == ["-c:a", "aac", "-b:a", "160k"]
    assert "-shortest" in cmd
    assert cmd[-1] == "final.mp4"


def test_compose_command_uses_cfg_final_cq_and_preset():
    cfg = CFG.model_copy(deep=True)
    cfg.render.nvenc_preset = "p6"
    cfg.render.final_cq = 21
    cmd = FFmpegRenderAdapter().compose_command(
        "m", "a", "c.ass", "f", cfg,
    )
    i = cmd.index("h264_nvenc")
    assert ["-preset", "p6", "-rc", "vbr", "-cq", "21"] == cmd[i + 1: i + 7]


def test_compose_final_returns_partial_artifact(tmp_path, monkeypatch):
    adapter = FFmpegRenderAdapter()
    paths = {
        "mezz": _touch(tmp_path, "clips/jobX-clip01/mezzanine.mp4"),
        "audio": _touch(tmp_path, "clips/jobX-clip01/audio.m4a"),
        "ass": _touch(tmp_path, "clips/jobX-clip01/captions.ass"),
    }
    dst = str(tmp_path / "clips" / "jobX-clip01" / "final.mp4")
    install_fake(monkeypatch, FakeFFmpeg())
    font_file = tmp_path / "fonts" / "Montserrat-ExtraBold.ttf"
    font_file.parent.mkdir(parents=True, exist_ok=True)
    font_file.write_bytes(b"fake ttf")
    monkeypatch.setattr(
        render_mod, "_caption_font_path", lambda: font_file,
    )
    monkeypatch.setattr(adapter._media, "probe", lambda p: probe_result())
    art = adapter.compose_final(
        paths["mezz"], paths["audio"], paths["ass"], dst, CFG,
    )
    assert isinstance(art, RenderArtifact)
    assert art.schema_version == 1
    # PARTIAL: the render stage (later M7) fills clip_id + thumbnail_path
    assert art.clip_id == ""
    assert art.thumbnail_path == ""
    # probed geometry + fixed encoder/hwaccel
    assert (art.width, art.height) == (1080, 1920)
    assert art.fps == pytest.approx(30.0)
    assert art.duration_s == pytest.approx(15.02)
    assert art.encoder == "h264_nvenc"
    assert art.hwaccel == "cuda"
    assert art.render_seconds >= 0.0
    assert art.final_path == Path(dst).resolve().as_posix()


def test_compose_final_missing_font_raises_clean_error(monkeypatch, tmp_path):
    adapter = FFmpegRenderAdapter()
    missing_font = Path(tmp_path) / "Montserrat-ExtraBold.ttf"
    monkeypatch.setattr(render_mod, "_caption_font_path", lambda: missing_font)
    mezz = _touch(tmp_path, "m.mp4")
    with pytest.raises(MediaError) as ei:
        adapter.compose_final(mezz, mezz, mezz, str(tmp_path / "f.mp4"), CFG)
    assert "install_fonts.sh" in str(ei.value)
    assert "Montserrat-ExtraBold.ttf" in str(ei.value)


def test_compose_final_missing_ass_filter_raises_clean_error(
        tmp_path, monkeypatch,
):
    adapter = FFmpegRenderAdapter()
    fake = FakeFFmpeg(filters_out=FILTERS_NO_ASS)
    monkeypatch.setattr(subprocess, "run", fake)
    present = _touch(tmp_path, "input.bin")
    with pytest.raises(MediaError) as ei:
        adapter.compose_final(
            present, present, present, str(tmp_path / "f.mp4"), CFG,
        )
    assert "ass" in str(ei.value)
    assert not any("-vf" in c for c in fake.calls)  # no encode was attempted


def test_compose_final_missing_inputs_raise_clean_error(
        tmp_path, monkeypatch,
):
    adapter = FFmpegRenderAdapter()
    missing = str(tmp_path / "nope.mp4")
    present = _touch(tmp_path, "yes.mp4")
    for label, idx in (
        ("mezzanine", 0), ("audio", 1), ("captions", 2),
    ):
        args = [present, present, present]
        args[idx] = missing
        with pytest.raises(MediaError) as ei:
            adapter.compose_final(
                *args, str(tmp_path / "f.mp4"), CFG,
            )
        assert label in str(ei.value)


# -- M7c: thumbnail ---------------------------------------------------------

def test_thumbnail_command_exact():
    cmd = FFmpegRenderAdapter().thumbnail_command("final.mp4", "thumb.jpg")
    assert cmd == [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "1.0",
        "-i", "final.mp4",
        "-vframes", "1",
        "thumb.jpg",
    ]


def test_make_thumbnail_returns_dst(tmp_path, monkeypatch):
    adapter = FFmpegRenderAdapter()
    final = _touch(tmp_path, "final.mp4")
    dst = str(tmp_path / "thumbnail.jpg")
    fake = FakeFFmpeg()
    install_fake(monkeypatch, fake)
    out = adapter.make_thumbnail(final, dst)
    assert out == Path(dst).resolve().as_posix()
    assert len(fake.calls) == 1
    assert fake.calls[0] == adapter.thumbnail_command(final, dst)


def test_make_thumbnail_missing_final_raises_clean_error():
    with pytest.raises(MediaError) as ei:
        FFmpegRenderAdapter().make_thumbnail("/nonexistent/final.mp4", "/tmp/t.jpg")
    assert "does not exist" in str(ei.value)


# -- M7c: FixtureRenderAdapter partial artifact -----------------------------

def test_fixture_compose_final_partial_artifact(tmp_path):
    adapter = FixtureRenderAdapter()
    dst = tmp_path / "jobX-clip01" / "final.mp4"
    art = adapter.compose_final("m", "a", "c", str(dst), CFG)
    assert dst.exists()  # placeholder written
    assert isinstance(art, RenderArtifact)
    assert art.schema_version == 1
    assert art.clip_id == "" and art.thumbnail_path == ""  # stage fills these
    assert (art.width, art.height) == (1080, 1920)
    assert art.fps > 0 and art.duration_s > 0
    assert art.encoder == "h264_nvenc"
    assert art.hwaccel == "cuda"
    assert art.render_seconds >= 0.0
    assert art.final_path == str(dst.resolve())


def test_fixture_make_thumbnail_placeholder(tmp_path):
    adapter = FixtureRenderAdapter()
    dst = str(tmp_path / "thumbnail.jpg")
    out = adapter.make_thumbnail("/nonexistent/final.mp4", dst)
    assert Path(dst).exists()
    assert out == Path(dst).resolve().as_posix()


# -- protocol conformance ---------------------------------------------------

def test_protocol_conformance():
    assert isinstance(FFmpegRenderAdapter(), RenderAdapter)
    assert isinstance(FixtureRenderAdapter(), RenderAdapter)
