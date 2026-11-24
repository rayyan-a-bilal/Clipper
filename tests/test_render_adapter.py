"""M7b CPU-only tests: render adapter (NVDEC/NVENC mezzanine + audio).

All ffmpeg I/O is mocked via `subprocess.run` — no GPU, no real ffmpeg,
no network, no downloads (docs/spec/m7.md M7b).
"""

import subprocess

import pytest

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
from clipper.models import ClipAudioInfo, ClipPlan, Layout, MezzanineInfo, ProbeResult

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
    assert "hwupload_cuda,scale_cuda=1080:1920:format=nv12" in vf
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


def test_mezzanine_command_rejects_noncenter_crop_layouts():
    adapter = FFmpegRenderAdapter()
    for layout in (Layout.face_track, Layout.split_screen, Layout.facecam, "bogus"):
        with pytest.raises(MediaError):
            adapter.mezzanine_command(
                "s", make_plan(), layout, "d", CFG, native_fps=30.0, gpu_scale=True)


def test_mezzanine_command_rejects_nonpositive_native_fps():
    adapter = FFmpegRenderAdapter()
    with pytest.raises(MediaError, match="native fps"):
        adapter.mezzanine_command(
            "s", make_plan(), Layout.center_crop, "d", CFG,
            native_fps=0.0, gpu_scale=True)


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
    assert "scale_cuda=1080:1920:format=nv12" in vf


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
        adapter.render_mezzanine("s", make_plan(), Layout.face_track,
                                 str(tmp_path / "m2.mp4"), CFG)


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


# -- protocol conformance ---------------------------------------------------

def test_protocol_conformance():
    assert isinstance(FFmpegRenderAdapter(), RenderAdapter)
    assert isinstance(FixtureRenderAdapter(), RenderAdapter)
