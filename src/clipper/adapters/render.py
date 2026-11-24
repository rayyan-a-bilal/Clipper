"""
Render adapter (M7b + M7c — docs/spec/m7.md).

GPU0 (RTX 2080 Super, `cuda:0`) ONLY for production render: NVDEC decode
(`-hwaccel cuda`) + NVENC encode (`h264_nvenc`). P100s never do video.

M7b implements the mezzanine path (per-clip NVDEC→NVENC 1080x1920 native
fps) and per-clip audio extraction (AAC 48kHz stereo). M7c adds the final
compose (ASS captions burn + loudnorm + NVENC final) and the thumbnail.
The render STAGE is a LATER M7 milestone — NOT implemented here.

M7 fixed decisions implemented (docs/spec/m7.md):

1. Production render = NVDEC decode + NVENC encode on GPU0 only
   (`cfg.render.hwaccel = "cuda"`).
2. NO libx264 production path. `cpu_render=True` is an EXPLICIT debug
   path (logged loudly) — never an automatic fallback. `scale_cuda`
   absence degrades the SCALE to CPU only; the encode STAYS NVENC and a
   LOUD warning is logged.
3. GPU crop/scale/composite where supported (`hwupload_cuda` +
   `scale_cuda`).
8. `center_crop` static path renders with `faces == 0` — the static crop
   vf needs no face tracks at all.

`center_crop` static vf (docs/spec/m7.md M7b):

    GPU scale: crop=w=trunc(ih*9/16/2)*2:h=ih:x=(iw-ow)/2:y=0,
               hwupload_cuda,scale_cuda={w}:{h}:format=nv12
    CPU fallback (scale_cuda absent, warned loudly):
               crop=w=trunc(ih*9/16/2)*2:h=ih:x=(iw-ow)/2:y=0,
               scale={w}:{h}:flags=lanczos
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import Config
from ..models import (
    ClipAudioInfo,
    ClipPlan,
    Layout,
    MezzanineInfo,
    RenderArtifact,
)
from .media import FFmpegMediaAdapter, MediaError

logger = logging.getLogger(__name__)

# LOUD warnings (M7 decision 2: debug paths must announce themselves).
CPU_SCALE_FALLBACK_WARNING = (
    "LOUD WARNING [render]: `scale_cuda` filter NOT available — falling "
    "back to CPU scale (scale=...:flags=lanczos). The NVIDIA encode is "
    "RETAINED (h264_nvenc). This is explicitly NOT the silent production "
    "path (docs/spec/m7.md decision 2)."
)
CPU_RENDER_WARNING = (
    "LOUD WARNING [render]: cpu_render=True — EXPLICIT CPU/DEBUG render "
    "path active (no NVDEC, CPU scale, libx264 encode). This is a DEBUG "
    "tool, NOT a production fallback and NOT the M7 production path "
    "(docs/spec/m7.md decision 2)."
)

# M7b center_crop static crop (exact 9:16 portrait window, even width,
# horizontally centered — identical geometry rule to M6d's exact-9kx16k
# window). The scale step is appended per path.
_CENTER_CROP_VF = "crop=w=trunc(ih*9/16/2)*2:h=ih:x=(iw-ow)/2:y=0"

# M7c final-compose constants (docs/spec/m7.md M7c, decisions 6/7):
LOUDNORM_AAF = "loudnorm=I=-16:TP=-1.5:LRA=11"  # decision 6 (CPU audio)
CAPTION_FONT_NAME = "Montserrat-ExtraBold.ttf"
FONT_INSTALL_SCRIPT = "bash scripts/install_fonts.sh"
THUMBNAIL_SS = "1.0"  # thumbnail seek (card: `ffmpeg -ss 1.0 ...`)


def _fonts_dir() -> Path:
    """The repo's `assets/fonts` directory (M7a font installer target)."""
    return Path(__file__).resolve().parents[3] / "assets" / "fonts"


def _caption_font_path() -> Path:
    """`assets/fonts/Montserrat-ExtraBold.ttf` (tests monkeypatch this)."""
    return _fonts_dir() / CAPTION_FONT_NAME


def _resolve_layout(layout: Layout | str) -> Layout:
    """Coerce `Layout | str` to the `Layout` enum (clean MediaError)."""
    try:
        return Layout(layout)
    except ValueError as e:
        raise MediaError(f"Unknown layout: {layout!r}") from e


@runtime_checkable
class RenderAdapter(Protocol):
    """Protocol for M7b mezzanine render + audio extraction."""

    def render_mezzanine(
        self,
        source: str,
        plan: ClipPlan,
        layout: Layout | str,
        dst: str,
        cfg: Config,
        native_fps: float | None = None,
    ) -> MezzanineInfo:
        """Render the `[plan.start, plan.end]` window to a 1080x1920
        mezzanine (native fps) and probe the result.

        `native_fps` is an optional caller-provided value (judge note,
        docs/spec/m7.md M7b); when `None` the FFmpeg impl probes the
        source.
        """
        ...

    def extract_audio(
        self,
        source: str,
        start_s: float,
        end_s: float,
        dst: str,
    ) -> ClipAudioInfo:
        """Extract `[start_s, end_s]` audio as AAC 48kHz stereo m4a and
        probe the result."""
        ...

    def compose_final(
        self,
        mezzanine: str,
        audio: str,
        ass_path: str,
        dst: str,
        cfg: Config,
    ) -> RenderArtifact:
        """M7c final compose: burn the ASS captions (`ass=` filter with
        `fontsdir`), loudnorm the audio, NVENC-encode the final and return
        a PARTIAL `RenderArtifact` (stage fills `clip_id` /
        `thumbnail_path`)."""
        ...

    def make_thumbnail(self, final: str, dst: str) -> str:
        """M7c thumbnail: one still at t=1.0s of `final`; returns `dst`."""
        ...


class FFmpegRenderAdapter:
    """NVDEC + NVENC mezzanine / audio extraction (subprocess, GPU0).

    Raises `MediaError` (with the stderr tail) on any ffmpeg failure —
    NO silent libx264 production fallback (docs/spec/m7.md decision 2).
    ``cpu_render=True`` is an explicit, LOUDLY-logged DEBUG path only.
    """

    def __init__(self, timeout: int = 1800, cpu_render: bool = False):
        self.timeout = timeout
        self.cpu_render = cpu_render
        self._media = FFmpegMediaAdapter()
        self._scale_cuda: bool | None = None  # lazily detected + cached
        self._ass_filter: bool | None = None  # lazily detected + cached

    # -- scale_cuda capability detection ---------------------------------

    def scale_cuda_available(self) -> bool:
        """`ffmpeg -filters` contains `scale_cuda` (cached per instance)."""
        if self._scale_cuda is None:
            self._scale_cuda = self._probe_scale_cuda()
        return self._scale_cuda

    def _probe_scale_cuda(self) -> bool:
        return self._filter_available("scale_cuda")

    def _filter_available(self, name: str) -> bool:
        try:
            process = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True,
                text=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if process.returncode != 0:
            return False
        out = (process.stdout or b"").decode("utf-8", errors="replace")
        # Token match (not substring): `ass` must not match e.g. a future
        # filter merely CONTAINING the letters.
        return name in out.replace("\n", " ").split()

    # -- command builders (public for CPU-only testing) ------------------

    def mezzanine_command(
        self,
        source: str,
        plan: ClipPlan,
        layout: Layout | str,
        dst: str,
        cfg: Config,
        native_fps: float,
        gpu_scale: bool,
    ) -> list[str]:
        """Exact M7b mezzanine command (docs/spec/m7.md M7b) for ONE path:

        - `gpu_scale=True` and not `cpu_render`: `-hwaccel cuda` +
          `hwupload_cuda,scale_cuda=W:H:format=nv12` + `h264_nvenc`.
        - `gpu_scale=False` and not `cpu_render`: CPU scale
          `scale=W:H:flags=lanczos`, encode STAYS `h264_nvenc` (loud
          warning is logged by the caller).
        - `cpu_render=True`: NO `-hwaccel`, CPU scale, `libx264 -crf`
          (explicit DEBUG path only; loudly warned at construction).

        `center_crop` is the ONLY renderable layout in M7b (static path).
        """
        resolved = _resolve_layout(layout)
        if resolved is not Layout.center_crop:
            raise MediaError(
                f"layout {resolved.value!r} is not renderable in M7b — "
                "only the center_crop static path is implemented "
                "(docs/spec/m7.md M7b); tracking/composite rendering is a "
                "later M7 milestone"
            )
        native = float(native_fps)
        if native <= 0.0:
            raise MediaError(
                f"native fps must be > 0 (got {native_fps!r}) — the "
                "mezzanine renders at the SOURCE native fps (no re-time)"
            )
        width = int(cfg.render.output_width)
        height = int(cfg.render.output_height)
        vf = _CENTER_CROP_VF
        if self.cpu_render:
            vf = f"{vf},scale={width}:{height}:flags=lanczos"
            cmd: list[str] = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{plan.start:.3f}",
                "-to", f"{plan.end:.3f}",
                "-i", source,
                "-vf", vf,
                "-an",
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(int(cfg.render.crf)),
                "-r", f"{native:g}",
                dst,
            ]
            return cmd
        if gpu_scale:
            vf = f"{vf},hwupload_cuda,scale_cuda={width}:{height}:format=nv12"
        else:
            vf = f"{vf},scale={width}:{height}:flags=lanczos"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{plan.start:.3f}",
            "-to", f"{plan.end:.3f}",
            "-i", source,
            "-vf", vf,
            "-an",
            "-c:v", "h264_nvenc",
            "-preset", str(cfg.render.nvenc_preset),
            "-rc", "vbr",
            "-cq", str(int(cfg.render.mezz_cq)),
            "-r", f"{native:g}",
            dst,
        ]
        return cmd

    def audio_command(
        self,
        source: str,
        start_s: float,
        end_s: float,
        dst: str,
    ) -> list[str]:
        """Exact M7b audio-extraction command (docs/spec/m7.md M7b):
        AAC 48kHz stereo, `-b:a 192k`."""
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{float(start_s):.3f}",
            "-to", f"{float(end_s):.3f}",
            "-i", source,
            "-vn",
            "-ac", "2",
            "-ar", "48000",
            "-c:a", "aac",
            "-b:a", "192k",
            dst,
        ]

    # -- ffmpeg execution (clean failures, no fallback) -------------------

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        try:
            process = subprocess.run(
                cmd, capture_output=True, text=False, timeout=self.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise MediaError(
                f"ffmpeg render timed out after {self.timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise MediaError("FFmpeg not found. Please install ffmpeg.") from e

        if process.returncode != 0:
            stderr_text = process.stderr.decode("utf-8", errors="ignore")[-2000:]
            raise MediaError(
                f"ffmpeg render failed with code {process.returncode}; "
                f"there is NO silent encode fallback (docs/spec/m7.md "
                f"decision 2). stderr tail: {stderr_text}"
            )

    # -- outputs -----------------------------------------------------------

    def _probe_audio_info(self, dst: str, expected_s: float) -> tuple[int, int, float]:
        """`(sample_rate, channels, duration_s)` from the rendered m4a."""
        info: dict[str, Any]
        try:
            info = self._media._run_ffprobe([
                "-i", dst,
                "-show_entries", "stream=sample_rate,channels",
                "-show_entries", "format=duration",
            ])
        except MediaError:
            info = {}
        stream = None
        for st in info.get("streams") or []:
            if st.get("codec_type") == "audio" or st.get("sample_rate"):
                stream = st
                break
        if stream is None:
            stream = (info.get("streams") or [None])[0]
        sample_rate = (
            int(float(str(stream.get("sample_rate", 48000)))) if stream else 48000
        )
        channels = (
            int(float(str(stream.get("channels", 2)))) if stream else 2
        )
        duration_s = 0.0
        try:
            duration_s = float(info.get("format", {}).get("duration", 0.0))
        except (TypeError, ValueError):
            duration_s = 0.0
        if duration_s <= 0.0:
            duration_s = round(max(0.0, expected_s), 3)
        return sample_rate, channels, duration_s

    def render_mezzanine(
        self,
        source: str,
        plan: ClipPlan,
        layout: Layout | str,
        dst: str,
        cfg: Config,
        native_fps: float | None = None,
    ) -> MezzanineInfo:
        """Render one clip's mezzanine (center_crop static path, M7b)."""
        if not os.path.exists(source):
            raise MediaError(f"Source file does not exist: {source}")
        if native_fps is None:
            native_fps = self._media.probe(source).fps
        gpu_scale = not self.cpu_render and self.scale_cuda_available()
        if not self.cpu_render and not gpu_scale:
            logger.warning(CPU_SCALE_FALLBACK_WARNING)
        elif self.cpu_render:
            logger.warning(CPU_RENDER_WARNING)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg(self.mezzanine_command(
            source, plan, layout, dst, cfg, native_fps, gpu_scale
        ))
        probe = self._media.probe(dst)
        return MezzanineInfo(
            path=os.path.abspath(dst),
            width=int(probe.width),
            height=int(probe.height),
            fps=float(probe.fps),
            duration_s=float(probe.duration_s),
        )

    def extract_audio(
        self,
        source: str,
        start_s: float,
        end_s: float,
        dst: str,
    ) -> ClipAudioInfo:
        """Extract one clip's audio window (AAC 48kHz stereo, M7b)."""
        if not os.path.exists(source):
            raise MediaError(f"Source file does not exist: {source}")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg(self.audio_command(source, start_s, end_s, dst))
        sample_rate, channels, duration_s = self._probe_audio_info(
            dst, max(0.0, float(end_s) - float(start_s))
        )
        return ClipAudioInfo(
            path=os.path.abspath(dst),
            sample_rate=int(sample_rate),
            channels=int(channels),
            duration_s=float(duration_s),
        )

    # -- M7c: final compose (ASS burn + loudnorm + NVENC) ------------------

    def ass_filter_available(self) -> bool:
        """`ffmpeg -filters` contains the `ass` filter (cached per
        instance). The ASS burn is REQUIRED for the final compose."""
        if self._ass_filter is None:
            self._ass_filter = self._filter_available("ass")
        return self._ass_filter

    @staticmethod
    def _require_caption_font() -> None:
        font = _caption_font_path()
        if not font.is_file():
            raise MediaError(
                f"caption font missing: {font} — install it with `"
                f"{FONT_INSTALL_SCRIPT}` (from the repo root) before the "
                f"final compose (docs/spec/m7.md M7c)"
            )

    def compose_command(
        self,
        mezzanine: str,
        audio: str,
        ass_path: str,
        dst: str,
        cfg: Config,
    ) -> list[str]:
        """Exact M7c final-compose command (docs/spec/m7.md M7c):

        `ffmpeg -hide_banner -loglevel error -y -i {mezz} -i {audio}
        -vf ass={ass}:fontsdir={assets/fonts}
        -af loudnorm=I=-16:TP=-1.5:LRA=11
        -c:v h264_nvenc -preset {nvenc_preset} -rc vbr -cq {final_cq}
        -c:a aac -b:a 160k -movflags +faststart -shortest {dst}`
        """
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", mezzanine,
            "-i", audio,
            "-vf", f"ass={ass_path}:fontsdir={_fonts_dir()}",
            "-af", LOUDNORM_AAF,
            "-c:v", "h264_nvenc",
            "-preset", str(cfg.render.nvenc_preset),
            "-rc", "vbr",
            "-cq", str(int(cfg.render.final_cq)),
            "-c:a", "aac",
            "-b:a", "160k",
            "-movflags", "+faststart",
            "-shortest",
            dst,
        ]

    def compose_final(
        self,
        mezzanine: str,
        audio: str,
        ass_path: str,
        dst: str,
        cfg: Config,
    ) -> RenderArtifact:
        """Compose the final clip (M7c) and probe the result.

        Fails CLEANLY (no encode, no fallback) when the `ass` filter is
        unavailable or `assets/fonts/Montserrat-ExtraBold.ttf` is missing
        (pointing at `scripts/install_fonts.sh`). Returns a PARTIAL
        `RenderArtifact`: `clip_id` = `""` and `thumbnail_path` = `""`
        are filled by the (later M7) render STAGE, `render_seconds`
        covers this ffmpeg invocation only.
        """
        for label, path in (
            ("mezzanine", mezzanine), ("audio", audio), ("captions", ass_path),
        ):
            if not os.path.exists(path):
                raise MediaError(f"{label} file does not exist: {path}")
        if not self.ass_filter_available():
            raise MediaError(
                "ffmpeg build lacks the `ass` filter — cannot burn the "
                "caption subtitles (docs/spec/m7.md M7c). Reinstall an "
                "ffmpeg built with libass."
            )
        self._require_caption_font()
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        self._run_ffmpeg(self.compose_command(
            mezzanine, audio, ass_path, dst, cfg))
        render_seconds = time.monotonic() - started
        probe = self._media.probe(dst)
        return RenderArtifact(
            clip_id="",  # filled by the render stage (later M7)
            mezzanine_path=os.path.abspath(mezzanine),
            audio_path=os.path.abspath(audio),
            captions_path=os.path.abspath(ass_path),
            final_path=os.path.abspath(dst),
            thumbnail_path="",  # filled by the render stage (make_thumbnail)
            width=int(probe.width),
            height=int(probe.height),
            fps=float(probe.fps),
            duration_s=float(probe.duration_s),
            encoder="h264_nvenc",
            hwaccel=str(cfg.render.hwaccel),
            render_seconds=round(float(render_seconds), 3),
        )

    # -- M7c: thumbnail ----------------------------------------------------

    def thumbnail_command(self, final: str, dst: str) -> list[str]:
        """Exact M7c thumbnail command (docs/spec/m7.md M7c):
        `ffmpeg -ss 1.0 -i {final} -vframes 1 {dst}`."""
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", THUMBNAIL_SS,
            "-i", final,
            "-vframes", "1",
            dst,
        ]

    def make_thumbnail(self, final: str, dst: str) -> str:
        """Grab one still at t=1.0s of `final` (M7c); returns `dst`."""
        if not os.path.exists(final):
            raise MediaError(f"Final file does not exist: {final}")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg(self.thumbnail_command(final, dst))
        return os.path.abspath(dst)


class FixtureRenderAdapter:
    """CPU-only placeholder adapter (tests / no-GPU wiring; M7b).

    Writes placeholder files and returns canned, valid
    `MezzanineInfo` / `ClipAudioInfo` — never touches ffmpeg or a GPU.
    """

    def render_mezzanine(
        self,
        source: str,
        plan: ClipPlan,
        layout: Layout | str,
        dst: str,
        cfg: Config,
        native_fps: float | None = None,
    ) -> MezzanineInfo:
        resolved = _resolve_layout(layout)
        if resolved is not Layout.center_crop:
            raise MediaError(
                f"layout {resolved.value!r} is not renderable in M7b — "
                "only the center_crop static path is implemented "
                "(docs/spec/m7.md M7b)"
            )
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("fixture mezzanine (M7b)")
        return MezzanineInfo(
            path=os.path.abspath(dst),
            width=int(cfg.render.output_width),
            height=int(cfg.render.output_height),
            fps=float(native_fps) if native_fps else 30.0,
            duration_s=round(max(0.0, plan.end - plan.start), 3),
        )

    def extract_audio(
        self,
        source: str,
        start_s: float,
        end_s: float,
        dst: str,
    ) -> ClipAudioInfo:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("fixture clip audio (M7b)")
        return ClipAudioInfo(
            path=os.path.abspath(dst),
            sample_rate=48000,
            channels=2,
            duration_s=round(max(0.0, float(end_s) - float(start_s)), 3),
        )

    # -- M7c (placeholders; no ffmpeg, no GPU) -----------------------------

    def compose_final(
        self,
        mezzanine: str,
        audio: str,
        ass_path: str,
        dst: str,
        cfg: Config,
    ) -> RenderArtifact:
        """Placeholder final (M7c): valid PARTIAL `RenderArtifact`, canned
        geometry — `clip_id`/`thumbnail_path` left for the stage."""
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("fixture final (M7c)")
        return RenderArtifact(
            clip_id="",
            mezzanine_path=os.path.abspath(mezzanine),
            audio_path=os.path.abspath(audio),
            captions_path=os.path.abspath(ass_path),
            final_path=os.path.abspath(dst),
            thumbnail_path="",
            width=int(cfg.render.output_width),
            height=int(cfg.render.output_height),
            fps=30.0,
            duration_s=15.0,
            encoder="h264_nvenc",
            hwaccel=str(cfg.render.hwaccel),
            render_seconds=0.0,
        )

    def make_thumbnail(self, final: str, dst: str) -> str:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("fixture thumbnail (M7c)")
        return os.path.abspath(dst)


_render_adapter_instance: RenderAdapter | None = None


def get_render_adapter(
    cfg: Config | None = None,
    cpu_render: bool | None = None,
) -> RenderAdapter:
    """Factory per docs/spec/m7.md M7b (singleton like `get_adapter`).

    - `cpu_render=True` (explicit arg OR `cfg.render.cpu_render`):
      `FFmpegRenderAdapter(cpu_render=True)` + LOUD warning (EXPLICIT
      DEBUG path only — docs/spec/m7.md decision 2).
    - `cfg.devices.vision` in (`"fixture"`, `"mock"`) and no cpu render:
      `FixtureRenderAdapter` (tests only; CPU-only).
    - otherwise: `FFmpegRenderAdapter` (NVDEC/NVENC production path).
    """
    global _render_adapter_instance
    if _render_adapter_instance is not None:
        return _render_adapter_instance
    fixture = (
        cfg is not None
        and (cfg.devices.vision or "").strip().lower() in ("fixture", "mock")
    )
    if cpu_render is None and cfg is not None:
        cpu_render = bool(cfg.render.cpu_render)
    if cpu_render:
        logger.warning(CPU_RENDER_WARNING)
        _render_adapter_instance = FFmpegRenderAdapter(cpu_render=True)
    elif fixture:
        _render_adapter_instance = FixtureRenderAdapter()
    else:
        _render_adapter_instance = FFmpegRenderAdapter()
    return _render_adapter_instance


def reset_render_adapter() -> None:
    """Test helper: drop the singleton so factories re-select."""
    global _render_adapter_instance
    _render_adapter_instance = None
