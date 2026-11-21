"""
Vision-segment adapter (M6a — docs/spec/m6.md).

GPU0 (RTX 2080 Super, `cuda:0`) ONLY for render work: NVDEC decode
(`-hwaccel cuda`) + NVENC encode (`h264_nvenc`). P100s never do video.

M6a renders the per-clip vision segment (960x540 fit, aspect preserved,
NO padding, even dims, fps = min(native, 30.0) with native 0/unknown ->
30.0) for the SELECTED clip range only. Face detection, tracking, layout
and the M6 stage are M6b+ and explicitly NOT implemented here.
"""

import os
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Config
from ..models import VisionSegmentInfo
from .media import FFmpegMediaAdapter, MediaError

# M6a fixed cap (docs/spec/m6.md): fps = min(native, 30.0); native
# 0/unknown -> 30.0.
VISION_FPS_MAX_CAP = 30.0

# No padding — only fit-then-even-dims (NVENC requires even dims).
_VISION_VF = (
    "scale={width}:{height}:force_original_aspect_ratio=decrease,"
    "scale=trunc(iw/2)*2:trunc(ih/2)*2"
)


def select_vision_fps(native_fps: float, cap: float = VISION_FPS_MAX_CAP) -> float:
    """M6a fps rule: `min(native_fps, cap)`; native fps 0/unknown -> `cap`."""
    cap = float(cap)
    native = float(native_fps)
    if native <= 0.0:
        return cap
    return min(native, cap)


@runtime_checkable
class VisionAdapter(Protocol):
    """Protocol for per-clip vision segment creation (docs/spec/m6.md)."""

    def make_segment(
        self,
        source_path: str,
        dst_path: str,
        start_s: float,
        end_s: float,
        fps: float,
        width: int,
        height: int,
        clip_id: str = "unknown",
    ) -> VisionSegmentInfo:
        """Render the `[start_s, end_s)` source window to a vision segment.

        `clip_id` is optional (default `"unknown"`) so callers using the
        7-argument form from the M6a card still work; M6b passes the real
        clip_id (judgment note in docs/spec/m6.md).
        """
        ...


class FFmpegVisionAdapter:
    """NVDEC + NVENC implementation (subprocess, GPU0 only).

    Raises `MediaError` (with the stderr tail) on NVDEC/NVENC failure —
    NO silent CPU/libx264 fallback (docs/spec/m6.md).
    """

    def __init__(self, timeout: int = 1800):
        self.timeout = timeout
        self._media = FFmpegMediaAdapter()

    def build_command(
        self,
        source_path: str,
        dst_path: str,
        start_s: float,
        end_s: float,
        fps: float,
        width: int,
        height: int,
    ) -> list[str]:
        """Exact M6a command (docs/spec/m6.md). No `pad` filter."""
        vf = _VISION_VF.format(width=width, height=height)
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-hwaccel", "cuda",
            "-ss", f"{start_s:.3f}",
            "-to", f"{end_s:.3f}",
            "-i", source_path,
            "-vf", vf,
            "-r", f"{fps:g}",
            "-an",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "23",
            dst_path,
        ]

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        try:
            process = subprocess.run(
                cmd, capture_output=True, text=False, timeout=self.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise MediaError(
                f"ffmpeg vision segment timed out after {self.timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise MediaError("FFmpeg not found. Please install ffmpeg.") from e

        if process.returncode != 0:
            stderr_text = process.stderr.decode("utf-8", errors="ignore")[-2000:]
            raise MediaError(
                f"NVDEC/NVENC vision segment failed with code "
                f"{process.returncode}; there is NO silent CPU fallback "
                f"(docs/spec/m6.md). stderr tail: {stderr_text}"
            )

    def _probe_frame_count(self, dst_path: str, fallback: int) -> int:
        """`nb_frames` from the rendered file; fallback = round(dur*fps)."""
        try:
            info = self._media._run_ffprobe([
                "-i", dst_path,
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
            ])
            streams = info.get("streams") or []
            if streams and streams[0].get("nb_frames"):
                count = int(float(str(streams[0]["nb_frames"]).strip()))
                if count > 0:
                    return count
        except (MediaError, ValueError):
            pass  # missing/odd nb_frames -> computed fallback below
        return max(1, fallback)

    def make_segment(
        self,
        source_path: str,
        dst_path: str,
        start_s: float,
        end_s: float,
        fps: float,
        width: int,
        height: int,
        clip_id: str = "unknown",
    ) -> VisionSegmentInfo:
        """Render one per-clip vision segment and probe the result."""
        if not os.path.exists(source_path):
            raise MediaError(f"Source file does not exist: {source_path}")
        fps = min(float(fps), VISION_FPS_MAX_CAP)  # spec cap, always
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        self._run_ffmpeg(self.build_command(
            source_path, dst_path, start_s, end_s, fps, width, height
        ))

        probe = self._media.probe(dst_path)
        return VisionSegmentInfo(
            clip_id=clip_id,
            path=os.path.abspath(dst_path),
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            frame_count=self._probe_frame_count(
                dst_path, int(round((end_s - start_s) * probe.fps))
            ),
            source_start=start_s,
            source_end=end_s,
        )


class FixtureVisionAdapter:
    """CPU-only placeholder adapter (tests / no-GPU wiring; M6a).

    Writes a placeholder file and returns canned, valid
    `VisionSegmentInfo` — never touches ffmpeg or a GPU.
    """

    def make_segment(
        self,
        source_path: str,
        dst_path: str,
        start_s: float,
        end_s: float,
        fps: float,
        width: int,
        height: int,
        clip_id: str = "unknown",
    ) -> VisionSegmentInfo:
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dst_path).write_text("fixture vision segment (M6a)")
        fps = min(float(fps), VISION_FPS_MAX_CAP)
        return VisionSegmentInfo(
            clip_id=clip_id,
            path=os.path.abspath(dst_path),
            width=int(width),
            height=int(height),
            fps=fps,
            frame_count=max(1, int(round((end_s - start_s) * fps))),
            source_start=start_s,
            source_end=end_s,
        )


_vision_adapter_instance: VisionAdapter | None = None


def get_vision_adapter(cfg: Config | None = None) -> VisionAdapter:
    """Factory per docs/spec/m6.md: `cfg.devices.vision` selects.

    - `"cuda:0"` (default) / any real device -> `FFmpegVisionAdapter`
    - `"fixture"` -> `FixtureVisionAdapter` (tests only; CPU-only)
    """
    global _vision_adapter_instance
    if _vision_adapter_instance is None:
        device = ""
        if cfg is not None and cfg.devices is not None:
            device = (cfg.devices.vision or "").strip().lower()
        if device in ("fixture", "cpu", "mock"):
            _vision_adapter_instance = FixtureVisionAdapter()
        else:
            _vision_adapter_instance = FFmpegVisionAdapter()
    return _vision_adapter_instance
