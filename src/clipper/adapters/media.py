"""
Media adapter implementations for clipper.
"""

import json
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import AudioInfo, ProbeResult, ProxyInfo


class MediaError(Exception):
    """Custom exception for media operations."""
    pass


class MediaAdapter(ABC):
    """Abstract media adapter interface."""

    @abstractmethod
    def probe(self, path: str) -> ProbeResult:
        """Probe a media file to get metadata."""
        pass

    @abstractmethod
    def make_proxy(self, src: str, dst: str) -> ProxyInfo:
        """Create a proxy video file."""
        pass

    @abstractmethod
    def extract_wav(self, src: str, dst: str) -> AudioInfo:
        """Extract audio to WAV format."""
        pass


class FFmpegMediaAdapter(MediaAdapter):
    """FFmpeg-based media adapter."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def _run_ffmpeg(self, cmd: list, input_path: str = None, timeout: int = None):
        """Run ffmpeg command with error handling."""
        if timeout is None:
            timeout = self.timeout

        # Add common flags
        full_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + cmd

        try:
            process = subprocess.run(
                full_cmd,
                capture_output=True,
                text=False,
                timeout=timeout
            )

            if process.returncode != 0:
                stderr_text = process.stderr.decode('utf-8', errors='ignore')[-2000:]
                raise MediaError(
                    f"FFmpeg failed with code {process.returncode}: {stderr_text}"
                )

        except subprocess.TimeoutExpired as e:
            raise MediaError(f"FFmpeg timed out after {timeout} seconds") from e
        except FileNotFoundError as e:
            raise MediaError("FFmpeg not found. Please install ffmpeg.") from e

    def _run_ffprobe(self, cmd: list, timeout: int = 60) -> dict[str, Any]:
        """Run ffprobe command with error handling."""
        full_cmd = ["ffprobe", "-v", "error", "-print_format", "json"] + cmd

        try:
            process = subprocess.run(
                full_cmd,
                capture_output=True,
                text=False,
                timeout=timeout
            )

            if process.returncode != 0:
                stderr_text = process.stderr.decode('utf-8', errors='ignore')[-2000:]
                raise MediaError(
                    f"FFprobe failed with code {process.returncode}: {stderr_text}"
                )

            output = process.stdout.decode('utf-8')
            return json.loads(output)

        except subprocess.TimeoutExpired as e:
            raise MediaError(f"FFprobe timed out after {timeout} seconds") from e
        except FileNotFoundError as e:
            raise MediaError(
                "FFprobe not found. Please install ffprobe (part of ffmpeg)."
            ) from e
        except json.JSONDecodeError as e:
            raise MediaError(f"Failed to parse ffprobe output: {e}") from e

    def probe(self, path: str) -> ProbeResult:
        """Probe a media file to get metadata."""
        if not os.path.exists(path):
            raise MediaError(f"Source file does not exist: {path}")

        info = self._run_ffprobe([
            "-i", path,
            "-show_format",
            "-show_streams"
        ])

        # Get format information
        format_info = info.get('format', {})
        size_bytes = int(format_info.get('size', 0))
        duration_s = float(format_info.get('duration', 0))

        # Get stream information
        streams = info.get('streams', [])
        video_stream = None
        audio_stream = None

        for stream in streams:
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_stream = stream

        # Extract video info
        width = int(video_stream.get('width', 0)) if video_stream else 0
        height = int(video_stream.get('height', 0)) if video_stream else 0
        vcodec = video_stream.get('codec_name') if video_stream else ''

        # Extract fps
        fps = 0.0
        if video_stream:
            # Try avg_frame_rate first, fallback to r_frame_rate
            avg_frame_rate = video_stream.get('avg_frame_rate')
            r_frame_rate = video_stream.get('r_frame_rate')

            if avg_frame_rate and avg_frame_rate != '0/0':
                try:
                    num, den = map(int, avg_frame_rate.split('/'))
                    fps = num / den if den != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    pass
            elif r_frame_rate and r_frame_rate != '0/0':
                try:
                    num, den = map(int, r_frame_rate.split('/'))
                    fps = num / den if den != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    pass

        # Extract audio info
        acodec = audio_stream.get('codec_name') if audio_stream else None
        has_audio = audio_stream is not None

        return ProbeResult(
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
            vcodec=vcodec,
            acodec=acodec,
            has_audio=has_audio,
            size_bytes=size_bytes
        )

    def make_proxy(self, src: str, dst: str) -> ProxyInfo:
        """Create a proxy video file."""
        # Filter chain for proxy: fit within 960x540, then force even dimensions
        vf_chain = ",".join([
            "scale=960:540:force_original_aspect_ratio=decrease",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ])

        cmd = ["-i", src, "-vf", vf_chain, "-r", "15", "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", dst]

        self._run_ffmpeg(cmd, input_path=src, timeout=1800)

        # Get proxy information after creating it
        probe_result = self.probe(dst)
        return ProxyInfo(
            width=probe_result.width,
            height=probe_result.height,
            fps=probe_result.fps,
            path=os.path.abspath(dst)
        )

    def extract_wav(self, src: str, dst: str) -> AudioInfo:
        """Extract audio to WAV format."""
        cmd = ["-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]

        self._run_ffmpeg(cmd, input_path=src, timeout=600)

        # Get audio info
        probe_result = self.probe(dst)
        return AudioInfo(
            sample_rate=16000,
            channels=1,
            duration_s=probe_result.duration_s,
            path=os.path.abspath(dst)
        )


class FixtureMediaAdapter(MediaAdapter):
    """Mock media adapter for testing."""

    def __init__(self, probe_result: ProbeResult = None, has_audio: bool = True):
        self._probe_result = probe_result or ProbeResult(
            duration_s=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            vcodec="h264",
            acodec="aac",
            has_audio=has_audio,
            size_bytes=1000000
        )

    def probe(self, path: str) -> ProbeResult:
        """Use the fixed probe result."""
        return self._probe_result

    def make_proxy(self, src: str, dst: str) -> ProxyInfo:
        """Create a small placeholder proxy file."""
        # Create a dummy proxy file
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with open(dst, 'w') as f:
            f.write("dummy proxy")
        return ProxyInfo(
            width=960,
            height=540,
            fps=15.0,
            path=os.path.abspath(dst)
        )

    def extract_wav(self, src: str, dst: str) -> AudioInfo:
        """Create a dummy WAV file."""
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with open(dst, 'w') as f:
            f.write("dummy audio")
        return AudioInfo(
            sample_rate=16000,
            channels=1,
            duration_s=10.0,
            path=os.path.abspath(dst)
        )


_adapter_instance: MediaAdapter | None = None


def get_adapter() -> MediaAdapter:
    """Get the configured media adapter."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = FFmpegMediaAdapter()
    return _adapter_instance
