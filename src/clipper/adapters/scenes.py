"""
Shot-detection adapter protocol, PySceneDetect adapter and fixture (M3b).

PySceneDetectAdapter uses the scenedetect convenience API
(``detect`` + ``ContentDetector``) with SHOT_THRESHOLD. Heavy imports
(scenedetect, cv2) are deferred into ``detect_shots`` so that importing
this module stays light.

This project MUST use opencv-python-headless and NEVER opencv-python:
it runs on a headless server where the GUI build fails with
``libGL.so.1``. (See docs/spec/m3.md and AGENTS.md.)
"""

import os
from abc import ABC, abstractmethod

from ..models import Shot

# PySceneDetect ContentDetector threshold (see docs/spec/m3.md).
SHOT_THRESHOLD = 27.0


def _timecode_seconds(tc) -> float:
    """Convert a scenedetect frame timecode to seconds across API versions."""
    seconds = getattr(tc, "seconds", None)
    if seconds is not None:
        return float(seconds)
    return float(tc.get_seconds())  # fallback for scenedetect < 0.7 (deprecated)


class SceneError(Exception):
    """Custom exception for shot-detection operations."""

    pass


class SceneAdapter(ABC):
    """Abstract interface for shot detection."""

    @abstractmethod
    def detect_shots(self, video_path: str, duration_s: float) -> list[Shot]:
        """Detect shot boundaries in a video, within [0, duration_s]."""
        pass


class PySceneDetectAdapter(SceneAdapter):
    """Shot detection via PySceneDetect's content detector (M3b).

    ``detect_shots`` converts the detected scene boundaries into an
    ordered, contiguous list of ``Shot(start, end)`` covering
    ``[0, duration_s]``. When no cut is detected it returns a single
    shot spanning the full duration.
    """

    def detect_shots(self, video_path: str, duration_s: float) -> list[Shot]:
        if duration_s <= 0:
            raise SceneError(f"Invalid duration: {duration_s}")
        if not os.path.exists(video_path):
            raise SceneError(f"Video file does not exist: {video_path}")

        # Lazy imports: scenedetect/cv2 are heavy and must not be imported
        # at module load time. cv2 must be the headless build
        # (opencv-python-headless, never opencv-python).
        try:
            import cv2
            from scenedetect import ContentDetector, detect
        except ImportError as e:
            raise SceneError(
                "PySceneDetect is unavailable: install 'scenedetect' and "
                "'opencv-python-headless' (never 'opencv-python') to enable "
                "shot detection"
            ) from e

        if getattr(cv2, "__version__", None) is None:
            raise SceneError("opencv-python-headless is installed but failed to load")

        try:
            scene_list = detect(
                str(video_path),
                ContentDetector(threshold=SHOT_THRESHOLD),
                show_progress=False,
            )
        except Exception as e:
            raise SceneError(f"PySceneDetect failed on {video_path}: {e}") from e

        # Scene boundaries = clip starts from the scene list. Build a
        # contiguous, clamped set of edges over [0, duration_s].
        bounds = [0.0]
        for scene, _next in scene_list:
            t = min(max(_timecode_seconds(scene), 0.0), duration_s)
            if t > bounds[-1]:
                bounds.append(t)
        bounds.append(duration_s)

        shots = [
            Shot(start=bounds[i], end=bounds[i + 1])
            for i in range(len(bounds) - 1)
        ]
        if not shots:  # defensive: duration_s > 0 guarantees at least one
            shots = [Shot(start=0.0, end=duration_s)]
        return shots


class FixtureSceneAdapter(SceneAdapter):
    """Canned shot adapter for CPU-only tests.

    Defaults to a single shot spanning the full duration.
    """

    def __init__(self, shots: list[Shot] | None = None):
        self._shots = shots

    def detect_shots(self, video_path: str, duration_s: float) -> list[Shot]:
        if self._shots is not None:
            return [Shot(start=s.start, end=s.end) for s in self._shots]
        return [Shot(start=0.0, end=duration_s)]


_scene_adapter_instance: SceneAdapter | None = None


def get_scene_adapter() -> SceneAdapter:
    """Return the shared ``PySceneDetectAdapter`` instance.

    No heavy imports happen here (``PySceneDetectAdapter`` defers them
    into ``detect_shots``). Tests monkeypatch this factory.
    """
    global _scene_adapter_instance
    if _scene_adapter_instance is None:
        _scene_adapter_instance = PySceneDetectAdapter()
    return _scene_adapter_instance
