"""
Shot-detection adapter protocol and fixture for clipper (M3a).

PySceneDetectAdapter is intentionally NOT implemented yet (M3b).
M3b will use PySceneDetect ContentDetector with SHOT_THRESHOLD and MUST
depend on opencv-python-headless (never opencv-python; headless server).
"""

from abc import ABC, abstractmethod

from ..models import Shot

# PySceneDetect ContentDetector threshold (see docs/spec/m3.md).
SHOT_THRESHOLD = 27.0


class SceneAdapter(ABC):
    """Abstract interface for shot detection."""

    @abstractmethod
    def detect_shots(self, video_path: str, duration_s: float) -> list[Shot]:
        """Detect shot boundaries in a video, within [0, duration_s]."""
        pass


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
