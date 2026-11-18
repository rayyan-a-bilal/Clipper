"""
Analysis artifacts for clipper pipeline (M3a — shot/silence/energy).

Pure CPU data contracts; no detector logic lives here.
"""

from pydantic import BaseModel

from . import Shot


class SilenceRegion(BaseModel):
    """A contiguous silent interval (seconds, relative to audio start)."""

    start: float
    end: float


class EnergyWindow(BaseModel):
    """RMS energy over one non-overlapping audio window (seconds)."""

    start: float
    end: float
    rms_db: float


class ShotsArtifact(BaseModel):
    """shots.json artifact (schema_version=1)."""

    schema_version: int = 1
    shots: list[Shot]


class SilenceArtifact(BaseModel):
    """silence.json artifact (schema_version=1)."""

    schema_version: int = 1
    noise_db: float = -35.0
    min_dur: float = 0.5
    regions: list[SilenceRegion]


class EnergyArtifact(BaseModel):
    """energy.json artifact (schema_version=1)."""

    schema_version: int = 1
    window_s: float = 0.5
    windows: list[EnergyWindow]
