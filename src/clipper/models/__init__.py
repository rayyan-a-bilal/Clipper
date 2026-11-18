"""
Data models for clipper pipeline.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


# Basic data structures
class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float

class Sentence(BaseModel):
    start: float
    end: float
    text: str
    word_idx: tuple[int, int]

class Shot(BaseModel):
    start: float
    end: float

class FaceTrack(BaseModel):
    track_id: int
    boxes: list[tuple[float, float, float, float, float]]  # t, x, y, w, h normalized
    persistence: float
    corner_affinity: Literal["tl", "tr", "bl", "br"] | None

class Layout(StrEnum):
    center_crop = "center_crop"
    face_track = "face_track"
    split_screen = "split_screen"
    facecam = "facecam"

class CropKey(BaseModel):
    t: float
    x: int
    y: int
    w: int
    h: int

# Pipeline artifacts - M1 contracts
class IngestRecord(BaseModel):
    job_id: str
    source_path: str
    content_hash: str
    client: str
    niche: str
    schema_version: int = 1

class ProbeResult(BaseModel):
    duration_s: float
    width: int
    height: int
    fps: float
    vcodec: str
    acodec: str | None
    has_audio: bool
    size_bytes: int
    schema_version: int = 1

class ProxyInfo(BaseModel):
    width: int
    height: int
    fps: float
    path: str
    schema_version: int = 1

class AudioInfo(BaseModel):
    sample_rate: int
    channels: int
    duration_s: float
    path: str
    schema_version: int = 1

# Pipeline stages - M0 contracts
class Candidate(BaseModel):
    cand_id: str
    start: float
    end: float
    sentence_range: tuple[int, int]
    cheap_score: float

class ScoredClip(BaseModel):
    cand_id: str
    hook: int
    payoff: int
    self_contained: int
    emotion: int
    clarity: int
    suggested_in: float
    suggested_out: float
    reject_reason: str | None = None

class ClipPlan(BaseModel):
    clip_id: str
    job_id: str
    start: float
    end: float
    layout: Layout
    crop_path: list[CropKey]
    weighted_score: float
    title: str
    hook_overlay: str
    description: str
    hashtags: list[str]
    status: Literal[
        "pending_qc", "approved", "rejected", "rendered", "exported"
    ] = "pending_qc"

class Job(BaseModel):
    job_id: str
    source_path: str
    client: str
    niche: str
    state: Literal["queued", "running", "awaiting_qc", "done", "failed"]
    stages: dict[str, str]


# Re-export transcript artifacts (defined in transcript_artifacts.py, which
# Re-export artifacts (defined in transcript_artifacts.py, which imports the
# base classes above, so this must stay at the end of the module)
from .transcript_artifacts import SentencesArtifact as SentencesArtifact  # noqa: E402
from .transcript_artifacts import WordsArtifact as WordsArtifact  # noqa: E402
