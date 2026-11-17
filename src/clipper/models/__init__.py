"""
Data models for clipper pipeline.
"""

from typing import Literal, Tuple
from pydantic import BaseModel
from enum import Enum

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
    word_idx: Tuple[int, int]

class Shot(BaseModel):
    start: float
    end: float

class FaceTrack(BaseModel):
    track_id: int
    boxes: list[tuple[float, float, float, float, float]]  # t, x, y, w, h normalized
    persistence: float
    corner_affinity: Literal["tl", "tr", "bl", "br"] | None

class Layout(str, Enum):
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

# Pipeline artifacts
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
    status: Literal["pending_qc", "approved", "rejected", "rendered", "exported"] = "pending_qc"

class Job(BaseModel):
    job_id: str
    source_path: str
    client: str
    niche: str
    state: Literal["queued", "running", "awaiting_qc", "done", "failed"]
    stages: dict[str, str]