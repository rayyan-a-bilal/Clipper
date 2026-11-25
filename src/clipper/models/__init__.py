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

class TrackSummary(BaseModel):
    """Summary of face tracks for one clip (M6c tracking pure logic).

    Not a persisted artifact (no `schema_version`); computed in memory
    by `clipper.tracking.summarize` from the per-clip face tracks.
    `dominant_persistence` (M6f) is the dominant track's persistence
    (0.0 for the empty case).
    """
    dominant_track_id: int | None
    persistent_track_ids: list[int]
    facecam_candidate_track_ids: list[int]
    side_by_side_pair: tuple[int, int] | None
    dominant_persistence: float = 0.0

class Layout(StrEnum):
    center_crop = "center_crop"
    face_track = "face_track"
    face_follow = "face_follow"  # M6f — zoom crop that follows a (weaker) face
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


# Re-export artifacts (defined in *_artifacts.py, which import the base
# classes above, so these must stay at the end of the module)
from .analysis_artifacts import EnergyArtifact as EnergyArtifact  # noqa: E402
from .analysis_artifacts import EnergyWindow as EnergyWindow  # noqa: E402
from .analysis_artifacts import ShotsArtifact as ShotsArtifact  # noqa: E402
from .analysis_artifacts import SilenceArtifact as SilenceArtifact  # noqa: E402
from .analysis_artifacts import SilenceRegion as SilenceRegion  # noqa: E402
from .candidate_artifacts import CandidatesArtifact as CandidatesArtifact  # noqa: E402
from .render_artifacts import ClipAudioInfo as ClipAudioInfo  # noqa: E402
from .render_artifacts import MezzanineInfo as MezzanineInfo  # noqa: E402
from .render_artifacts import RenderArtifact as RenderArtifact  # noqa: E402
from .scoring_artifacts import CandidateContext as CandidateContext  # noqa: E402
from .scoring_artifacts import ScoresArtifact as ScoresArtifact  # noqa: E402
from .scoring_artifacts import ScoringBatch as ScoringBatch  # noqa: E402
from .scoring_artifacts import ScoringResponse as ScoringResponse  # noqa: E402
from .selection_artifacts import ClipDecision as ClipDecision  # noqa: E402
from .selection_artifacts import ClipMetadata as ClipMetadata  # noqa: E402
from .selection_artifacts import ClipPlansArtifact as ClipPlansArtifact  # noqa: E402
from .selection_artifacts import MetadataResponse as MetadataResponse  # noqa: E402
from .selection_artifacts import SelectionArtifact as SelectionArtifact  # noqa: E402
from .transcript_artifacts import SentencesArtifact as SentencesArtifact  # noqa: E402
from .transcript_artifacts import WordsArtifact as WordsArtifact  # noqa: E402
from .vision_artifacts import ClipLayoutArtifact as ClipLayoutArtifact  # noqa: E402
from .vision_artifacts import FaceDetection as FaceDetection  # noqa: E402
from .vision_artifacts import RegionTrack as RegionTrack  # noqa: E402
from .vision_artifacts import VisionSegmentInfo as VisionSegmentInfo  # noqa: E402
