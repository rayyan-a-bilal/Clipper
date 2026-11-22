"""
Vision & layout contracts for clipper pipeline (M6a — per-clip vision
segments and contracts, docs/spec/m6.md).

Pure CPU data contracts. No segment creation, face detection, tracking,
or layout decision lives here (M6b+).
"""

from typing import Literal

from pydantic import BaseModel

from . import CropKey, Layout


class VisionSegmentInfo(BaseModel):
    """One rendered per-clip vision segment (960x540 fit, <= 30fps,
    aspect preserved, no padding, even dims — docs/spec/m6.md).

    Not a persisted artifact (no `schema_version`, matching the
    `ProxyInfo`/`AudioInfo` pattern). `source_start`/`source_end` are the
    clip's SOURCE-video times; `t`/frames inside the segment are
    segment-local.
    """

    clip_id: str
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    source_start: float
    source_end: float


class FaceDetection(BaseModel):
    """One face detection in a vision frame.

    `x`/`y`/`w`/`h` are normalized 0..1 relative to the vision frame;
    `t` is the vision-segment-local timestamp (seconds).
    """

    t: float
    x: float
    y: float
    w: float
    h: float
    score: float


class RegionTrack(BaseModel):
    """A cropped-region track over one clip (docs/spec/m6.md).

    `keys` are the M0 `CropKey`: timestamps CLIP-RELATIVE (t=0 at clip
    start) emitted at NATIVE video fps (linear interpolation in M6b),
    coordinates in SOURCE pixels.
    """

    role: Literal["main", "top", "bottom", "face"]
    keys: list[CropKey]


class ClipLayoutArtifact(BaseModel):
    """Per-clip layout artifact, written to
    `data/jobs/<job_id>/clips/<clip_id>/` by the M6e layout stage
    (`src/clipper/stages/layout.py`, schema_version=1).

    `layout_source` is `"auto"` (M6 decision) or `"manual"` (QC override,
    M9). `detector` records which detector produced the face tracks (e.g.
    "yunet-cuda" / "yunet-cpu"); empty for center_crop-only clips.
    """

    schema_version: int = 1
    clip_id: str
    layout: Layout
    source_width: int
    source_height: int
    native_fps: float
    tracks: list[RegionTrack]
    face_track_count: int
    detector: str
    layout_source: Literal["auto", "manual"]
