"""M7 render contracts (additive).

Pure CPU data contracts for the (future M7b+) render stage. Nothing here
performs rendering — no ffmpeg, no GPU. See `docs/spec/m7.md` M7a decisions.

- `MezzanineInfo` — one rendered 1080x1920 mezzanine segment (native fps);
  mirrors the `ProxyInfo`/`AudioInfo` pattern (in-memory info,
  schema_version=1).
- `ClipAudioInfo` — the extracted/loudnorm'd per-clip audio (`audio.m4a`).
- `RenderArtifact` — the `render.json` artifact: the full per-clip render
  record (paths + geometry + encoder/hwaccel + timing), schema_version=1.
"""

from pydantic import BaseModel


class MezzanineInfo(BaseModel):
    """One rendered mezzanine segment (1080x1920, native fps)."""

    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    schema_version: int = 1


class ClipAudioInfo(BaseModel):
    """The extracted/loudnorm'd per-clip audio track."""

    path: str
    sample_rate: int
    channels: int
    duration_s: float
    schema_version: int = 1


class RenderArtifact(BaseModel):
    """render.json artifact (schema_version=1).

    One per rendered clip; paths are under
    `data/jobs/<job_id>/clips/<clip_id>/` (mezzanine.mp4, audio.m4a,
    captions.ass, final.mp4, thumbnail.jpg, render.json — docs/spec/m7.md
    decision 4).
    """

    schema_version: int = 1
    clip_id: str
    mezzanine_path: str
    audio_path: str
    captions_path: str
    final_path: str
    thumbnail_path: str
    width: int
    height: int
    fps: float
    duration_s: float
    encoder: str
    hwaccel: str
    render_seconds: float
