# Data Contracts

## Word
```python
{
    "start": float,
    "end": float,
    "text": str,
    "prob": float
}
```

## Sentence
```python
{
    "start": float,
    "end": float,
    "text": str,
    "word_idx": tuple[int, int]
}
```

## Shot
```python
{
    "start": float,
    "end": float
}
```

## FaceTrack
```python
{
    "track_id": int,
    "boxes": list[tuple[float, float, float, float, float]],  # t, x, y, w, h normalized
    "persistence": float,
    "corner_affinity": Literal["tl", "tr", "bl", "br"] | None
}
```

## Layout enum
```python
center_crop
face_track
face_follow   # M6f — zoom crop that follows a (weaker, below-bar) face;
              # rendered via the same M7e `sendcmd`+`crop` path as face_track
split_screen
facecam
```

## CropKey
```python
{
    "t": float,
    "x": int,
    "y": int,
    "w": int,
    "h": int
}
```

## Candidate
```python
{
    "cand_id": str,
    "start": float,
    "end": float,
    "sentence_range": tuple[int, int],
    "cheap_score": float
}
```

## CandidatesArtifact (M4a, candidates.json)
```python
{
    "schema_version": 1,
    "candidates": list[Candidate]  # pre-ranked via cheap_score (a PRE-RANK, NOT a quality verdict)
}
```

## ScoredClip
```python
{
    "cand_id": str,
    "hook": int,
    "payoff": int,
    "self_contained": int,
    "emotion": int,
    "clarity": int,
    "suggested_in": float,
    "suggested_out": float,
    "reject_reason": str | None = None
}
```

## CandidateContext (M5a)
```python
{
    "cand_id": str,
    "start": float,
    "end": float,
    "transcript": str,
    "context_before": str = "",   # ~10s before window (cfg.llm.context_before_s)
    "context_after": str = ""     # ~10s after window  (cfg.llm.context_after_s)
}
```

## ScoringBatch (M5a, request payload)
```python
{
    "schema_version": 1,
    "candidates": list[CandidateContext]  # up to 5 per batch (cfg.llm.batch_size)
}
```

## ScoringResponse (M5a, exact model output)
```python
{
    "scores": list[ScoredClip]  # one per candidate in the batch, same order
}
```

## ScoresArtifact (M5a, scores.json)
```python
{
    "schema_version": 1,
    "model": str,
    "scores": list[ScoredClip]
}
```

## ClipPlan
```python
{
    "clip_id": str,
    "job_id": str,
    "start": float,
    "end": float,
    "layout": Layout,
    "crop_path": list[CropKey],
    "weighted_score": float,
    "title": str,
    "hook_overlay": str,
    "description": str,
    "hashtags": list[str],
    "status": Literal["pending_qc", "approved", "rejected", "rendered", "exported"] = "pending_qc"
}
```

## Job
```python
{
    "job_id": str,
    "source_path": str,
    "client": str,
    "niche": str,
    "state": Literal["queued", "running", "awaiting_qc", "done", "failed"],
    "stages": dict[str, str]
}
```
## M6a — vision & layout contracts (`models/vision_artifacts.py`, see `docs/spec/m6.md`)

```python
# VisionSegmentInfo — one rendered per-clip vision segment (960x540 fit,
# aspect preserved, no padding, even dims; fps = min(native, 30), native
# 0/unknown -> 30). NOT a persisted artifact (no schema_version).
{
    "clip_id": str,
    "path": str,
    "width": int,
    "height": int,
    "fps": float,
    "frame_count": int,
    "source_start": float,   # clip time in the SOURCE video
    "source_end": float
}

# FaceDetection — one detection in a vision frame (x/y/w/h normalized
# 0..1 relative to the vision frame; t segment-local seconds).
{
    "t": float, "x": float, "y": float, "w": float, "h": float, "score": float
}

# RegionTrack — keys are M0 CropKey: t CLIP-RELATIVE (t=0 at clip start)
# at NATIVE fps, coordinates in SOURCE pixels.
{
    "role": Literal["main", "top", "bottom", "face"],
    "keys": list[CropKey]
}

# ClipLayoutArtifact — per-clip layout artifact (data/jobs/<job>/clips/<clip>/).
{
    "schema_version": 1,
    "clip_id": str,
    "layout": Layout,
    "source_width": int,
    "source_height": int,
    "native_fps": float,
    "tracks": list[RegionTrack],
    "face_track_count": int,
    "detector": str,
    "layout_source": Literal["auto", "manual"]
}
```

## M6c — track summary contract (`clipper.models`, see `docs/spec/m6.md` M6c)

```python
# TrackSummary — per-clip in-memory summary from tracking.summarize over
# FaceTrack lists. NOT a persisted artifact (no schema_version, mirrors
# VisionSegmentInfo); persistence/corner stats computed by tracking.track_stats.
{
    "dominant_track_id": Optional[int],          # highest persistence, tie-break larger mean_area
    "persistent_track_ids": list[int],           # persistence >= 0.6
    "facecam_candidate_track_ids": list[int],    # corner_affinity set AND persistence >= 0.8 AND mean_area < 0.04
    "side_by_side_pair": Optional[tuple[int, int]],  # two persistent tracks, horizontally separated, mean_area within 2x
    "dominant_persistence": float                # M6f — dominant track's persistence (0.0 empty); drives face_follow auto-selection
}
```
