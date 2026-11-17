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