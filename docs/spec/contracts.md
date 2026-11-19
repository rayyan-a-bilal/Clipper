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