"""Scoring contracts for the M5 LLM scoring stage (M5a — additive).

Pure CPU data contracts; no scoring / selection / cut logic lives here.
Existing models in `clipper.models` are re-exported, not modified.

- `CandidateContext` — one candidate row of a scoring batch (the LLM-facing
  context: window, transcript span, ±10s surrounding context).
- `ScoringBatch` — the request payload handed to the LLM (up to 5 candidates).
- `ScoringResponse` — exactly what the model must return: a list of
  `ScoredClip` (one per candidate in the batch).
- `ScoresArtifact` — the job artifact persisted by the M5b scoring stage.
"""

from pydantic import BaseModel, Field

from . import ScoredClip


class CandidateContext(BaseModel):
    """One candidate with its transcript context (LLM-facing view)."""

    cand_id: str
    start: float
    end: float
    transcript: str
    context_before: str = ""
    context_after: str = ""


class ScoringBatch(BaseModel):
    """Scoring batch payload (schema_version=1).

    At most `cfg.llm.batch_size` (5) candidates per batch.
    """

    schema_version: int = 1
    candidates: list[CandidateContext] = Field(max_length=5)


class ScoringResponse(BaseModel):
    """What the model returns: one scored entry per candidate in the batch."""

    scores: list[ScoredClip]


class ScoresArtifact(BaseModel):
    """scores.json artifact (schema_version=1), persisted by the M5b stage."""

    schema_version: int = 1
    model: str
    scores: list[ScoredClip]
