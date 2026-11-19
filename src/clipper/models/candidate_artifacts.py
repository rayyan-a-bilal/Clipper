"""
Candidate artifacts for clipper pipeline (M4a — candidate generation).

Pure CPU data contract; no candidate logic lives here. Re-exports the
existing `Candidate` base model.
"""

from pydantic import BaseModel

from . import Candidate


class CandidatesArtifact(BaseModel):
    """candidates.json artifact (schema_version=1).

    Holds the pre-ranked candidate windows to be scored by the LLM (M5).
    `cheap_score` is a PRE-RANK, not a quality verdict.
    """

    schema_version: int = 1
    candidates: list[Candidate]
