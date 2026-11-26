"""M8 worker report contracts (additive).

Pure CPU data contracts for the M8 orchestration layer. Nothing here
runs a worker loop — no I/O, no adapters, no GPU. See
`docs/spec/m8.md` (M8a decisions).

- `StageReport` — one coarse worker stage's outcome (name, its
  required fine-grained DB stages, state, timestamps, artifacts, error).
  Embedded element: it carries NO `schema_version` of its own (the
  owning `JobReport` is versioned).
- `JobReport` — the per-job `report.json` record (M8): stage timings,
  artifacts, clip count, errors. `schema_version=1`.
"""

from pydantic import BaseModel


class StageReport(BaseModel):
    """One coarse worker stage within a JobReport (M8a)."""

    name: str
    required: list[str]
    state: str
    started_at: str | None = None
    finished_at: str | None = None
    artifact_paths: list[str] = []
    error: str | None = None


class JobReport(BaseModel):
    """Per-job report.json record (M8a, schema_version=1)."""

    schema_version: int = 1
    job_id: str
    state: str
    clip_count: int
    stages: list[StageReport] = []
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
