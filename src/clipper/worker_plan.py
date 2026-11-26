"""M8a — coarse worker stage plan (PURE: plan data + DB/file queries only).

No worker loop, no stage execution, no adapters, no network. See
`docs/spec/m8.md` (M8a decisions).

The coarse stages orchestrate the EXISTING fine-grained stages
(M1–M7); they are never reimplemented here. A coarse stage is "done"
only when ALL of its required fine-grained DB stages are done AND
their recorded artifacts exist AND (for JSON artifacts)
`schema_version == 1`.

Note on stage states: the M1/M2 stages (`ingest`, `probe`, `proxy`,
`audio`, `transcribe`, `sentences`) record the terminal state as
`"completed"`; M3+ stages record `"done"`. Both are accepted as done
(M8a judgment note — recorded in `docs/spec/m8.md`).
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

#: Stage-state values that count as "done" (see module docstring).
DONE_STATES = frozenset({"done", "completed"})


@dataclass(frozen=True)
class StageSpec:
    """One coarse worker stage.

    - `name` — the coarse stage name.
    - `required` — the fine-grained DB stage names this coarse stage
      orchestrates (existing M1–M7 stages; never reimplemented here).
    - `artifacts` — the canonical artifact names (relative to the job
      directory) this coarse stage needs in place.
    """

    name: str
    required: tuple[str, ...]
    artifacts: tuple[str, ...]


#: Ordered coarse-stage pipeline (M8a). Order is exact and tested.
STAGE_PLAN: tuple[StageSpec, ...] = (
    StageSpec(
        name="prep",
        required=("ingest", "probe", "proxy", "audio"),
        artifacts=("ingest.json", "probe.json", "proxy.mp4", "audio.wav"),
    ),
    StageSpec(
        name="transcribe",
        required=("transcribe", "sentences"),
        artifacts=("words.json", "sentences.json"),
    ),
    StageSpec(
        name="analyze",
        required=("shots", "silence", "energy"),
        artifacts=("shots.json", "silence.json", "energy.json"),
    ),
    StageSpec(
        name="candidates",
        required=("candidates",),
        artifacts=("candidates.json",),
    ),
    StageSpec(
        name="scoring",
        required=("scoring",),
        artifacts=("scores.json",),
    ),
    StageSpec(
        name="selection",
        required=("selection",),
        artifacts=("selection.json",),
    ),
    StageSpec(
        name="clip_plans",
        required=("clip_plans",),
        artifacts=("clip_plans.json",),
    ),
    # Per-clip artifacts under `clips/<clip_id>/`: the coarse skip check
    # uses the existing `clips` directory (M8a judgment note); the M8
    # worker deep-validates each clip's layout.json / render.json.
    StageSpec(name="layout", required=("layout",), artifacts=("clips",)),
    StageSpec(name="render", required=("render",), artifacts=("clips",)),
)


def _spec_by_name(stage_name: str) -> StageSpec:
    for spec in STAGE_PLAN:
        if spec.name == stage_name:
            return spec
    raise ValueError(
        f"unknown coarse stage {stage_name!r}; "
        f"expected one of {[s.name for s in STAGE_PLAN]}"
    )


def required_artifacts(stage_name: str) -> tuple[str, ...]:
    """The canonical artifact names for one coarse stage.

    Raises `ValueError` for a name that is not a coarse stage in
    `STAGE_PLAN`.
    """
    return _spec_by_name(stage_name).artifacts


def _fine_stage_names() -> tuple[str, ...]:
    """All fine-grained stage names, in pipeline order."""
    names: list[str] = []
    for spec in STAGE_PLAN:
        for name in spec.required:
            if name not in names:
                names.append(name)
    return tuple(names)


def is_coarse_done(
    conn: sqlite3.Connection, job_id: str, spec: StageSpec
) -> bool:
    """True iff EVERY required fine-grained stage of `spec` is satisfied:

    1. its DB stage row exists with a done state (`done` or
       `completed` — see module docstring),
    2. it recorded a non-empty `artifact_path` whose file/directory
       still exists,
    3. for `*.json` artifacts: the JSON is valid and
       `schema_version == 1`.
    """
    cursor = conn.cursor()
    for name in spec.required:
        cursor.execute(
            "SELECT state, artifact_path "
            "FROM stages WHERE job_id = ? AND name = ?",
            (job_id, name),
        )
        row = cursor.fetchone()
        if row is None or row[0] not in DONE_STATES or not row[1]:
            return False
        path = Path(row[1])
        if not path.exists():
            return False
        if path.suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                return False
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                return False
    return True


def next_coarse_stage(
    conn: sqlite3.Connection, job_id: str
) -> StageSpec | None:
    """The first coarse stage in `STAGE_PLAN` that is not yet done.

    Returns `None` when every coarse stage of the job is done — the
    job's pipeline work is complete.
    """
    for spec in STAGE_PLAN:
        if not is_coarse_done(conn, job_id, spec):
            return spec
    return None


def failed_fine_stages(conn: sqlite3.Connection, job_id: str) -> list[str]:
    """Fine-grained stage names currently `failed` for this job.

    Ordered by pipeline position (`STAGE_PLAN` order), not by DB row
    order. Empty list when nothing failed.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM stages WHERE job_id = ? AND state = 'failed'",
        (job_id,),
    )
    failed = {row[0] for row in cursor.fetchall()}
    return [name for name in _fine_stage_names() if name in failed]
