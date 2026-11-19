"""
Candidates stage implementation for the clipper pipeline (M4b).

Wires the M4a pure generator (``clipper.candidates.generate_candidates``)
into the pipeline as the single ``candidates`` stage:

- inputs (per-job artifacts from earlier milestones):
    * ``sentences.json`` — M2b transcribe stage (``scripts/transcribe_job.py``)
    * ``energy.json``    — M3b analysis stage  (``scripts/analyze_job.py``)
    * ``silence.json``   — M3b analysis stage  (``scripts/analyze_job.py``)
- output: ``candidates.json`` (``CandidatesArtifact``, ``schema_version=1``)
- upserts each candidate into the existing ``candidates`` table with
  ``score_json = NULL`` (M5 fills this in later)

A stage is skipped only when the DB stage state is ``done`` AND the artifact
file exists AND its JSON has ``schema_version == 1``. On success the job
returns to ``queued``; on failure the stage is marked ``failed`` (with the
error text), the job is set to ``failed``, and a clean ``StageError`` is
raised. Zero candidates is a VALID success.

No LLM scoring, no selection, no cut snapping (M5). CPU-only; no I/O beyond
the job artifacts and the SQLite connection (stage bookkeeping lives in
``stages/analyze.py`` and is reused here).

Contracts and rules: docs/spec/m4.md.
"""

import json
from pathlib import Path

from ..candidates import generate_candidates
from ..config import Config
from ..models import (
    CandidatesArtifact,
    EnergyArtifact,
    SentencesArtifact,
    SilenceArtifact,
)
from . import StageError
from .analyze import (
    _finish_stage,
    _is_done,
    _set_job_state,
    _start_stage,
)

_SCHEMA_VERSION = 1
_STAGE_NAME = "candidates"
_ARTIFACT_NAME = "candidates.json"

# (filename, expected model, producer hint for clean errors)
_REQUIRED_INPUTS = (
    (
        "sentences.json",
        SentencesArtifact,
        "produced by the M2b transcribe stage (scripts/transcribe_job.py)",
    ),
    (
        "energy.json",
        EnergyArtifact,
        "produced by the M3b analysis stage (scripts/analyze_job.py)",
    ),
    (
        "silence.json",
        SilenceArtifact,
        "produced by the M3b analysis stage (scripts/analyze_job.py)",
    ),
)


def _resolve_niche_keywords(cfg: Config, niche: str | None) -> list[str]:
    """Resolve ``clips.niche_keywords[job.niche]`` to a flat keyword list.

    Empty list when the job has no niche or no mapping is configured (the
    pure generator then falls back to the neutral 0.5 niche score).
    """
    mapping = cfg.clips.niche_keywords or {}
    if not niche:
        return []
    keywords = mapping.get(niche) or []
    return [k for k in keywords if isinstance(k, str) and k.strip()]


def _load_inputs(job_dir: Path) -> tuple[list, list, list]:
    """Load and validate the required M2/M3 input artifacts.

    Returns ``(sentences, energy_windows, silence_regions)`` as typed model
    lists. Raises a clean ``StageError`` naming the offending artifact (and,
    for missing files, the milestone stage/script that produces it).
    """
    payload = {}
    for name, model, hint in _REQUIRED_INPUTS:
        path = job_dir / name
        if not path.exists():
            raise StageError(
                f"Missing required input {path} — {hint}; run that stage first"
            )
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise StageError(f"Failed to read {name}: {e}") from e

        version = data.get("schema_version") if isinstance(data, dict) else None
        if version != _SCHEMA_VERSION:
            raise StageError(
                f"{name} has schema_version={version!r} "
                f"(expected {_SCHEMA_VERSION}) — {hint}"
            )
        try:
            artifact = model.model_validate(data)
        except Exception as e:
            raise StageError(f"{name} is not a valid artifact: {e}") from e
        payload[name] = artifact

    return (
        payload["sentences.json"].sentences,
        payload["energy.json"].windows,
        payload["silence.json"].regions,
    )


def _write_artifact(artifact_path: Path, artifact: CandidatesArtifact) -> None:
    try:
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2) + "\n")
    except OSError as e:
        raise StageError(f"Failed to write artifact {artifact_path}: {e}") from e


def _upsert_candidates(conn, job_id: str, candidates) -> None:
    """Replace the job's candidate rows (score_json = NULL, M5 fills it)."""
    conn.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
    for cand in candidates:
        conn.execute(
            """
            INSERT OR REPLACE INTO candidates
                (cand_id, job_id, candidate_json, score_json)
            VALUES (?, ?, ?, NULL)
            """,
            (cand.cand_id, job_id, cand.model_dump_json()),
        )
    conn.commit()


def run_candidates(job_id: str, cfg: Config, conn) -> dict:
    """Run the ``candidates`` stage for a job.

    Args:
        job_id: The job ID (must exist in the DB).
        cfg: Configuration object (data_dir, clips.niche_keywords).
        conn: sqlite3 connection over the clipper database.

    Returns:
        Summary dict: job_id, candidates (count), skipped (bool), artifact
        (artifact path).

    Raises:
        StageError: Clean, single-line error on any failure (missing job,
            missing/invalid M2/M3 input, generation or write failure). The
            stage and the job are marked failed in the DB.
    """
    row = conn.execute(
        "SELECT niche FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")
    niche = row[0]

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    artifact_path = job_dir / _ARTIFACT_NAME

    summary: dict = {
        "job_id": job_id,
        "candidates": 0,
        "skipped": False,
        "artifact": str(artifact_path),
    }

    skip, stored = _is_done(conn, job_id, _STAGE_NAME)
    if skip:
        with open(stored, encoding="utf-8") as f:
            data = json.load(f)
        summary["candidates"] = len(data.get("candidates", []))
        summary["skipped"] = True
        return summary

    _set_job_state(conn, job_id, "running")
    try:
        _start_stage(conn, job_id, _STAGE_NAME)

        sentences, windows, regions = _load_inputs(job_dir)
        niche_keywords = _resolve_niche_keywords(cfg, niche)

        candidates = generate_candidates(
            sentences,
            windows,
            regions,
            job_id=job_id,
            niche_keywords=niche_keywords,
        )
        artifact = CandidatesArtifact(
            schema_version=_SCHEMA_VERSION, candidates=candidates
        )
        _write_artifact(artifact_path, artifact)
        _upsert_candidates(conn, job_id, candidates)

        _finish_stage(conn, job_id, _STAGE_NAME, str(artifact_path))
        _set_job_state(conn, job_id, "queued")

        summary["candidates"] = len(candidates)
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"Candidates stage failed: {e}"
        )
        try:
            _finish_stage(conn, job_id, _STAGE_NAME, error=str(clean))
        except Exception:
            pass
        try:
            _set_job_state(conn, job_id, "failed")
        except Exception:
            pass
        if clean is e:
            raise
        raise clean from e


__all__ = ["run_candidates"]
