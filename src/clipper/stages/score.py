"""
Scoring stage implementation for the clipper pipeline (M5b).

Single ``scoring`` stage over the M4b ``candidates.json``:

- inputs (per-job artifacts from earlier milestones):
    * ``candidates.json`` — M4b candidates stage (``scripts/candidates_job.py``)
    * ``sentences.json``  — M2b transcribe stage  (``scripts/transcribe_job.py``)
    * ``probe.json``      — M1 ingest stage       (``python -m clipper ingest``)
- LLM: ``client.generate_json(SCORING_SYSTEM, batch_json, ScoringResponse)``
  per batch, using the pure helpers in ``clipper.scoring`` (context building,
  greedy batch packing, response validation).
- batch failure policy (docs/spec/m5.md): a batch that still fails after
  ``cfg.llm.max_retries`` retries on ``LLMError``/semantic ``ValueError``
  does NOT fail the job — its candidates are emitted with all axes 0 and
  ``reject_reason="llm_batch_failed: <msg>"``.
- output: ``scores.json`` (``ScoresArtifact``, ``schema_version=1``) with ALL
  candidates in original (artifact) candidate order; updates
  ``candidates.score_json`` for every ``cand_id`` (rows exist since M4b).
- stage skip rule: DB state = ``done`` AND artifact exists AND
  ``schema_version == 1``. Job goes ``queued -> running -> queued`` on
  success; on failure the stage + job are marked ``failed`` and a clean
  ``StageError`` is raised.

No selection, no cut refinement, no metadata (M5c+).
"""

import json
from pathlib import Path

from ..adapters.llm import LLMClient, LLMError
from ..config import Config
from ..models import (
    CandidateContext,
    CandidatesArtifact,
    ProbeResult,
    ScoredClip,
    ScoresArtifact,
    ScoringBatch,
    ScoringResponse,
    SentencesArtifact,
)
from ..prompts import SCORING_SYSTEM
from ..scoring import build_context, pack_batches, validate_scores
from . import StageError
from .analyze import (
    _finish_stage,
    _is_done,
    _set_job_state,
    _start_stage,
)

_SCHEMA_VERSION = 1
_STAGE_NAME = "scoring"
_ARTIFACT_NAME = "scores.json"

# (filename, expected model, producer hint for clean errors)
_REQUIRED_INPUTS = (
    (
        "candidates.json",
        CandidatesArtifact,
        "produced by the M4b candidates stage (scripts/candidates_job.py)",
    ),
    (
        "sentences.json",
        SentencesArtifact,
        "produced by the M2b transcribe stage (scripts/transcribe_job.py)",
    ),
    (
        "probe.json",
        ProbeResult,
        "produced by the M1 ingest stage (python -m clipper ingest)",
    ),
)


def _load_inputs(job_dir: Path) -> tuple[list, list]:
    """Load + schema-validate the required M1/M2/M4 inputs.

    Returns ``(candidates, sentences)`` as typed model lists. Raises a clean
    ``StageError`` naming the offending artifact (and, for missing files,
    the milestone stage/script that produces it).
    """
    payload: dict[str, object] = {}
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

    candidates = payload["candidates.json"].candidates  # type: ignore[union-attr]
    sentences = payload["sentences.json"].sentences  # type: ignore[union-attr]
    return candidates, sentences


def _write_artifact(artifact_path: Path, artifact: ScoresArtifact) -> None:
    try:
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2) + "\n")
    except OSError as e:
        raise StageError(f"Failed to write artifact {artifact_path}: {e}") from e


def _rejected_clip(cand: CandidateContext, reason_prefix: str, msg: str) -> ScoredClip:
    """Synthetic all-zero score emitted when a batch exhausts its retries."""
    return ScoredClip(
        cand_id=cand.cand_id,
        hook=0,
        payoff=0,
        self_contained=0,
        emotion=0,
        clarity=0,
        suggested_in=cand.start,
        suggested_out=cand.end,
        reject_reason=f"{reason_prefix}: {msg}",
    )


def _score_batch(
    client: LLMClient,
    batch: ScoringBatch,
    max_retries: int,
) -> list[ScoredClip]:
    """Score one batch; retry on LLMError / semantic ValueError.

    After ``max_retries`` retries the batch's candidates are REJECTED (not a
    job failure) with all axes 0 and ``reject_reason="llm_batch_failed: ..."``.
    """
    user_payload = batch.model_dump_json()
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = client.generate_json(
                SCORING_SYSTEM, user_payload, ScoringResponse
            )
            return validate_scores(response, batch)
        except (LLMError, ValueError) as e:
            last_error = e
    # Retries exhausted: reject this batch's candidates, never fail the job.
    msg = str(last_error)
    return [
        _rejected_clip(cand, "llm_batch_failed", msg) for cand in batch.candidates
    ]


def run_scoring(job_id: str, cfg: Config, conn, client: LLMClient) -> dict:
    """Run the ``scoring`` stage for a job.

    Args:
        job_id: The job ID (must exist in the DB, with M4b candidates rows).
        cfg: Configuration object (data_dir, llm.*).
        conn: sqlite3 connection over the clipper database.
        client: LLM client (real ``OllamaLLMClient`` or ``MockLLM``).

    Returns:
        Summary dict: job_id, model, batches, scored, rejected, skipped,
        artifact (path to scores.json).

    Raises:
        StageError: Clean error on any failure (missing job/inputs, schema
            mismatch, write failure). The stage and job are marked failed.
    """
    row = conn.execute(
        "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    artifact_path = job_dir / _ARTIFACT_NAME

    summary: dict = {
        "job_id": job_id,
        "model": cfg.llm.model,
        "batches": 0,
        "scored": 0,
        "rejected": 0,
        "skipped": False,
        "artifact": str(artifact_path),
    }

    skip, stored = _is_done(conn, job_id, _STAGE_NAME)
    if skip:
        with open(stored, encoding="utf-8") as f:
            data = json.load(f)
        scores = data.get("scores", [])
        summary["scored"] = len(scores)
        summary["rejected"] = sum(1 for s in scores if s.get("reject_reason"))
        summary["skipped"] = True
        return summary

    _set_job_state(conn, job_id, "running")
    try:
        _start_stage(conn, job_id, _STAGE_NAME)

        candidates, sentences = _load_inputs(job_dir)

        contexts = [build_context(cand, sentences, cfg) for cand in candidates]
        batches = pack_batches(
            contexts, cfg.llm.batch_size, cfg.llm.max_batch_tokens
        )
        summary["batches"] = len(batches)

        scored_by_id: dict[str, ScoredClip] = {}
        for batch in batches:
            for s in _score_batch(client, batch, cfg.llm.max_retries):
                scored_by_id[s.cand_id] = s

        # ALL candidates, in original (artifact) candidate order.
        ordered = [scored_by_id[cand.cand_id] for cand in candidates]

        artifact = ScoresArtifact(
            schema_version=_SCHEMA_VERSION, model=cfg.llm.model, scores=ordered
        )
        _write_artifact(artifact_path, artifact)

        for s in ordered:
            conn.execute(
                """
                UPDATE candidates
                SET score_json = ?
                WHERE job_id = ? AND cand_id = ?
                """,
                (s.model_dump_json(), job_id, s.cand_id),
            )
        conn.commit()

        _finish_stage(conn, job_id, _STAGE_NAME, str(artifact_path))
        _set_job_state(conn, job_id, "queued")

        summary["scored"] = len(ordered)
        summary["rejected"] = sum(
            1 for s in ordered if s.reject_reason is not None
        )
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"Scoring stage failed: {e}"
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


__all__ = ["run_scoring"]
