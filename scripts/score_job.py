#!/usr/bin/env python3
"""
Manual scoring script for clipper jobs (M5b).

Runs the M5b scoring stage (LLM batch scoring of M4b candidates) on an
already-candidated job. Mirrors scripts/candidates_job.py.

Default path:
    ONE managed ephemeral Ollama process (docs/spec/llm.md) serves all
    batches of the run and is killed again at the end (success AND failure).

Mock path (--mock):
    WIRING TEST ONLY — a deterministic synthetic scorer (NOT a model, never
    a real judgement). For each candidate at position *j* within its batch it
    returns the same value on every axis: 10 - (2*j) % 11  (i.e. 10, 8, 6,
    4, 2, 10, ...), with suggested_in = candidate.start and
    suggested_out = candidate.end.

Usage:
    python scripts/score_job.py <job-id>
    python scripts/score_job.py <job-id> --mock
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from clipper.adapters.llm import LLMClient, get_llm_client
from clipper.config import Config
from clipper.db import Database
from clipper.llm_runtime import managed_ollama
from clipper.models import (
    Candidate,
    ScoredClip,
    ScoringBatch,
    ScoringResponse,
)
from clipper.scoring import weighted_score
from clipper.stages.score import run_scoring

_TOP_N = 5


class _MockScorer:
    """--mock deterministic stand-in (wiring test only, never real judgement).

    Derives every axis from the candidate's ORDER WITHIN ITS BATCH (position
    j -> value ``10 - (2*j) % 11`` on all five axes) and proposes the
    candidate window itself as suggested_in/out. Implements the ``LLMClient``
    protocol; zero network I/O.
    """

    def generate_json(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        batch = ScoringBatch.model_validate_json(user)
        scores: list[ScoredClip] = []
        for j, cand in enumerate(batch.candidates):
            value = 10 - (2 * j) % 11
            scores.append(
                ScoredClip(
                    cand_id=cand.cand_id,
                    hook=value,
                    payoff=value,
                    self_contained=value,
                    emotion=value,
                    clarity=value,
                    suggested_in=cand.start,
                    suggested_out=cand.end,
                    reject_reason=None,
                )
            )
        return ScoringResponse(scores=scores)


def _print_top(job_dir: Path, scores: list[ScoredClip]) -> None:
    """Print the top ``_TOP_N`` clips by weighted score (for humans)."""
    cands: dict[str, Candidate] = {}
    candidates_path = job_dir / "candidates.json"
    if candidates_path.exists():
        data = json.loads(candidates_path.read_text(encoding="utf-8"))
        for c in data.get("candidates", []):
            cand = Candidate.model_validate(c)
            cands[cand.cand_id] = cand

    top = sorted(
        scores, key=lambda s: (-weighted_score(s), s.cand_id)
    )[:_TOP_N]
    if not top:
        return
    print(f"top {_TOP_N}:")
    for s in top:
        cand = cands.get(s.cand_id)
        window = (
            f"[{cand.start:.2f}-{cand.end:.2f}]" if cand else f"[{s.cand_id}]"
        )
        print(
            f"  {window} w={weighted_score(s):.2f} "
            f"(h{s.hook}/p{s.payoff}/s{s.self_contained}/"
            f"e{s.emotion}/c{s.clarity}) {s.cand_id}"
            f"{'  [rejected: ' + s.reject_reason + ']' if s.reject_reason else ''}"
        )


def main(job_id: str, use_mock: bool = False) -> int:
    """
    Run the scoring stage for a job and print a short summary.

    Args:
        job_id: The job ID to process.
        use_mock: --mock => deterministic synthetic scores (wiring test only).

    Returns:
        0 on success (including all-rejected), 1 on failure.
    """
    cfg = Config.load_config()

    db = Database(cfg)
    if db.get_job(job_id) is None:
        print(f"Error: job {job_id} not found")
        return 1

    conn = db.get_connection()
    try:
        if use_mock:
            client: LLMClient = _MockScorer()
            summary = run_scoring(job_id, cfg, conn, client)
        else:
            # ONE managed ephemeral Ollama for the whole run (all batches);
            # killed again on success AND failure (docs/spec/llm.md).
            with managed_ollama(cfg) as base_url:
                client = get_llm_client(cfg, base_url)
                summary = run_scoring(job_id, cfg, conn, client)
    except Exception as e:
        print(f"Error: scoring stage failed: {e}")
        return 1
    finally:
        conn.close()

    artifact_path = Path(summary["artifact"])
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    scores = [ScoredClip.model_validate(s) for s in data.get("scores", [])]

    print(f"job:      {job_id}")
    print(f"model:    {summary['model']}")
    print(f"batches:  {summary['batches']}")
    print(f"scored:   {summary['scored']}")
    print(f"rejected: {summary['rejected']}")
    if summary.get("skipped"):
        print("note:     scoring stage already done (skipped)")
    if use_mock:
        print("note:     --mock deterministic scores (wiring test only)")
    _print_top(artifact_path.parent, scores)
    print(f"artifact: {artifact_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the M5b scoring stage for a clipper job"
    )
    parser.add_argument("job_id", help="Job ID to score")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Deterministic synthetic scores (wiring test only, never real judgement)",
    )

    args = parser.parse_args()
    sys.exit(main(args.job_id, use_mock=args.mock))
