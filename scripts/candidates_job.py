#!/usr/bin/env python3
"""
Manual candidates script for clipper jobs (M4b).

Runs the candidates stage (M4a pre-rank, deterministic) on an
already-transcribed and analyzed job. Mirrors scripts/analyze_job.py.

Usage:
    python scripts/candidates_job.py <job-id>
"""

import argparse
import json
import sys
from pathlib import Path

from clipper.config import Config
from clipper.db import Database
from clipper.stages.candidates import run_candidates

_PREVIEW_WORDS = 6
_TOP_N = 5


def _preview(text: str, limit: int = _PREVIEW_WORDS) -> str:
    words = (text or "").split()
    return " ".join(words[:limit])


def main(job_id: str) -> int:
    """
    Run the candidates stage for a job and print a short summary.

    Args:
        job_id: The job ID to process

    Returns:
        0 on success (including zero candidates), 1 on failure
    """
    cfg = Config.load_config()

    db = Database(cfg)
    if db.get_job(job_id) is None:
        print(f"Error: job {job_id} not found")
        return 1

    conn = db.get_connection()
    try:
        try:
            summary = run_candidates(job_id, cfg, conn)
        except Exception as e:
            print(f"Error: candidates stage failed: {e}")
            return 1
    finally:
        conn.close()

    artifact_path = Path(summary["artifact"])
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])

    # First-words preview comes from the M2 sentences.json (index = first
    # sentence of the candidate window).
    sentences: list[dict] = []
    sentences_path = artifact_path.parent / "sentences.json"
    if sentences_path.exists():
        try:
            sentences = json.loads(
                sentences_path.read_text(encoding="utf-8")
            ).get("sentences", [])
        except (OSError, json.JSONDecodeError):
            sentences = []

    print(f"job:        {job_id}")
    print(f"candidates: {len(candidates)}")
    if summary.get("skipped"):
        print("note:       candidates stage already done (skipped)")
    if candidates:
        print(f"top {min(_TOP_N, len(candidates))}:")
    for cand in candidates[:_TOP_N]:
        start, end = cand["start"], cand["end"]
        start_idx = cand["sentence_range"][0]
        text = (
            sentences[start_idx].get("text", "")
            if start_idx < len(sentences)
            else ""
        )
        print(
            f"  [{start:.2f}-{end:.2f}] "
            f"{end - start:.1f}s "
            f"{cand['cheap_score']:.6f} "
            f"{_preview(text)}"
        )
    print(f"artifact:   {artifact_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate candidates for a clipper job"
    )
    parser.add_argument("job_id", help="Job ID to process")

    args = parser.parse_args()
    sys.exit(main(args.job_id))
