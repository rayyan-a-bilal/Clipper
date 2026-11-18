#!/usr/bin/env python3
"""
Manual transcription script for clipper jobs.

Runs the transcribe and sentences stages on an already-ingested job.

Usage:
    python scripts/transcribe_job.py <job-id>
"""

import argparse
import json
import sys

from clipper.adapters.asr import get_adapter
from clipper.config import Config
from clipper.db import Database
from clipper.stages.transcribe import run_transcribe


def main(job_id: str) -> int:
    """
    Transcribe a job.

    Args:
        job_id: The job ID to transcribe

    Returns:
        0 on success, 1 on failure
    """
    cfg = Config.load_config()

    db = Database(cfg)
    if db.get_job(job_id) is None:
        print(f"Error: job {job_id} not found")
        return 1

    asr = get_adapter(cfg)
    try:
        if not run_transcribe(job_id, cfg, asr):
            print("Error: transcription stage did not report success")
            return 1
    except Exception as e:
        print(f"Error: transcription failed: {e}")
        return 1

    # Report results
    job_dir = cfg.resolved_data_dir / "jobs" / job_id

    words_file = job_dir / "words.json"
    if words_file.exists():
        with open(words_file) as f:
            print(f"words:     {len(json.load(f).get('words', []))}")

    sentences_file = job_dir / "sentences.json"
    if sentences_file.exists():
        with open(sentences_file) as f:
            print(f"sentences: {len(json.load(f).get('sentences', []))}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a clipper job")
    parser.add_argument("job_id", help="Job ID to transcribe")

    args = parser.parse_args()
    sys.exit(main(args.job_id))
