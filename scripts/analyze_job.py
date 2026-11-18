#!/usr/bin/env python3
"""
Manual analysis script for clipper jobs.

Runs the shots, silence and energy analysis stages (M3b) on an
already-ingested job. Mirrors scripts/transcribe_job.py.

Usage:
    python scripts/analyze_job.py <job-id>
"""

import argparse
import sys

from clipper.adapters import media as media_adapter
from clipper.adapters import scenes as scenes_adapter
from clipper.config import Config
from clipper.db import Database
from clipper.stages.analyze import run_analysis


def main(job_id: str) -> int:
    """
    Analyze a job (shots, silence, energy).

    Args:
        job_id: The job ID to analyze

    Returns:
        0 on success, 1 on failure
    """
    cfg = Config.load_config()

    db = Database(cfg)
    if db.get_job(job_id) is None:
        print(f"Error: job {job_id} not found")
        return 1

    # Adapters go through the module-level factories so tests can
    # monkeypatch clipper.adapters.{media,scenes}.get_*_adapter().
    media = media_adapter.get_adapter()
    scenes = scenes_adapter.get_scene_adapter()

    conn = db.get_connection()
    try:
        try:
            summary = run_analysis(job_id, cfg, conn, media, scenes)
        except Exception as e:
            print(f"Error: analysis failed: {e}")
            return 1
    finally:
        conn.close()

    # Report results
    print(f"job:             {job_id}")
    print(f"shots:           {summary.get('shots', 0)}")
    print(f"silence regions: {summary.get('silence_regions', 0)}")
    print(f"energy windows:  {summary.get('energy_windows', 0)}")
    for name in ("shots", "silence", "energy"):
        artifact = summary.get("artifacts", {}).get(name)
        if artifact:
            print(f"{name}: {artifact}")
    if summary.get("skipped"):
        print(f"skipped stages:  {', '.join(summary['skipped'])}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a clipper job")
    parser.add_argument("job_id", help="Job ID to analyze")

    args = parser.parse_args()
    sys.exit(main(args.job_id))
