#!/usr/bin/env python3
"""
Re-run the M2a ingest chain (ingest/probe/proxy/audio) for a job.

For a job whose state is ``failed`` (e.g. after its artifacts under
``data/jobs/<job_id>/`` were removed), ``run_ingest`` reuses the existing
job by content hash, resets its stage rows, and rebuilds
``ingest.json / probe.json / proxy.mp4 / audio.wav``.

Usage:
    python scripts/ingest_job.py <job-id>
"""

import argparse
import sys

from clipper.adapters.media import get_adapter
from clipper.config import Config
from clipper.db import Database
from clipper.stages import StageError
from clipper.stages.ingest import run_ingest


def main(job_id: str) -> int:
    cfg = Config.load_config()
    db = Database(cfg)
    row = db.get_job(job_id)
    if row is None:
        print(f"Error: job {job_id} not found")
        return 1

    conn = db.get_connection()
    media = get_adapter()
    try:
        actual_id, is_new, should_retry = run_ingest(
            source=str(row["source_path"]),
            client=row.get("client") or "manual",
            niche=row.get("niche") or "generic",
            cfg=cfg,
            conn=conn,
            media=media,
        )
    except (StageError, Exception) as e:
        print(f"Error: ingest failed: {e}")
        return 1
    finally:
        conn.close()

    if should_retry or is_new:
        print(f"ingest complete for {actual_id}")
    else:
        print(f"job {actual_id} unchanged (not new / not a retry)")
    if actual_id != job_id:
        print(f"NOTE: content hash resolved to job {actual_id}, not {job_id}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run M2a ingest for a job.")
    parser.add_argument("job_id")
    args = parser.parse_args()
    sys.exit(main(args.job_id))
