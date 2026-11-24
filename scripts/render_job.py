#!/usr/bin/env python3
"""
Manual M7d render-stage runner (docs/spec/m7.md) — renders center_crop
clips end to end.

For every ClipPlan (or only ``--clip``) in ``clip_plans.json`` it renders
under ``data/jobs/<job_id>/clips/<clip_id>/``:

  mezzanine.mp4   NVDEC decode + center crop/scale + NVENC (1080x1920,
                  native fps)
  audio.m4a       clip audio window (AAC 48kHz stereo)
  captions.ass    deterministic ASS captions (M7a engine)
  final.mp4       ASS burn + loudnorm + NVENC final (decision 5/6)
  thumbnail.jpg   single still at t=1.0s
  render.json     RenderArtifact record (schema_version=1)

and then rewrites ``clip_plans.json`` (``status = "rendered"``) and
upserts the ``clips`` rows.

Renderer selection:
  default     : GPU path — ``FFmpegRenderAdapter`` on GPU0 (NVDEC/NVENC,
                docs/spec/m7.md decision 1). HW failure fails LOUDLY;
                there is NO silent CPU fallback (decision 2).
  --cpu-render: EXPLICIT CPU/DEBUG path (libx264, no NVDEC) — a LOUD
                warning is printed. A debug tool, NOT a fallback.

With ZERO selected clips the stage marks itself done and does NO render
work (valid success — exit 0). A second run skips the completed stage.

Usage:
    python scripts/render_job.py <job-id>
    python scripts/render_job.py <job-id> --clip <clip-id>
    python scripts/render_job.py <job-id> --cpu-render
"""

import argparse
import logging
import sys
import time

from clipper.adapters.media import MediaError
from clipper.adapters.render import CPU_RENDER_WARNING, get_render_adapter
from clipper.config import Config
from clipper.db import Database
from clipper.stages import StageError
from clipper.stages.render import run_render

logger = logging.getLogger("clipper.render_job")


def _print_clips(summary: dict) -> None:
    clips = summary.get("clips") or []
    if not clips:
        print("render: 0 clip(s) — VALID success (no render work)")
        return
    print(f"render: {len(clips)} clip(s)")
    for rec in clips:
        print(
            f"  {rec['clip_id']}"
            f"  final={rec['final_path']}"
            f"  duration={rec['duration_s']}s"
            f"  {rec['width']}x{rec['height']} @ {rec.get('fps')}fps"
            f"  encoder={rec['encoder']}"
            f"  render_seconds={rec['render_seconds']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render_job.py",
        description="Run the M7d render stage (center_crop, end to end) "
        "for a job.",
    )
    parser.add_argument("job_id", help="clipper job id")
    parser.add_argument(
        "--clip",
        default=None,
        help="restrict the run to ONE clip id",
    )
    parser.add_argument(
        "--cpu-render",
        action="store_true",
        help="EXPLICIT CPU/DEBUG render path (libx264, no NVDEC) — "
        "LOUD warning, NOT a production fallback",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.cpu_render:
        print(CPU_RENDER_WARNING)

    cfg = Config()
    render = get_render_adapter(cfg, cpu_render=args.cpu_render or None)

    started = time.monotonic()
    conn = Database(cfg).get_connection()
    try:
        summary = run_render(
            args.job_id,
            cfg,
            conn,
            render,
            clip_id=args.clip,
            cpu_render=bool(args.cpu_render),
        )
    except (StageError, MediaError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        conn.close()

    if summary.get("skipped"):
        print("render: skipped — stage already done and all "
              "final.mp4/render.json valid")
    _print_clips(summary)
    print(f"done in {time.monotonic() - started:.2f}s (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
