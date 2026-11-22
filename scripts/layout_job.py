#!/usr/bin/env python3
"""
Manual M6e layout-stage runner (docs/spec/m6.md) — completes the M6
per-clip layout + crop path.

Renders the per-clip vision segment (GPU0, NVDEC/NVENC), runs per-frame
face detection (EVERY frame), then M6c tracking/M6d layout+crop-track pure
logic, and writes:

  data/jobs/<job_id>/clips/<clip_id>/layout.json   (ClipLayoutArtifact)
  data/jobs/<job_id>/clip_plans.json               (layout + crop_path filled)
  `clips` rows upserted

Detector selection:
  default     : YuNet ONNX on GPU0 (``cfg.devices.vision``) — fails LOUDLY
                if the CUDA path is unavailable (no silent CPU fallback).
  --cpu-faces : cv2 FaceDetectorYN CPU debug path — LOUD WARNING printed.

Layout selection:
  default  : auto (``clipper.layouts.decide_layout``)
  --layout : forced layout, recorded as ``layout_source="manual"`` in
             ``layout.json``.

With ZERO selected clips the stage marks itself done and does NO detector
work (valid success — exit 0). A second run skips the completed stage.

Usage:
    python scripts/layout_job.py <job-id>
    python scripts/layout_job.py <job-id> --clip <clip-id>
    python scripts/layout_job.py <job-id> --layout split_screen
    python scripts/layout_job.py <job-id> --cpu-faces
"""

import argparse
import logging
import sys
import time

from clipper.adapters.faces import FaceDetectorError, get_face_detector
from clipper.adapters.vision import MediaError, get_vision_adapter
from clipper.config import Config
from clipper.db import Database
from clipper.models import Layout
from clipper.stages import StageError
from clipper.stages.layout import run_layout

logger = logging.getLogger("clipper.layout_job")

CPU_FACES_WARNING = (
    "=============================================================== "
    "WARNING: --cpu-faces — face detection on CPU via cv2. DEBUG PATH "
    "ONLY; production is YuNet ONNX on GPU0 (docs/spec/m6.md, decision "
    "9)."
)
LAYOUT_CHOICES = ["center_crop", "face_track", "split_screen", "facecam"]


def _print_clips(summary: dict) -> None:
    clips = summary.get("clips") or []
    if not clips:
        print("layout: 0 clip(s) — VALID success (no detector work)")
        return
    print(f"layout: {len(clips)} clip(s)")
    for rec in clips:
        elapsed = rec.get("elapsed_seconds")
        if elapsed is None:
            elapsed_txt = "-"
        else:
            elapsed_txt = f"{elapsed:.2f}s"
        print(
            f"  {rec['clip_id']}  layout={rec['layout']}  "
            f"face_tracks={rec['face_track_count']}  "
            f"detector={rec['detector']}  keyframes={rec['keyframe_count']}"
            f"  layout_source={rec['layout_source']}  elapsed={elapsed_txt}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="layout_job.py",
        description="Run the M6e layout stage (layout + crop path) for a job.",
    )
    parser.add_argument("job_id", help="clipper job id")
    parser.add_argument(
        "--clip",
        default=None,
        help="restrict the run to ONE clip id",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUT_CHOICES,
        default=None,
        help="force a layout (recorded as layout_source=manual)",
    )
    parser.add_argument(
        "--cpu-faces",
        action="store_true",
        help="CPU face detector (cv2) — debug path only, LOUD warning",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.cpu_faces:
        print(CPU_FACES_WARNING)

    cfg = Config()
    forced_layout = Layout(args.layout) if args.layout else None
    vision = get_vision_adapter(cfg)
    detector = get_face_detector(cfg, cpu=args.cpu_faces)

    started = time.monotonic()
    conn = Database(cfg).get_connection()
    try:
        summary = run_layout(
            args.job_id,
            cfg,
            conn,
            vision,
            detector,
            clip_id=args.clip,
            forced_layout=forced_layout,
        )
    except (StageError, FaceDetectorError, MediaError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        conn.close()

    if summary.get("skipped"):
        print("layout: skipped — stage already done and all layout.json valid")
    _print_clips(summary)
    print(f"done in {time.monotonic() - started:.2f}s (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
