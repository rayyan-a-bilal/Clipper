#!/usr/bin/env python3
"""
M6 benchmark proof (docs/spec/m6.md; SKILL.md M6/M7 rule — must include
benchmarks: decode fps, inference fps, GPU utilization, NVENC session).

Runs the M6e layout stage for a job with timing wrappers around the
vision adapter, the frame decoder and the face detector, sampling
``nvidia-smi`` on GPU0 (``utilization.gpu`` + ``utilization.encoder``)
every 0.5s during the run, and writes per-clip:

  data/jobs/<job_id>/clips/<clip_id>/bench.json
    vision_segment_seconds, decode_fps, inference_fps, frames_processed,
    layout_seconds, total_seconds, detector, device,
    peak_gpu0_utilization, nvenc_session_seen

Then prints a summary table. Exit 0 on success, 1 on a clean error.

Timing attribution: the stage processes clips IN PLAN ORDER with exactly
ONE vision call + ONE decode pass + N detect calls per clip, so the
wrapper call lists are sliced positionally (order-based) to attribute
seconds/frames to each clip.

MANUAL GPU BENCHMARK — run with GPU0 (2080 Super) FREE and the
PI/coding agent STOPPED (it competes for VRAM; SKILL.md hardware rule).
The P100s (Ollama LLM) are never involved.

Usage:
    python scripts/bench_m6.py <job-id>
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import threading
import time

from clipper.adapters.faces import FaceDetectorError, get_face_detector
from clipper.adapters.vision import (
    MediaError,
    get_vision_adapter,
)
from clipper.config import Config
from clipper.db import Database
from clipper.stages import StageError
from clipper.stages.layout import (
    default_frame_reader,
    run_layout,
)

logger = logging.getLogger("clipper.bench_m6")

_SAMPLE_INTERVAL_S = 0.5


class GpuSampler:
    """Poll ``nvidia-smi`` GPU0 utilization (GPU + encoder = NVENC
    activity) in a background thread."""

    def __init__(self) -> None:
        self.peak_gpu = 0
        self.peak_encoder = 0
        self.samples = 0
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.encoder",
                        "--format=csv,noheader,nounits",
                        "-i", "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                gpu, enc = (int(v) for v in out.stdout.split(",")[:2])
            except (OSError, subprocess.SubprocessError, ValueError):
                break
            self.samples += 1
            self.peak_gpu = max(self.peak_gpu, gpu)
            self.peak_encoder = max(self.peak_encoder, enc)
            self._stop.wait(self._SAMPLE_INTERVAL_S)

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


class TimedVision:
    """Wraps the vision adapter; records per-call wall seconds."""

    def __init__(self, impl) -> None:
        self._impl = impl
        self.seconds: list[float] = []

    def make_segment(self, *args, **kwargs):
        started = time.monotonic()
        segment = self._impl.make_segment(*args, **kwargs)
        self.seconds.append(time.monotonic() - started)
        return segment


class TimedReader:
    """Wraps ``default_frame_reader``; records frames decoded + seconds
    (one call per clip, in stage processing order)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, path: str, fps: float):
        iterator = default_frame_reader(path, fps)
        count = 0
        started = time.monotonic()
        for item in iterator:
            count += 1
            yield item
        self.calls.append({
            "path": path,
            "frames": count,
            "seconds": time.monotonic() - started,
        })


class TimedDetector:
    """Wraps the face detector; records per-detect wall seconds (one
    entry per frame, in stage processing order)."""

    def __init__(self, impl) -> None:
        self._impl = impl
        self.detect_seconds: list[float] = []

    def detect(self, frame, t: float):
        started = time.monotonic()
        detections = self._impl.detect(frame, t)
        self.detect_seconds.append(time.monotonic() - started)
        return detections

    def name(self) -> str:
        return self._impl.name()

    def close(self) -> None:
        self._impl.close()


def _fps(count: int, seconds: float) -> float:
    return round(count / seconds, 2) if (seconds > 0.01 and count) else 0.0


def _bench_payload(
    index: int,
    rec: dict,
    clips: list[dict],
    frames_by_clip: dict[str, int],
    vision: TimedVision,
    reader: TimedReader,
    detector: TimedDetector,
    device: str,
    sampler: GpuSampler,
) -> dict:
    """Assemble the bench.json payload for the clip at `index` in the
    stage's processing order (positional slice of the wrapper lists)."""
    clip_id = rec["clip_id"]
    frames = int(frames_by_clip.get(clip_id, 0))

    vision_sec = vision.seconds[index] if index < len(vision.seconds) else 0.0
    decode = reader.calls[index] if index < len(reader.calls) else None
    if decode is not None:
        decode_frames = int(decode["frames"] or frames)
    else:
        decode_frames = frames
    decode_sec = float(decode["seconds"]) if decode is not None else 0.0

    offset = sum(frames_by_clip[c["clip_id"]] for c in clips[:index])
    infer_sec = sum(detector.detect_seconds[offset:offset + frames]) if frames else 0.0

    total_sec = float(rec.get("elapsed_seconds") or 0.0)
    layout_sec = max(0.0, total_sec - vision_sec - decode_sec - infer_sec)

    return {
        "clip_id": clip_id,
        "vision_segment_seconds": round(float(vision_sec), 3),
        "decode_fps": _fps(decode_frames, decode_sec),
        "inference_fps": _fps(frames, infer_sec),
        "frames_processed": frames,
        "layout_seconds": round(layout_sec, 3),
        "total_seconds": round(total_sec, 3),
        "detector": detector.name(),
        "device": device,
        "peak_gpu0_utilization": sampler.peak_gpu,
        "nvenc_session_seen": bool(sampler.available and sampler.peak_encoder > 0),
        "nvidia_smi_samples": sampler.samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_m6.py",
        description="Benchmark the M6e layout stage (manual GPU0 run).",
    )
    parser.add_argument("job_id", help="clipper job id")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if shutil.which("nvidia-smi") is None:
        print("ERROR: nvidia-smi not found — cannot verify GPU0 usage.")
        return 1

    cfg = Config()
    sampler = GpuSampler()

    # Pre-flight: is GPU0 already busy? (Ollama / PI agent compete for VRAM.)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            capture_output=True, text=True, timeout=5,
        )
        util, mem = (int(v) for v in out.stdout.split(",")[:2])
        if util > 50:
            print(
                f"WARNING: GPU0 utilization already {util}% "
                f"({mem} MiB in use) — benchmark will be polluted. "
                "Stop other GPU work (Ollama / PI agent) and re-run."
            )
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"WARNING: GPU0 pre-flight sample failed ({e}); continuing.")

    vision = TimedVision(get_vision_adapter(cfg))
    detector = TimedDetector(get_face_detector(cfg))
    reader = TimedReader()

    sampler.start()
    started = time.monotonic()
    conn = Database(cfg).get_connection()
    try:
        summary = run_layout(
            args.job_id, cfg, conn,
            vision=vision, detector=detector, frame_reader=reader,
        )
    except (StageError, FaceDetectorError, MediaError, ValueError) as e:
        print(f"ERROR: {e}")
        print("hint: GPU0 (2080 Super) must be free; PI agent stopped.")
        return 1
    finally:
        conn.close()
    wall = time.monotonic() - started
    sampler.stop()

    clips = list(summary.get("clips") or [])
    if summary.get("skipped") or not clips:
        print(
            "bench: stage skipped / no clips — nothing measured. "
            "Remove data/jobs/<job>/clips and reset the stage to "
            "benchmark for real."
        )
        return 1

    # frames per clip: the decoded reader call (positionally matched);
    # fall back to the crop keyframe count for the stage-record window.
    frames_by_clip: dict[str, int] = {}
    for i, rec in enumerate(clips):
        frame_count = 0
        if i < len(reader.calls):
            frame_count = int(reader.calls[i]["frames"])
        if not frame_count:
            frame_count = int(rec.get("keyframe_count") or 0)
        frames_by_clip[rec["clip_id"]] = frame_count

    job_dir = cfg.resolved_data_dir / "jobs" / args.job_id
    for i, rec in enumerate(clips):
        clip_id = rec["clip_id"]
        payload = _bench_payload(
            i, rec, clips, frames_by_clip,
            vision, reader, detector,
            cfg.devices.vision, sampler,
        )
        clip_dir = job_dir / "clips" / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "bench.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8",
        )

    hdr = (
        f"{'clip_id':<24} {'vision_s':>9} {'decode_fps':>11} "
        f"{'infer_fps':>10} {'frames':>7} {'layout_s':>9} {'total_s':>8}"
    )
    print()
    print(
        f"M6 benchmark — {args.job_id}  (detector={detector.name()}, "
        f"device={cfg.devices.vision}, wall={wall:.2f}s)"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, rec in enumerate(clips):
        payload = _bench_payload(
            i, rec, clips, frames_by_clip,
            vision, reader, detector,
            cfg.devices.vision, sampler,
        )
        print(
            f"{payload['clip_id']:<24} "
            f"{payload['vision_segment_seconds']:>9.2f} "
            f"{payload['decode_fps']:>11.2f} "
            f"{payload['inference_fps']:>10.2f} "
            f"{payload['frames_processed']:>7d} "
            f"{payload['layout_seconds']:>9.2f} "
            f"{payload['total_seconds']:>8.2f}"
        )
    print(
        f"GPU0 peak utilization: {sampler.peak_gpu}%  "
        f"NVENC session seen: {bool(sampler.peak_encoder > 0)}  "
        f"(nvidia-smi samples: {sampler.samples})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
