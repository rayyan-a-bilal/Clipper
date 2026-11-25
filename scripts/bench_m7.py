#!/usr/bin/env python3
"""
M7 benchmark proof (docs/spec/m7.md M7f; SKILL.md M6/M7 rule — must
include benchmark proof: render fps, GPU utilization, NVENC session).

Benchmarks the M7d render stage (per-clip mezzanine render + final
compose) for a job by running it once with a timing wrapper around the
render adapter (mezzanine/compose invocations, one per clip, in plan
order), sampling `nvidia-smi` on GPU0 (`utilization.gpu` +
`utilization.encoder` = NVENC activity) every 0.5s during the run, then
writing per clip:

    data/jobs/<job_id>/clips/<clip_id>/bench.json
      mezzanine_seconds, compose_seconds, total_seconds,
      decode_fps, encode_fps, output_fps, frames_processed,
      peak_gpu0_util, nvenc_session_seen, encoder, hwaccel

and printing a summary table. Exit 0 on success, 1 on a clean error.

Timing attribution: the stage processes clips IN PLAN ORDER with exactly
ONE `render_mezzanine` + ONE `compose_final` call per clip, so the
wrapper call lists are sliced positionally (same pattern as bench_m6).

Frame / fps fields: `frames_processed` = `final.mp4` frame count
(ffprobe `nb_read_frames`, `duration*avg_frame_rate` fallback);
`output_fps` = probed final fps; `decode_fps` = frames /
`mezzanine_seconds` (NVDEC decode + crop/scale pass) and `encode_fps` =
frames / `compose_seconds` (NVENC pass) — approximations documented in
`docs/spec/m7.md` M7f.

MANUAL BENCHMARK — GPU0 (RTX 2080 Super) must be FREE and the
PI/coding agent STOPPED (it competes for VRAM; SKILL.md hardware rule).
The P100s (Ollama LLM) are never involved.

Usage:
    python scripts/bench_m7.py <job-id>
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from clipper.adapters.media import MediaError
from clipper.adapters.render import get_render_adapter
from clipper.config import Config
from clipper.db import Database
from clipper.models import RenderArtifact
from clipper.stages import StageError
from clipper.stages.render import run_render

logger = logging.getLogger("clipper.bench_m7")

_SAMPLE_INTERVAL_S = 0.5


class GpuSampler:
    """Poll `nvidia-smi` GPU0 utilization (GPU + encoder = NVENC
    activity) in a background thread (same pattern as bench_m6)."""

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
            self._stop.wait(_SAMPLE_INTERVAL_S)

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


class TimedRender:
    """Wraps the render adapter; records per-call wall seconds for the
    mezzanine render and the final compose (one entry per clip, in the
    stage's processing order). Everything else passes through."""

    def __init__(self, impl) -> None:
        self._impl = impl
        self.mezzanine_seconds: list[float] = []
        self.compose_seconds: list[float] = []
        self.extract_seconds: list[float] = []

    def render_mezzanine(self, *args, **kwargs):
        started = time.monotonic()
        info = self._impl.render_mezzanine(*args, **kwargs)
        self.mezzanine_seconds.append(time.monotonic() - started)
        return info

    def compose_final(self, *args, **kwargs):
        started = time.monotonic()
        artifact = self._impl.compose_final(*args, **kwargs)
        self.compose_seconds.append(time.monotonic() - started)
        return artifact

    def extract_audio(self, *args, **kwargs):
        started = time.monotonic()
        info = self._impl.extract_audio(*args, **kwargs)
        self.extract_seconds.append(time.monotonic() - started)
        return info

    def __getattr__(self, name):
        return getattr(self._impl, name)


def _fps(count: int, seconds: float) -> float:
    return round(count / seconds, 2) if (seconds > 0.01 and count) else 0.0


def _probe_frames(path: str) -> int:
    """Frame count of a rendered file: `nb_read_frames` (accurate, counts
    decoded frames) with a `duration * avg_frame_rate` fallback; 0 when
    neither is available (bench.json then falls back to
    `duration_s * fps` from `render.json`)."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                "-i", path,
            ],
            capture_output=True, text=True, timeout=300,
        )
        first = (out.stdout or "").strip().splitlines()[0].strip() \
            if out.stdout else ""
        if first.isdigit() and int(first) > 0:
            return int(first)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "csv=p=0",
                "-i", path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        lines = out.stdout.strip().splitlines()
        rate, duration = (line.strip() for line in lines[:2])
        numerator, _, denominator = rate.partition("/")
        if numerator.isdigit():
            num = int(numerator)
            den = int(denominator) if (
                denominator.isdigit() and int(float(denominator)) > 0
            ) else 1
            frames = float(duration) * num / den
            if frames > 0:
                return int(frames)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return 0


def _bench_payload(
    index: int,
    rec: dict,
    timed: TimedRender,
    clip_dir: Path,
    sampler: GpuSampler,
) -> dict:
    """Assemble the M7f bench.json payload for the clip at `index` in
    the stage's processing order (positional slice of the wrapper
    lists). `encoder`/`hwaccel`/`output_fps` come from the stage's
    `render.json` for that clip."""
    mezz = (
        timed.mezzanine_seconds[index]
        if index < len(timed.mezzanine_seconds) else 0.0
    )
    compose = (
        timed.compose_seconds[index]
        if index < len(timed.compose_seconds) else 0.0
    )
    total = float(rec.get("elapsed_seconds") or 0.0)
    if total <= 0.0:
        total = mezz + compose

    artifact = RenderArtifact.model_validate_json(
        (clip_dir / "render.json").read_text(encoding="utf-8")
    )
    frames = _probe_frames(artifact.final_path)
    if frames <= 0:
        frames = int(round(artifact.duration_s * artifact.fps))

    return {
        "clip_id": rec["clip_id"],
        "mezzanine_seconds": round(float(mezz), 3),
        "compose_seconds": round(float(compose), 3),
        "total_seconds": round(float(total), 3),
        "decode_fps": _fps(frames, mezz),
        "encode_fps": _fps(frames, compose),
        "output_fps": round(float(artifact.fps), 3),
        "frames_processed": int(frames),
        "peak_gpu0_util": int(sampler.peak_gpu),
        "nvenc_session_seen": bool(sampler.available and sampler.peak_encoder > 0),
        "encoder": artifact.encoder,
        "hwaccel": artifact.hwaccel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_m7.py",
        description="Benchmark the M7 render stage (manual GPU0 run).",
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

    timed = TimedRender(get_render_adapter(cfg))

    sampler.start()
    conn = Database(cfg).get_connection()
    try:
        summary = run_render(args.job_id, cfg, conn, render=timed)
    except (StageError, MediaError, ValueError) as e:
        sampler.stop()
        conn.close()
        print(f"ERROR: {e}")
        print("hint: GPU0 (2080 Super) must be free; PI agent stopped.")
        return 1
    conn.close()
    sampler.stop()

    clips = list(summary.get("clips") or [])
    if summary.get("skipped") or not clips:
        print(
            "bench: stage skipped / no clips — nothing measured. "
            "Remove data/jobs/<job>/clips + reset the render stage to "
            "benchmark for real."
        )
        return 1

    job_dir = cfg.resolved_data_dir / "jobs" / args.job_id
    payloads: list[dict] = []
    for i, rec in enumerate(clips):
        clip_dir = job_dir / "clips" / rec["clip_id"]
        payload = _bench_payload(i, rec, timed, clip_dir, sampler)
        payloads.append(payload)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "bench.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8",
        )

    print()
    print(f"M7 benchmark — {args.job_id}")
    hdr = (
        f"{'clip_id':<24} {'mezz_s':>8} {'compose_s':>10} {'total_s':>9} "
        f"{'dec_fps':>8} {'enc_fps':>8} {'frames':>7} {'out_fps':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in payloads:
        print(
            f"{p['clip_id']:<24} "
            f"{p['mezzanine_seconds']:>8.2f} "
            f"{p['compose_seconds']:>10.2f} "
            f"{p['total_seconds']:>9.2f} "
            f"{p['decode_fps']:>8.2f} "
            f"{p['encode_fps']:>8.2f} "
            f"{p['frames_processed']:>7d} "
            f"{p['output_fps']:>8.2f}"
        )
    print(
        f"encoder: {payloads[0]['encoder']}  hwaccel: {payloads[0]['hwaccel']}  "
        f"GPU0 peak utilization: {sampler.peak_gpu}%  "
        f"NVENC session seen: {payloads[0]['nvenc_session_seen']}  "
        f"(nvidia-smi samples: {sampler.samples})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
