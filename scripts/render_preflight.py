#!/usr/bin/env python3
"""M7b render preflight (MANUAL — docs/spec/m7.md M7b).

Checks the local ffmpeg build for the M7 render prerequisites:

  REQUIRED (exit 1 on any failure):
    - `ffmpeg -hwaccels` contains `cuda`        (NVDEC, GPU0)
    - `ffmpeg -encoders` contains `h264_nvenc`  (NVENC, GPU0)
    - `ffmpeg -filters`  contains `ass`         (caption burn, M7 final)

  WARN-ONLY (CPU fallback allowed, docs/spec/m7.md M7b judgment note):
    - `scale_cuda` filter present → GPU scale path; ABSENT → the render
      adapter falls back to CPU `scale=...:flags=lanczos` + NVENC and
      logs a LOUD warning.

Exit 0 = all required checks pass (scale_cuda absence is a warning).
Exit 1 = ffmpeg missing, or a REQUIRED check failed.
"""

import re
import shutil
import subprocess
import sys

PROBE_TIMEOUT_S = 60


def _run(args: list[str]) -> tuple[int, str]:
    """Run a CLI; returns (rc, combined stdout+stderr). Missing binary → 127."""
    try:
        process = subprocess.run(args, capture_output=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"error: {e}"
    out = process.stdout.decode("utf-8", errors="replace")
    err = process.stderr.decode("utf-8", errors="replace")
    return process.returncode, out + ("\n" + err if err.strip() else "")


def _filter_names(output: str) -> set[str]:
    """Filter names from `ffmpeg -filters` output (3rd token per line)."""
    names = set()
    for line in output.splitlines():
        tokens = line.split()
        # lines look like: " T. scale_cuda V->V GPU video scaler for CUDA"
        # (flag column, then the filter name)
        if len(tokens) >= 2 and re.fullmatch(r"[T.*F]{1,5}", tokens[0] or ""):
            names.add(tokens[1])
        elif re.fullmatch(r"[A-Za-z0-9_]+", tokens[0] or "") and len(tokens) >= 2:
            names.add(tokens[0])  # tolerate stripped/compact formats
    return names


def main() -> int:
    print("== M7b render preflight (docs/spec/m7.md M7b) ==")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("FAIL  ffmpeg not found on PATH (required)")
        return 1
    print(f"      ffmpeg: {ffmpeg}")

    results: list[tuple[str, bool, bool, str]] = []  # (name, required, ok, detail)

    rc, out = _run([ffmpeg, "-hide_banner", "-hwaccels"])
    ok = rc == 0 and "cuda" in out
    results.append(("cuda in -hwaccels (NVDEC, GPU0)", True, ok,
                    "cuda present" if ok else
        f"rc={rc}; output: {out.strip()[:200]!r}"))

    rc, out = _run([ffmpeg, "-hide_banner", "-encoders"])
    ok = rc == 0 and "h264_nvenc" in out
    results.append(("h264_nvenc in -encoders (NVENC, GPU0)", True, ok,
                    "h264_nvenc present" if ok else
        f"rc={rc}; output: {out.strip()[:200]!r}"))

    rc, out = _run([ffmpeg, "-hide_banner", "-filters"])
    names = _filter_names(out)
    ok = rc == 0 and "ass" in names
    results.append(("ass in -filters (caption burn)", True, ok,
                    "ass present" if ok else
        f"rc={rc}; ass not found ({len(names)} filters listed)"))

    scale_cuda = rc == 0 and "scale_cuda" in names
    results.append(("scale_cuda in -filters (GPU scale)", False, scale_cuda,
                    "scale_cuda present → GPU scale path"
                    if scale_cuda
                    else "ABSENT → CPU scale (lanczos) + NVENC fallback is ALLOWED; "
                        "the adapter logs a LOUD warning"))

    failed = False
    for name, required, ok, detail in results:
        kind = "required" if required else "optional"
        if ok:
            print(f"PASS  [{kind}] {name} — {detail}")
        elif required:
            failed = True
            print(f"FAIL  [{kind}] {name} — {detail}")
        else:
            print(f"WARN  [{kind}] {name} — {detail}")

    if failed:
        print("RESULT: FAIL (required M7 render prerequisite missing)")
        return 1
    print("RESULT: OK (all required M7 render prerequisites present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
