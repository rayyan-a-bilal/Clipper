"""Sendcmd crop-path generation for the face_track mezzanine (M7e).

docs/spec/m7.md M7e. PURE — imports only `clipper.models`; no I/O, no
ffmpeg, no GPU (unit-testable like the M7a `clipper.captions` engine).

`crop_cmd_from_path(crop_path, fps) -> str` returns the FULL content of
the `crop.cmd` file that is written BESIDE the mezzanine for the
face_track path, and that the `sendcmd` filter reads via
`sendcmd=f=…`. One line per crop key, in `crop_path` order:

    {t} crop x {x}, crop y {y};

- `t` is the CLIP-RELATIVE timestamp stored on the key (the M6d grid at
  native fps: `0.0, 1/fps, …`).
- DENSE keys are allowed — one line per key, no decimation/rounding of
  the key grid.
- The t=0 initial command is INCLUDED as the first line (M6d crop paths
  always start at `t == 0.0`; a path that does not is an input error and
  raises `ValueError`).
"""
from __future__ import annotations

from clipper.models import CropKey

# File name written BESIDE the mezzanine (docs/spec/m7.md M7e).
CROP_CMD_FILENAME = "crop.cmd"


def crop_cmd_from_path(crop_path: list[CropKey], fps: float) -> str:
    """Sendcmd file content driving the `crop` filter's x/y per key.

    Deterministic: same inputs -> byte-identical output. Raises
    `ValueError` on an invalid grid (non-positive `fps`, empty path, or a
    first key that is not `t == 0.0`).
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    if not crop_path:
        raise ValueError(
            "crop_path is empty — the face_track sendcmd file needs at "
            "least the t=0 initial crop command (M6d crop paths start "
            "at t == 0.0)"
        )
    if crop_path[0].t != 0.0:
        raise ValueError(
            f"crop_path must start at t=0 (first key t={crop_path[0].t!r}) — "
            "sendcmd needs the initial crop x/y command"
        )
    lines: list[str] = []
    for key in crop_path:
        lines.append(f"{key.t:g} crop x {int(key.x)}, crop y {int(key.y)};")
    return "\n".join(lines) + "\n"
