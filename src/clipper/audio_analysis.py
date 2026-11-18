"""
Pure CPU logic for silence/energy analysis (M3a).

Both functions here are pure: no file I/O, no ffmpeg/subprocess, and no
heavy-library imports beyond numpy at module level. See docs/spec/m3.md.
"""

import math
import re

import numpy as np

from .models import EnergyWindow, SilenceRegion

# Module-level constants (fixed by docs/spec/m3.md — no config changes).
SILENCE_NOISE_DB = -35.0
SILENCE_MIN_DUR = 0.5
ENERGY_WINDOW_S = 0.5

# rms_db floor: 20 * log10(1e-6) == -120.0 dB for silent windows.
_RMS_FLOOR = 1e-6
# A final partial window shorter than this is skipped.
_MIN_PARTIAL_WINDOW_S = 0.1

_SILENCE_START_RE = re.compile(
    r"silence_start:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def parse_silence_log(text: str, duration_s: float) -> list[SilenceRegion]:
    """Parse ffmpeg silencedetect stderr into silence regions.

    Recognized lines (anything else is ignored):
      silence_start: X
      silence_end: Y | silence_duration: Z

    Starts are paired with the next end. A trailing unpaired start is
    closed at the audio duration. All boundaries are clamped into
    [0, duration_s] and zero/negative-length regions are dropped.
    """

    def close(start: float, end: float) -> None:
        start = max(0.0, start)
        end = min(duration_s, end)
        if end > start:
            regions.append(SilenceRegion(start=start, end=end))

    regions: list[SilenceRegion] = []
    open_start: float | None = None

    for line in text.splitlines():
        m_end = _SILENCE_END_RE.search(line)
        if m_end is not None:
            if open_start is not None:
                close(open_start, float(m_end.group(1)))
                open_start = None
            # A silence_end without a matching start is malformed: ignore.
            continue

        m_start = _SILENCE_START_RE.search(line)
        if m_start is not None and open_start is None:
            open_start = float(m_start.group(1))
        # A second unpaired start (malformed) is ignored.

    if open_start is not None:
        close(open_start, duration_s)
    return regions


def rms_windows(
    samples: np.ndarray,
    sample_rate: int,
    window_s: float = ENERGY_WINDOW_S,
) -> list[EnergyWindow]:
    """Tile float samples (in [-1, 1]) into non-overlapping energy windows.

    Window boundaries fall exactly on multiples of ``window_s``. The final
    (partial) window is included when it is at least 0.1s long and skipped
    otherwise. ``rms_db = 20 * log10(max(rms, 1e-6))``, so silent windows
    clamp at -120.0 dB.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window_s <= 0:
        raise ValueError("window_s must be positive")

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D array")

    windows: list[EnergyWindow] = []
    n = samples.size
    if n == 0:
        return windows

    win_len = int(round(window_s * sample_rate))
    if win_len <= 0:
        return windows

    min_partial = int(round(_MIN_PARTIAL_WINDOW_S * sample_rate))
    for offset in range(0, n, win_len):
        chunk = samples[offset : offset + win_len]
        # Only the final window can be partial.
        if chunk.size < win_len and chunk.size < min_partial:
            break
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        rms_db = 20.0 * math.log10(max(rms, _RMS_FLOOR))
        start = offset / sample_rate
        windows.append(
            EnergyWindow(
                start=start,
                end=start + chunk.size / sample_rate,
                rms_db=rms_db,
            )
        )
    return windows
