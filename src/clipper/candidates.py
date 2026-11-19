"""
Pure CPU logic for clip candidate generation (M4a — foundations).

All functions here are PURE: no file I/O, no DB, no ffmpeg, no network.
See ``docs/spec/m4.md`` for the fixed decisions.

The public entry point is :func:`generate_candidates`, which takes the M2
sentences, M3 energy windows and M3 silence regions and returns the top-K
pre-ranked ``Candidate`` list. Small, individually testable helpers are
exposed for each feature, the temporal IoU, and non-max suppression.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence

from .models import Candidate, EnergyWindow, Sentence, SilenceRegion

# ---------------------------------------------------------------------------
# Constants (fixed by docs/spec/m4.md)
# ---------------------------------------------------------------------------
MIN_CLIP_S = 15.0
MAX_CLIP_S = 75.0
MIN_SENTENCES = 3
MAX_SENTENCES = 8
STRIDE = 1
TOP_K = 20
NMS_IOU = 0.5

W_HOOK = 0.35
W_DENSITY = 0.20
W_ENERGY = 0.20
W_SILENCE = 0.10
W_NICHE = 0.15

# cheap_score normalization / penalty scales
_ENERGY_STD_SCALE = 12.0
_SILENCE_FRACTION_SCALE = 0.25
_NICHE_NEUTRAL = 0.5

# Hook keyword families (matched as whole words, case-insensitive).
_HOOK_START_WORDS = {"who", "what", "when", "where", "why", "how"}
_HOOK_NEGATION = (
    "not",
    "never",
    "no",
    "don't",
    "dont",
    "can't",
    "cant",
    "won't",
    "wont",
)
_HOOK_CONTRAST = ("but", "however", "actually", "instead")
_HOOK_SUPERLATIVE = ("most", "best", "worst", "biggest", "greatest")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _has_word(text: str, words: Sequence[str]) -> bool:
    """True if any word in ``words`` occurs in ``text`` as a whole word."""
    for word in words:
        if not word:
            continue
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return True
    return False


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    o = min(a_end, b_end) - max(a_start, b_start)
    return o if o > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Feature scores
# ---------------------------------------------------------------------------
def hook_score(sentences: Sequence[Sentence]) -> float:
    """Score the FIRST sentence's text for hook-like phrasing (0..1).

    +0.25 each for: question mark; interrogative lead word; negation; any
    digit; contrast word; superlative. Capped at 1.0.
    """
    if not sentences:
        return 0.0
    text = (sentences[0].text or "").lower()
    if not text.strip():
        return 0.0

    score = 0.0
    if "?" in text:
        score += 0.25

    first_word = text.split()[0]
    if first_word in _HOOK_START_WORDS:
        score += 0.25

    if _has_word(text, _HOOK_NEGATION):
        score += 0.25

    if any(ch.isdigit() for ch in text):
        score += 0.25

    if _has_word(text, _HOOK_CONTRAST):
        score += 0.25

    if _has_word(text, _HOOK_SUPERLATIVE):
        score += 0.25

    return min(score, 1.0)


def density_score(sentences: Sequence[Sentence], window_duration: float) -> float:
    """Fraction of the window duration that is occupied by speech (0..1)."""
    if window_duration <= 0.0:
        return 0.0
    speech = sum((s.end - s.start) for s in sentences)
    return _clamp01(speech / window_duration)


def energy_score(
    energy_windows: Sequence[EnergyWindow], start: float, end: float
) -> float:
    """Dynamic range of the energy windows overlapping a window (0..1).

    Uses the population std of ``rms_db`` over windows whose MIDPOINT lies in
    ``[start, end]``; scaled by /12. 0.0 with fewer than 2 such windows.
    """
    values = [
        w.rms_db
        for w in energy_windows
        if start <= (w.start + w.end) / 2.0 <= end
    ]
    if len(values) < 2:
        return 0.0
    std = statistics.pstdev(values)
    return min(std / _ENERGY_STD_SCALE, 1.0)


def silence_score(
    silence_regions: Sequence[SilenceRegion], start: float, end: float
) -> float:
    """Penalize windows that sit over silent intervals (0..1, higher = better).

    value = 1 - min(silence_fraction / 0.25, 1), clamped to [0, 1].
    """
    window_duration = end - start
    if window_duration <= 0.0:
        return 0.0
    total = sum(_overlap(start, end, r.start, r.end) for r in silence_regions)
    fraction = _clamp01(total / window_duration)
    return _clamp01(1.0 - min(fraction / _SILENCE_FRACTION_SCALE, 1.0))


def niche_score(window_text: str, niche_keywords: Sequence[str]) -> float:
    """Fraction of configured niche keywords present in the window text (0..1).

    Falls back to a neutral 0.5 when no keywords are configured.
    """
    keywords = [
        k.lower() for k in (niche_keywords or []) if k and k.strip()
    ]
    if not keywords:
        return _NICHE_NEUTRAL
    text = (window_text or "").lower()
    present = sum(1 for k in keywords if _has_word(text, (k,)))
    return _clamp01(present / len(keywords))


def cheap_score(
    hook: float,
    density: float,
    energy: float,
    silence: float,
    niche: float,
) -> float:
    """Weighted PRE-RANK (not a quality verdict), clamped to [0,1]."""
    total = (
        W_HOOK * hook
        + W_DENSITY * density
        + W_ENERGY * energy
        + W_SILENCE * silence
        + W_NICHE * niche
    )
    return round(_clamp01(total), 6)


# ---------------------------------------------------------------------------
# IoU / NMS
# ---------------------------------------------------------------------------
def temporal_iou(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """temporal IoU = overlap / union (0.0 when there is no overlap)."""
    overlap = _overlap(a_start, a_end, b_start, b_end)
    if overlap <= 0.0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - overlap
    if union <= 0.0:
        return 0.0
    return overlap / union


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _build_windows(
    sentences: Sequence[Sentence]
) -> list[tuple[int, int, float, float]]:

    """Enumerate all valid (start_idx, end_idx, start, end) windows.

    For each start index ``i`` (stride STRIDE) and length L in 3..8 inclusive,
    end_idx = i + L - 1; keep only if 15.0 <= duration <= 75.0.
    """
    n = len(sentences)
    windows: list[tuple[int, int, float, float]] = []
    for i in range(0, n, STRIDE):
        for length in range(MIN_SENTENCES, MAX_SENTENCES + 1):
            end_idx = i + length - 1
            if end_idx >= n:
                break
            start = sentences[i].start
            end = sentences[end_idx].end
            duration = end - start
            if not (MIN_CLIP_S <= duration <= MAX_CLIP_S):
                continue
            windows.append((i, end_idx, start, end))
    return windows


def _score_window(
    start_idx: int,
    end_idx: int,
    start: float,
    end: float,
    sentences: Sequence[Sentence],
    energy_windows: Sequence[EnergyWindow],
    silence_regions: Sequence[SilenceRegion],
    niche_keywords: Sequence[str],
) -> float:
    win_sents = sentences[start_idx : end_idx + 1]
    window_duration = end - start
    text = " ".join((s.text or "") for s in win_sents)

    hook = hook_score(win_sents)
    density = density_score(win_sents, window_duration)
    energy = energy_score(energy_windows, start, end)
    silence = silence_score(silence_regions, start, end)
    niche = niche_score(text, niche_keywords)
    return cheap_score(hook, density, energy, silence, niche)


def _nms(
    ordered: list[tuple[float, int, int, float, float]], iou_threshold: float
) -> list[tuple[float, int, int, float, float]]:
    """Greedy non-max suppression over an already-order-ranked window list.

    Keep a candidate only if its temporal IoU against every already-kept
    candidate is < iou_threshold. Stop at TOP_K.
    """
    kept: list[tuple[float, int, int, float, float]] = []
    for score, start_idx, end_idx, start, end in ordered:
        suppressed = any(
            temporal_iou(start, end, k_start, k_end) >= iou_threshold
            for (_s, _si, _ei, k_start, k_end) in kept
        )
        if suppressed:
            continue
        kept.append((score, start_idx, end_idx, start, end))
        if len(kept) >= TOP_K:
            break
    return kept


def generate_candidates(
    sentences: Sequence[Sentence],
    energy_windows: Sequence[EnergyWindow] = (),
    silence_regions: Sequence[SilenceRegion] = (),
    job_id: str = "job",
    niche_keywords: Sequence[str] = (),
) -> list[Candidate]:
    """Enumerate, pre-rank and NMS the candidate windows for a job.

    Pure function: returns a deterministic ``list[Candidate]`` ordered by
    cheap_score DESC, start ASC, end ASC, capped at TOP_K with pairwise
    temporal IoU < NMS_IOU. No I/O is performed.
    """
    if not sentences:
        return []

    windows = _build_windows(sentences)
    if not windows:
        return []

    scored: list[tuple[float, int, int, float, float]] = [
        (
            _score_window(
                start_idx,
                end_idx,
                start,
                end,
                sentences,
                energy_windows,
                silence_regions,
                niche_keywords,
            ),
            start_idx,
            end_idx,
            start,
            end,
        )
        for (start_idx, end_idx, start, end) in windows
    ]

    # Order: cheap_score DESC, then start ASC, then end ASC (by index).
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    kept = _nms(scored, NMS_IOU)

    candidates: list[Candidate] = []
    for score, start_idx, end_idx, start, end in kept:
        candidates.append(
            Candidate(
                cand_id=f"{job_id}-c{start_idx:04d}-{end_idx:04d}",
                start=start,
                end=end,
                sentence_range=(start_idx, end_idx),
                cheap_score=score,
            )
        )
    return candidates
