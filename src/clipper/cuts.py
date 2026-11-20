"""M5c — pure cut refinement (NO I/O, NO LLM calls, NO DB).

Implements the fixed M5c rules and STRICT order from ``docs/spec/m5.md``:

```
clamp suggestion -> sentence snap -> word padding ->
silence snap -> shot snap -> validate 15..75s
```

i.e. each step refines the previous one:

1. **clamp**   — the LLM suggestion is clamped to
   ``[candidate.start - SUGGEST_SLACK, candidate.end + SUGGEST_SLACK]``
   and then to ``[0, duration]``.
2. **sentence snap** — each boundary is snapped to the nearest sentence
   boundary (a sentence START for the in-point, a sentence END for the
   out-point) within ``SENTENCE_SNAP``; when nothing is closer the
   containing sentence (or, last, the nearest one) is used.
3. **word padding**   — ``start = first word of the start sentence .start
   - PAD_START`` and ``end = last word of the end sentence .end + PAD_END``
   (words addressed through ``Sentence.word_idx`` over the flat words
   artifact, with a time-span fallback), then clamped to ``[0, duration]``.
4. **silence snap** — each boundary is moved to the nearest silence region
   edge within ``SILENCE_SNAP``.
5. **shot snap**    — each boundary is moved to the nearest shot boundary
   within ``SHOT_SNAP``.
6. **validate**     — ``start < end``, ``15.0 <= end - start <= 75.0``, and
   the clip contains at least one FULL sentence; a violation raises
   :class:`CutError` with a specific reason (the caller drops the candidate
   and backfills — it never crashes the job).
"""

from __future__ import annotations

from .models import Candidate, ScoredClip, Sentence, Shot, SilenceRegion, Word

# ---------------------------------------------------------------------------
# Cut constants (fixed verbatim by docs/spec/m5.md — do NOT drift)
# ---------------------------------------------------------------------------
SUGGEST_SLACK = 8.0    # LLM suggestion may deviate from candidate by <= 8s
SENTENCE_SNAP = 1.0    # snap suggestion to the nearest sentence boundary <= 1.0s
PAD_START = 0.120      # word-level padding before the first kept word (s)
PAD_END = 0.250        # word-level padding after the last kept word (s)
SILENCE_SNAP = 0.300   # snap to nearby silence region edge (s)
SHOT_SNAP = 0.400      # snap to nearby shot boundary (s)
MIN_CLIP_S = 15.0
MAX_CLIP_S = 75.0
_EPS = 1e-3


class CutError(Exception):
    """Raised when a suggestion cannot be refined into a valid clip.

    ``str(error)`` carries the specific reason (``too short`` /
    ``too long`` / ``no full sentence`` / ``inverted`` / ``no sentences``)
    for decision records and the rejection summary.
    """


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _snap_sentence_start(t: float, sentences: list[Sentence]) -> Sentence:
    """The sentence an in-point ``t`` belongs to (START side).

    Prefer sentences whose START is within ``SENTENCE_SNAP`` of ``t``
    (nearest one); else the containing sentence(s) (nearest start);
    else the single nearest sentence by start distance.
    """
    near = [s for s in sentences if abs(s.start - t) <= SENTENCE_SNAP]
    if not near:
        near = [s for s in sentences if s.start <= t <= s.end]
    if not near:
        near = [
            min(sentences, key=lambda s: (abs(s.start - t), s.start))
        ]
    return min(near, key=lambda s: (abs(s.start - t), s.start))


def _snap_sentence_end(t: float, sentences: list[Sentence]) -> Sentence:
    """The sentence an out-point ``t`` belongs to (END side)."""
    near = [s for s in sentences if abs(s.end - t) <= SENTENCE_SNAP]
    if not near:
        near = [s for s in sentences if s.start <= t <= s.end]
    if not near:
        near = [
            min(sentences, key=lambda s: (abs(s.end - t), s.end))
        ]
    return min(near, key=lambda s: (abs(s.end - t), s.end))


def _first_word_start(sentence: Sentence, words: list[Word]) -> float:
    lo = sentence.word_idx[0]
    if 0 <= lo < len(words):
        return words[lo].start
    span = [
        w.start
        for w in words
        if w.start >= sentence.start - _EPS and w.end <= sentence.end + _EPS
    ]
    return min(span) if span else float(sentence.start)


def _last_word_end(sentence: Sentence, words: list[Word]) -> float:
    hi = sentence.word_idx[1]
    if 0 <= hi < len(words):
        return words[hi].end
    span = [
        w.end
        for w in words
        if w.start >= sentence.start - _EPS and w.end <= sentence.end + _EPS
    ]
    return max(span) if span else float(sentence.end)


def _snap_edge(value: float, edges: list[float], tolerance: float) -> float:
    """Move ``value`` to the nearest boundary within ``tolerance``."""
    best = None
    for edge in edges:
        distance = abs(edge - value)
        if distance <= tolerance and (best is None or distance < best[0]):
            best = (distance, edge)
    return best[1] if best is not None else value


def refine_cut(
    candidate: Candidate,
    score: ScoredClip,
    sentences: list[Sentence],
    words: list[Word],
    silence_regions: list[SilenceRegion],
    shots: list[Shot],
    duration_s: float,
) -> tuple[float, float]:
    """Refine one LLM suggestion into a final ``(start, end)`` cut.

    Follows the strict M5c order (clamp -> sentence snap -> word padding ->
    silence snap -> shot snap -> validate). Raises :class:`CutError` with a
    specific reason on any violation — the caller rejects this candidate and
    backfills from the next eligible one (never crashes the job).
    """
    if not sentences:
        raise CutError("no sentences available to refine the cut")

    duration = float(duration_s)

    # 1. clamp suggestion to [cand.start - 8, cand.end + 8] then [0, duration]
    lower = candidate.start - SUGGEST_SLACK
    upper = candidate.end + SUGGEST_SLACK
    start = _clamp(score.suggested_in, lower, upper)
    end = _clamp(score.suggested_out, lower, upper)
    start = _clamp(start, 0.0, duration)
    end = _clamp(end, 0.0, duration)
    if start >= end:
        raise CutError(
            "inverted suggestion: suggested_in >= suggested_out after clamping"
        )

    # 2. sentence snap (start boundary -> sentence start, end -> sentence end)
    start_sentence = _snap_sentence_start(start, sentences)
    end_sentence = _snap_sentence_end(end, sentences)

    # 3. word padding, then clamp to [0, duration]
    start = _first_word_start(start_sentence, words) - PAD_START
    end = _last_word_end(end_sentence, words) + PAD_END
    start = _clamp(start, 0.0, duration)
    end = _clamp(end, 0.0, duration)

    # 4. silence snap (nearest region edge within SILENCE_SNAP)
    silence_edges = [edge for r in silence_regions for edge in (r.start, r.end)]
    start = _snap_edge(start, silence_edges, SILENCE_SNAP)
    end = _snap_edge(end, silence_edges, SILENCE_SNAP)

    # 5. shot snap (nearest shot boundary within SHOT_SNAP)
    shot_edges = [edge for s in shots for edge in (s.start, s.end)]
    start = _snap_edge(start, shot_edges, SHOT_SNAP)
    end = _snap_edge(end, shot_edges, SHOT_SNAP)

    start = _clamp(start, 0.0, duration)
    end = _clamp(end, 0.0, duration)

    # 6. validate LAST
    if not start < end:
        raise CutError("cut start >= end after snapping (inverted cut)")
    duration_cut = end - start
    if duration_cut < MIN_CLIP_S:
        raise CutError(f"too short: {duration_cut:.3f}s < {MIN_CLIP_S:.1f}s")
    if duration_cut > MAX_CLIP_S:
        raise CutError(f"too long: {duration_cut:.3f}s > {MAX_CLIP_S:.1f}s")
    if not any(
        s.start >= start - _EPS and s.end <= end + _EPS for s in sentences
    ):
        raise CutError(
            "clip contains no full sentence (needs >= 1 full sentence)"
        )

    return (round(start, 3), round(end, 3))


__all__ = [
    "SUGGEST_SLACK",
    "SENTENCE_SNAP",
    "PAD_START",
    "PAD_END",
    "SILENCE_SNAP",
    "SHOT_SNAP",
    "MIN_CLIP_S",
    "MAX_CLIP_S",
    "CutError",
    "refine_cut",
]
