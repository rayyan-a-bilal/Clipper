"""M5c — pure clip selection (NO I/O, NO LLM calls, NO DB).

Implements the fixed M5 selection rules from ``docs/spec/m5.md``:

- ``weighted_score(score)`` — the M5b weighted formula (delegates to
  ``clipper.scoring.weighted_score`` so the weights CANNOT drift).
- ``passes_gate(score, cfg) -> (bool, reasons)`` — the M5 quality gate:
  ``weighted >= 7.2 AND hook >= 6 AND self_contained >= 7`` (fixed by
  docs/spec/m5.md; ``cfg`` is accepted for a stable signature and future
  config-driven tuning).
- ``temporal_iou(a_start, a_end, b_start, b_end)`` — re-exported verbatim
  from the M4a pure helpers (clipper.candidates) so the two cannot drift.
- ``select_clips(candidates, scores, cfg) -> SelectionResult``:
  eligible = reject_reason is None AND all axes int 0..10 AND passes_gate;
  rank by weighted DESC, then cheap_score DESC, then start ASC (cand_id as
  the final deterministic tie-break); a candidate whose temporal IoU against
  ANY already-selected candidate is > 0.30 (SELECTION_IOU) is suppressed —
  not selected; take ``cfg.clips.target_count`` (default 3), hard-capped at
  ``cfg.clips.max_count`` (default 5, via ``min(target, max)``).
  Returns the selected candidates (rank order), the full ranked eligible
  list (used by the stage for backfill after CutErrors) and one decision
  record per candidate.

Zero selected clips is a VALID outcome (never fabricate clips).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import temporal_iou  # noqa: F401  (re-export, fixed M4a helper)
from .config import Config
from .models import Candidate, ScoredClip
from .scoring import weighted_score as _scoring_weighted_score

# ---------------------------------------------------------------------------
# Constants (fixed by docs/spec/m5.md — do NOT drift)
# ---------------------------------------------------------------------------
SELECTION_IOU = 0.30  # suppress when temporal IoU vs selected is STRICTLY > 0.30

# (defaults fixed by docs/spec/m5.md; production values come from
# ``cfg.clips.gate.*`` — these constants mirror the spec defaults)
GATE_MIN_WEIGHTED = 7.2
GATE_MIN_HOOK = 6
GATE_MIN_SELF_CONTAINED = 7
_GATE_EPS = 1e-9  # float tolerance for the 7.2 weighted boundary
_AXIS_NAMES = ("hook", "payoff", "self_contained", "emotion", "clarity")


def weighted_score(score: ScoredClip) -> float:
    """M5 weighted score: 0.30*hook + 0.25*payoff + 0.20*self_contained
    + 0.15*emotion + 0.10*clarity (docs/spec/m5.md)."""
    return _scoring_weighted_score(score)


def passes_gate(score: ScoredClip, cfg: Config) -> tuple[bool, list[str]]:
    """M5 quality gate.

    Passes IFF: ``weighted >= cfg.clips.gate.weighted AND hook >=
    cfg.clips.gate.hook AND self_contained >= cfg.clips.gate.self_contained``
    (spec defaults 7.2 / 6 / 7 per docs/spec/m5.md; a 1e-9 tolerance
    absorbs float noise at the weighted boundary). Returns
    ``(True, [])`` when passing, else ``(False, [reason, ...])`` with one
    reason per violated axis.
    """
    gate = cfg.clips.gate
    reasons: list[str] = []
    weighted = weighted_score(score)
    if weighted < gate.weighted - _GATE_EPS:
        reasons.append(f"weighted {weighted:.2f} < {gate.weighted}")
    if score.hook < gate.hook:
        reasons.append(f"hook {score.hook} < {gate.hook}")
    if score.self_contained < gate.self_contained:
        reasons.append(
            f"self_contained {score.self_contained} "
            f"< {gate.self_contained}"
        )
    return (not reasons), reasons


@dataclass
class SelectionDecision:
    """Per-candidate decision record (pure; the stage maps this to the
    pydantic ``ClipDecision`` contract)."""

    cand_id: str
    weighted: float
    eligible: bool
    gate_reasons: list[str] = field(default_factory=list)
    suppressed_by: str | None = None
    selected: bool = False


@dataclass
class SelectionResult:
    """Result of :func:`select_clips`.

    - ``selected`` — the picked candidates in rank order (after greedy
      temporal-IoU suppression), at most ``min(target, max)`` entries.
    - ``ranked`` — ALL eligible candidates in rank order (the order the
      stage walks for CutError backfill).
    - ``decisions`` — one :class:`SelectionDecision` per candidate, in
      original (artifact) candidate order.
    """

    selected: list[Candidate]
    ranked: list[Candidate]
    decisions: list[SelectionDecision]


def _axes_in_bounds(score: ScoredClip) -> bool:
    for name in _AXIS_NAMES:
        value = getattr(score, name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 10
        ):
            return False
    return True


def _eligible(
    score: ScoredClip | None, cfg: Config,
) -> tuple[bool, float, list[str]]:
    # eligible = reject_reason is None AND axes in 0..10 AND passes_gate
    if score is None:
        return False, 0.0, ["missing_score"]
    if score.reject_reason is not None:
        return False, weighted_score(score), [f"scored_out: {score.reject_reason}"]
    if not _axes_in_bounds(score):
        return False, weighted_score(score), ["axes_out_of_bounds"]
    ok, reasons = passes_gate(score, cfg)
    return ok, weighted_score(score), reasons


def select_clips(
    candidates: list[Candidate],
    scores: list[ScoredClip],
    cfg: Config,
) -> SelectionResult:
    """Rank, gate, dedupe and pick clips (see module docstring for rules).

    Args:
        candidates: the M4b candidate list (artifact order).
        scores: the M5b scores (one per candidate, matched by cand_id).
        cfg: configuration; target = ``cfg.clips.target_count`` (default 3),
            hard cap = ``cfg.clips.max_count`` (default 5).

    Returns:
        A :class:`SelectionResult`; ``selected`` may be empty (a VALID
        success per docs/spec/m5.md).
    """
    limit = max(0, min(cfg.clips.target_count, cfg.clips.max_count))

    records: list[tuple[Candidate, SelectionDecision]] = []
    eligible: list[tuple[Candidate, SelectionDecision]] = []
    for cand in candidates:
        score = next((s for s in scores if s.cand_id == cand.cand_id), None)
        ok, weighted, reasons = _eligible(score, cfg)
        decision = SelectionDecision(
            cand_id=cand.cand_id,
            weighted=weighted,
            eligible=ok,
            gate_reasons=reasons,
        )
        records.append((cand, decision))
        if ok:
            eligible.append((cand, decision))

    # Rank: weighted DESC, cheap_score DESC, start ASC (cand_id last).
    eligible.sort(
        key=lambda rc: (
            -rc[1].weighted,
            -rc[0].cheap_score,
            rc[0].start,
            rc[0].cand_id,
        )
    )

    # Greedy: keep a candidate only if its temporal IoU against EVERY
    # already-selected candidate is <= SELECTION_IOU.
    picked: list[Candidate] = []
    for cand, decision in eligible:
        if len(picked) >= limit:
            decision.selected = False
            continue
        clash = next(
            (
                sel
                for sel in picked
                if temporal_iou(sel.start, sel.end, cand.start, cand.end)
                > SELECTION_IOU
            ),
            None,
        )
        if clash is not None:
            decision.selected = False
            decision.suppressed_by = clash.cand_id
        else:
            picked.append(cand)
            decision.selected = True

    return SelectionResult(
        selected=picked,
        ranked=[cand for cand, _ in eligible],
        decisions=[decision for _, decision in records],
    )


__all__ = [
    "SELECTION_IOU",
    "GATE_MIN_WEIGHTED",
    "GATE_MIN_HOOK",
    "GATE_MIN_SELF_CONTAINED",
    "SelectionDecision",
    "SelectionResult",
    "passes_gate",
    "select_clips",
    "temporal_iou",
    "weighted_score",
]
