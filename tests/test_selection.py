"""
M5c pure selection tests (tests/test_selection.py) — CPU only.

Covers: weighted_score exact math (docs/spec/m5.md), passes_gate boundary
values, temporal_iou correctness, suppression at IoU > 0.30, ranking
tie-break determinism, eligibility rules, target/max cap, zero-eligible
valid success, decision records.
"""

import pytest

from clipper.candidates import temporal_iou as _cand_iou
from clipper.config import Config, PathsConfig
from clipper.models import Candidate, ScoredClip
from clipper.selection import (
    GATE_MIN_HOOK,
    GATE_MIN_SELF_CONTAINED,
    GATE_MIN_WEIGHTED,
    SELECTION_IOU,
    SelectionResult,
    passes_gate,
    select_clips,
    temporal_iou,
    weighted_score,
)

JOB = "job-seltest"


def _cfg(tmp_path) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(data_dir=str(data_dir)))


def _score(
    cand_id: str = "x",
    hook: int = 8,
    payoff: int = 8,
    self_contained: int = 10,
    emotion: int = 4,
    clarity: int = 0,
    suggested_in: float = 0.0,
    suggested_out: float = 20.0,
    reject_reason: str | None = None,
) -> ScoredClip:
    return ScoredClip(
        cand_id=cand_id,
        hook=hook,
        payoff=payoff,
        self_contained=self_contained,
        emotion=emotion,
        clarity=clarity,
        suggested_in=suggested_in,
        suggested_out=suggested_out,
        reject_reason=reject_reason,
    )


def _cand(i: int, start: float, end: float, cheap: float = 0.5) -> Candidate:
    return Candidate(
        cand_id=f"{JOB}-c{i:04d}-{i + 1:04d}",
        start=start,
        end=end,
        sentence_range=(i, i + 1),
        cheap_score=cheap,
    )


# ---------------------------------------------------------------------------
# weighted_score (exact spec math)
# ---------------------------------------------------------------------------
def test_weighted_score_exact_math():
    s = _score(hook=8, payoff=8, self_contained=10, emotion=4, clarity=2)
    expected = 0.30 * 8 + 0.25 * 8 + 0.20 * 10 + 0.15 * 4 + 0.10 * 2
    assert weighted_score(s) == expected
    assert weighted_score(s) == pytest.approx(7.2)


def test_weighted_score_zero_and_tenths():
    assert weighted_score(_score(hook=0, payoff=0, self_contained=0,
                                 emotion=0, clarity=0)) == 0.0
    s = _score(hook=1, payoff=1, self_contained=1, emotion=1, clarity=1)
    assert weighted_score(s) == pytest.approx(1.0)
    s = _score(hook=10, payoff=10, self_contained=10, emotion=10, clarity=10)
    assert weighted_score(s) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# passes_gate (boundary values, independent axes)
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)
    return _cfg(tmp_path)


def test_gate_pass_at_exact_boundaries(cfg):
    # weighted == 7.2 exactly, hook == 6, self_contained == 7 -> PASSES
    s = _score(cand_id="g", hook=6, payoff=6, self_contained=7,
               emotion=10, clarity=10)
    assert weighted_score(s) == pytest.approx(GATE_MIN_WEIGHTED)
    ok, reasons = passes_gate(s, cfg)
    assert ok is True
    assert reasons == []


def test_gate_fail_just_below_weighted(cfg):
    # same axes except clarity 9 -> weighted 7.1 < 7.2
    s = _score(cand_id="g", hook=6, payoff=6, self_contained=7,
               emotion=10, clarity=9)
    assert weighted_score(s) == pytest.approx(7.1)
    ok, reasons = passes_gate(s, cfg)
    assert ok is False
    assert any("weighted" in r for r in reasons)


def test_gate_fail_hook_below_and_pass_at_boundary(cfg):
    # weighted comfortably >= 7.2 so ONLY hook is exercised
    base = {"payoff": 10, "self_contained": 7, "emotion": 10, "clarity": 10}
    fail = _score(cand_id="g", hook=5, **base)
    assert weighted_score(fail) >= GATE_MIN_WEIGHTED  # 7.9
    ok, reasons = passes_gate(fail, cfg)
    assert ok is False
    assert [r for r in reasons if "weighted" in r] == []
    assert len(reasons) == 1 and "hook" in reasons[0]
    assert GATE_MIN_HOOK == 6
    ok2, reasons2 = passes_gate(_score(cand_id="g", hook=6, **base), cfg)
    assert ok2 is True and reasons2 == []


def test_gate_fail_self_contained_below_and_pass_at_boundary(cfg):
    # weighted comfortably >= 7.2 so only self_contained is tested
    ok, reasons = passes_gate(
        _score(cand_id="g", hook=10, payoff=10, self_contained=6,
               emotion=10, clarity=10),
        cfg,
    )
    assert ok is False
    assert any(str(GATE_MIN_SELF_CONTAINED) in r for r in reasons)
    ok2, reasons2 = passes_gate(
        _score(cand_id="g", hook=10, payoff=10, self_contained=7,
               emotion=10, clarity=10),
        cfg,
    )
    assert ok2 is True and reasons2 == []


def test_gate_reasons_one_per_violated_axis(cfg):
    s = _score(cand_id="g", hook=3, payoff=1, self_contained=4,
               emotion=2, clarity=1)
    _, reasons = passes_gate(s, cfg)
    assert len(reasons) == 3  # weighted, hook, self_contained
    assert any("weighted" in r for r in reasons)
    assert any("hook" in r for r in reasons)
    assert any("self_contained" in r for r in reasons)


# ---------------------------------------------------------------------------
# temporal_iou (re-exported helper = M4a pure helper)
# ---------------------------------------------------------------------------
def test_temporal_iou_values():
    assert temporal_iou(0.0, 10.0, 5.0, 15.0) == pytest.approx(5.0 / 15.0)
    assert temporal_iou(0.0, 10.0, 20.0, 30.0) == 0.0
    assert temporal_iou(5.0, 10.0, 5.0, 15.0) == pytest.approx(5.0 / 10.0)
    assert temporal_iou(0.0, 10.0, 0.0, 10.0) == pytest.approx(1.0)
    assert temporal_iou(0.0, 10.0, 9.0, 11.0) == pytest.approx(1.0 / 11.0)


def test_temporal_iou_reexport_is_m4_helper():
    assert temporal_iou is _cand_iou


# ---------------------------------------------------------------------------
# select_clips — eligibility, ranking, suppression, caps
# ---------------------------------------------------------------------------
def _passing(hook: int = 10, payoff: int = 10, self_contained: int = 10) -> ScoredClip:
    # weighted = 7.5 (hook/payoff/self_contained = 10) -> clears the gate
    return _score(
        cand_id="x",
        hook=hook,
        payoff=payoff,
        self_contained=self_contained,
        emotion=0,
        clarity=0,
    )


def _scored(score: ScoredClip, **overrides) -> ScoredClip:
    """Copy a ScoredClip with field overrides (re-validated)."""
    return ScoredClip.model_validate({**score.model_dump(), **overrides})


def test_eligibility_rules_all_paths(cfg):
    cands = [
        _cand(0, 0.0, 20.0),
        _cand(1, 30.0, 50.0),
        _cand(2, 60.0, 80.0),
        _cand(3, 90.0, 110.0),
        _cand(4, 120.0, 140.0),
    ]
    good = _passing()
    scores = [
        _scored(good, cand_id=cands[0].cand_id),
        _scored(
            good, cand_id=cands[1].cand_id, reject_reason="suggestion_out_of_bounds"
        ),
        _scored(good, cand_id=cands[2].cand_id, hook=3),
        _scored(good, cand_id=cands[3].cand_id, payoff=42),
    ]
    # cands[4] has NO score entry at all
    res = select_clips(cands, scores, cfg)
    by_id = {d.cand_id: d for d in res.decisions}
    assert by_id[cands[0].cand_id].eligible is True
    assert by_id[cands[1].cand_id].eligible is False
    assert any(
        "reject" in r or "scored_out" in r for r in by_id[cands[1].cand_id].gate_reasons
    )
    assert by_id[cands[2].cand_id].eligible is False
    assert any("hook" in r for r in by_id[cands[2].cand_id].gate_reasons)
    assert by_id[cands[3].cand_id].eligible is False
    assert any("bounds" in r for r in by_id[cands[3].cand_id].gate_reasons)
    assert by_id[cands[4].cand_id].eligible is False
    assert "missing_score" in by_id[cands[4].cand_id].gate_reasons
    assert [c.cand_id for c in res.selected] == [cands[0].cand_id]


def test_ranking_weighted_then_cheap_then_start(cfg):
    # equal weighted (7.5), different cheap -> higher cheap first
    hi_cheap = _passing()
    scores = [
        _scored(hi_cheap, cand_id="low", clarity=0),
        _scored(hi_cheap, cand_id="high", clarity=0),
    ]
    low = Candidate(cand_id="low", start=10.0, end=30.0,
                    sentence_range=(0, 1), cheap_score=0.2)
    high = Candidate(cand_id="high", start=40.0, end=60.0,
                     sentence_range=(2, 3), cheap_score=0.9)
    res = select_clips([low, high], scores, cfg)
    assert [c.cand_id for c in res.selected] == ["high", "low"]


def test_ranking_same_weighted_and_cheap_start_ascending(cfg):
    s = _passing()
    scores = [
        ScoredClip.model_validate({**s.model_dump(), "cand_id": "b"}),
        ScoredClip.model_validate({**s.model_dump(), "cand_id": "a"}),
    ]
    a = Candidate(cand_id="a", start=10.0, end=30.0,
                  sentence_range=(0, 1), cheap_score=0.5)
    b = Candidate(cand_id="b", start=40.0, end=60.0,
                  sentence_range=(2, 3), cheap_score=0.5)
    res = select_clips([b, a], scores, cfg)  # input order reversed on purpose
    assert [c.cand_id for c in res.selected] == ["a", "b"]


def test_suppression_strictly_above_030(cfg):
    # A [0,20], B [5,25]: IoU = 15/40 = 0.375 > 0.30 -> B suppressed by A
    a = _cand(0, 0.0, 20.0)
    b = _cand(1, 5.0, 25.0)
    scores = [
        _scored(_passing(), cand_id=a.cand_id, clarity=10),
        _scored(_passing(), cand_id=b.cand_id, clarity=0),
    ]
    res = select_clips([a, b], scores, cfg)
    assert [c.cand_id for c in res.selected] == [a.cand_id]
    by_id = {d.cand_id: d for d in res.decisions}
    assert by_id[b.cand_id].eligible is True
    assert by_id[b.cand_id].selected is False
    assert by_id[b.cand_id].suppressed_by == a.cand_id
    assert by_id[a.cand_id].suppressed_by is None


def test_not_suppressed_at_or_below_030(cfg):
    # A [0,20], B [18,38]: IoU = 2/38 = 0.0526 <= 0.30 -> BOTH selected
    a = _cand(0, 0.0, 20.0)
    b = _cand(1, 18.0, 38.0)
    scores = [
        ScoredClip.model_validate({**_passing().model_dump(), "cand_id": a.cand_id}),
        ScoredClip.model_validate({**_passing().model_dump(), "cand_id": b.cand_id}),
    ]
    res = select_clips([a, b], scores, cfg)
    assert {c.cand_id for c in res.selected} == {a.cand_id, b.cand_id}


def test_target_cap_of_three(cfg):
    # 5 eligible, non-overlapping candidates -> only 3 selected (target=3 < max=5)
    cands = [_cand(i, 25.0 * i, 25.0 * i + 20.0) for i in range(5)]
    scores = [
        ScoredClip.model_validate({**_passing().model_dump(), "cand_id": c.cand_id})
        for c in cands
    ]
    res = select_clips(cands, scores, cfg)
    assert len(res.selected) == 3
    assert len(res.ranked) == 5
    # first 3 in rank order (all equal weighted/cheap -> start ASC)
    expected = [c.cand_id for c in sorted(cands, key=lambda c: c.start)[:3]]
    assert [c.cand_id for c in res.selected] == expected


def test_gate_thresholds_come_from_config(cfg):
    """Gate thresholds are read from cfg.clips.gate (defaults per m5.md).

    Score: hook=7, payoff=10, self_contained=7, emotion=10, clarity=10
    -> weighted = 2.1 + 2.5 + 1.4 + 1.5 + 1.0 = 8.5 (passes default gate).
    """
    s = _score(cand_id="g", hook=7, payoff=10, self_contained=7,
               emotion=10, clarity=10)
    assert passes_gate(s, cfg)[0] is True

    cfg.clips.gate.weighted = 9.0  # 8.5 < 9.0 -> only weighted violated
    ok, reasons = passes_gate(s, cfg)
    assert ok is False
    assert [r for r in reasons if "weighted" in r] and len(reasons) == 1
    cfg.clips.gate.weighted = 7.2  # restore spec default

    cfg.clips.gate.self_contained = 8  # 7 < 8 -> self_contained violated
    ok, reasons = passes_gate(s, cfg)
    assert ok is False
    assert [r for r in reasons if "self_contained" in r] and len(reasons) == 1
    cfg.clips.gate.self_contained = 7  # restore spec default

    cfg.clips.gate.hook = 8  # 7 < 8 -> hook violated
    ok, reasons = passes_gate(s, cfg)
    assert ok is False
    assert [r for r in reasons if "hook" in r] and len(reasons) == 1
    cfg.clips.gate.hook = 6  # restore spec default

    # restored -> passes the (spec default) gate again
    assert passes_gate(s, cfg)[0] is True


def test_zero_eligible_is_valid_empty_selection(cfg):
    cands = [_cand(0, 0.0, 20.0), _cand(1, 40.0, 60.0)]
    scores = [
        _scored(_passing(), cand_id=c.cand_id, self_contained=4) for c in cands
    ]
    res: SelectionResult = select_clips(cands, scores, cfg)
    assert res.selected == []
    assert res.ranked == []
    assert all(not d.selected and not d.eligible for d in res.decisions)


def test_decisions_one_per_candidate_in_candidate_order(cfg):
    cands = [_cand(i, 20.0 * i, 20.0 * i + 15.0) for i in range(4)]
    scores = [
        _scored(_passing(), cand_id=cands[0].cand_id),
        _scored(_passing(), cand_id=cands[1].cand_id, hook=2),
        _scored(_passing(), cand_id=cands[2].cand_id),
    ]
    res = select_clips(cands, scores, cfg)
    assert [d.cand_id for d in res.decisions] == [c.cand_id for c in cands]
    eligible_only = [d for d in res.decisions if d.eligible]
    assert res.selected and all(d.gate_reasons == [] for d in eligible_only)


def test_constants_match_spec():
    assert SELECTION_IOU == 0.30
    assert GATE_MIN_WEIGHTED == 7.2
    assert GATE_MIN_HOOK == 6
    assert GATE_MIN_SELF_CONTAINED == 7
