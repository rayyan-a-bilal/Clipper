"""
M5b pure scoring logic tests (src/clipper/scoring.py) — CPU only, no I/O.

Covers build_context (transcript slice + before/after windows + video
start/end edges), estimate_tokens, pack_batches (budget / batch size /
oversized-candidate context truncation), and validate_scores (id matches,
axis ranges, out-of-bounds suggestions).
"""

import pytest

from clipper.config import Config, LLMConfig
from clipper.models import (
    Candidate,
    CandidateContext,
    ScoredClip,
    ScoringBatch,
    ScoringResponse,
)
from clipper.scoring import (
    SUGGEST_SLACK,
    build_context,
    estimate_tokens,
    pack_batches,
    validate_scores,
)

# -- fixtures ------------------------------------------------------------------


def _sentences(n: int = 12) -> list:
    """12 sentences of 4s each separated by 1s gaps: [0-4], [5-9], ..."""
    return [
        type("S", (), {"start": 5.0 * i, "end": 5.0 * i + 4.0})
        for i in range(n)
    ]


def _sentences_model(n: int = 12):
    from clipper.models import Sentence

    return [
        Sentence(start=5.0 * i, end=5.0 * i + 4.0, text=f"S{i}", word_idx=(i, i))
        for i in range(n)
    ]


def _cfg(context_before_s: float = 10.0, context_after_s: float = 10.0) -> Config:
    return Config(
        llm=LLMConfig(
            context_before_s=context_before_s,
            context_after_s=context_after_s,
        )
    )


def _cand(start_idx: int, end_idx: int, job_id: str = "jobX") -> Candidate:
    sentences = _sentences()
    return Candidate(
        cand_id=f"{job_id}-c{start_idx:04d}-{end_idx:04d}",
        start=sentences[start_idx].start,
        end=sentences[end_idx].end,
        sentence_range=(start_idx, end_idx),
        cheap_score=0.5,
    )


def _ctx(
    cand_id: str,
    transcript: str = "abc",
    before: str = "",
    after: str = "",
    start: float = 0.0,
    end: float = 1.0,
) -> CandidateContext:
    return CandidateContext(
        cand_id=cand_id,
        start=start,
        end=end,
        transcript=transcript,
        context_before=before,
        context_after=after,
    )


def _score(
    cand_id: str,
    in_: float = 0.0,
    out_: float = 1.0,
    hook: int = 5,
    reject_reason: str | None = None,
) -> ScoredClip:
    return ScoredClip(
        cand_id=cand_id,
        hook=hook,
        payoff=5,
        self_contained=5,
        emotion=5,
        clarity=5,
        suggested_in=in_,
        suggested_out=out_,
        reject_reason=reject_reason,
    )


# -- build_context ---------------------------------------------------------------


def test_build_context_transcript_is_inclusive_sentence_slice():
    candidates = _cand(3, 6)
    ctx = build_context(candidates, _sentences_model(), _cfg())
    assert ctx.transcript == "S3 S4 S5 S6"
    assert ctx.cand_id == candidates.cand_id
    assert ctx.start == 15.0
    assert ctx.end == 34.0


def test_build_context_before_and_after_windows():
    # candidate [15, 34]; before = sentences overlapping [5, 15) -> S1 S2
    # after = sentences overlapping (34, 44] -> S7 S8
    ctx = build_context(_cand(3, 6), _sentences_model(), _cfg())
    assert ctx.context_before == "S1 S2"
    assert ctx.context_after == "S7 S8"


def test_build_context_window_sizes_come_from_cfg():
    # candidate [15, 34]; context_before_s = 15 -> floor 0 -> S0 S1 S2
    # context_after_s = 20 -> ceil 54 -> S7 S8 S9 S10
    cfg = _cfg(context_before_s=15.0, context_after_s=20.0)
    ctx = build_context(_cand(3, 6), _sentences_model(), cfg)
    assert ctx.context_before == "S0 S1 S2"
    assert ctx.context_after == "S7 S8 S9 S10"


def test_build_context_candidate_at_video_start_has_no_before():
    ctx = build_context(_cand(0, 2), _sentences_model(), _cfg())
    assert ctx.transcript == "S0 S1 S2"
    assert ctx.context_before == ""
    assert ctx.context_after == "S3 S4"  # (14, 24] overlaps S3 [15-19], S4 [20-24]


def test_build_context_candidate_at_video_end_has_no_after():
    ctx = build_context(_cand(9, 11), _sentences_model(), _cfg())
    assert ctx.transcript == "S9 S10 S11"
    assert ctx.context_after == ""
    # before = sentences overlapping [35, 45) -> S7 [35-39], S8 [40-44]
    assert ctx.context_before == "S7 S8"


def test_build_context_single_sentence_window():
    ctx = build_context(_cand(5, 5), _sentences_model(), _cfg())
    assert ctx.transcript == "S5"


# -- estimate_tokens ---------------------------------------------------------------


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 1
    assert estimate_tokens("a" * 16) == 4
    assert estimate_tokens("a" * 7) == 1
    assert estimate_tokens("x" * 1000) == 250


# -- pack_batches -----------------------------------------------------------------


def test_pack_batches_max_size_and_order():
    contexts = [_ctx(f"c{i}", transcript="abc") for i in range(12)]
    batches = pack_batches(contexts, batch_size=5, max_batch_tokens=6000)
    assert [len(b.candidates) for b in batches] == [5, 5, 2]
    flat = [c.cand_id for b in batches for c in b.candidates]
    assert flat == [f"c{i}" for i in range(12)]
    assert all(b.schema_version == 1 for b in batches)


def test_pack_batches_respects_token_budget():
    # budget = 2000 - 1500 = 500. Each candidate costs ~461 tokens
    # (400 transcript + 2 empty contexts + 60) -> 2 per batch would exceed.
    contexts = [_ctx(f"c{i}", transcript="y" * 1600) for i in range(4)]
    batches = pack_batches(contexts, batch_size=5, max_batch_tokens=2000)
    from clipper.scoring import _candidate_cost

    for b in batches:
        assert sum(_candidate_cost(c) for c in b.candidates) <= 500
    assert [len(b.candidates) for b in batches] == [1, 1, 1, 1]


def test_pack_batches_prompt_overhead_reserve():
    # A batch that fits max_batch_tokens but NOT the reserved budget
    # must NOT be allowed: 450 tokens fits 6000-1500=4500 exactly.
    contexts = [
        _ctx(f"c{i}", transcript="x" * 1600)  # ~461 tokens apiece
        for i in range(9)
    ]
    batches = pack_batches(contexts, batch_size=5, max_batch_tokens=6000)
    from clipper.scoring import _candidate_cost

    for b in batches:
        assert sum(_candidate_cost(c) for c in b.candidates) <= 4500


def test_pack_batches_oversized_single_truncates_context_not_transcript():
    transcript = "t" * 400  # 100 tokens -> base 160, fits budget 500
    ctx = _ctx(
        "big",
        transcript=transcript,
        before="b" * 16000,  # 4000 tokens
        after="a" * 16000,  # 4000 tokens
    )
    batches = pack_batches([ctx], batch_size=5, max_batch_tokens=2000)
    assert len(batches) == 1
    out = batches[0].candidates[0]
    assert out.transcript == transcript  # NEVER truncated
    assert len(out.context_before) < len(ctx.context_before)
    # before took the room; at most 1 token left over for after
    assert len(out.context_after) <= 4
    from clipper.scoring import _candidate_cost

    assert _candidate_cost(out) <= 500


def test_pack_batches_transcript_alone_exceeds_budget_still_gets_own_batch():
    ctx = _ctx("huge", transcript="x" * 64000)  # ~16k tokens >> any budget
    batches = pack_batches([ctx], batch_size=5, max_batch_tokens=2000)
    assert len(batches) == 1
    out = batches[0].candidates[0]
    assert out.transcript == ctx.transcript
    assert out.context_before == ""
    assert out.context_after == ""


def test_pack_batches_empty_input():
    assert pack_batches([], 5, 6000) == []


# -- validate_scores ----------------------------------------------------------------


def _batch2() -> ScoringBatch:
    return ScoringBatch(
        schema_version=1,
        candidates=[
            _ctx("A", start=10.0, end=30.0),
            _ctx("B", start=40.0, end=60.0),
        ],
    )


def test_validate_scores_valid_in_any_order():
    batch = _batch2()
    resp = ScoringResponse(
        scores=[
            _score("B", in_=41.0, out_=59.0, hook=8),
            _score("A", in_=9.0, out_=31.0, hook=7),
        ]
    )
    out = validate_scores(resp, batch)
    # Returned in BATCH CANDIDATE ORDER
    assert [s.cand_id for s in out] == ["A", "B"]
    assert out[0].hook == 7
    assert out[1].hook == 8
    assert all(s.reject_reason is None for s in out)


def test_validate_scores_missing_cand_id_raises():
    batch = _batch2()
    resp = ScoringResponse(scores=[_score("A")])
    with pytest.raises(ValueError, match="missing"):
        validate_scores(resp, batch)


def test_validate_scores_extra_cand_id_raises():
    batch = _batch2()
    resp = ScoringResponse(scores=[_score("A"), _score("B"), _score("Z")])
    with pytest.raises(ValueError, match="extra"):
        validate_scores(resp, batch)


def test_validate_scores_duplicate_cand_id_raises():
    batch = _batch2()
    resp = ScoringResponse(scores=[_score("A"), _score("A")])
    with pytest.raises(ValueError, match="duplicate"):
        validate_scores(resp, batch)


@pytest.mark.parametrize("bad", [11, -1, "7", None, 7.5, True])
def test_validate_scores_bad_axis_raises(bad):
    batch = _batch2()
    scores = []
    for cid in ("A", "B"):
        s = _score(cid)
        scores.append(s.model_copy(update={"hook": bad}))
    resp = ScoringResponse(scores=scores)
    with pytest.raises(ValueError, match="hook"):
        validate_scores(resp, batch)


def test_validate_scores_axis_bounds_ok():
    batch = _batch2()
    resp = ScoringResponse(
        scores=[
            _score("A", hook=0).model_copy(update={"payoff": 10}),
            _score("B", hook=10).model_copy(update={"clarity": 0}),
        ]
    )
    out = validate_scores(resp, batch)
    assert len(out) == 2


def _oob_batch() -> ScoringBatch:
    return ScoringBatch(
        schema_version=1,
        candidates=[_ctx("A", start=10.0, end=30.0)],
    )


def test_validate_scores_out_of_bounds_suggestion_sets_reject_reason():
    batch = _oob_batch()
    # suggested_in = 1.999 < 10.0 - 8.0 = 2.0 -> out of bounds
    resp = ScoringResponse(scores=[_score("A", in_=1.999, out_=28.0, hook=9)])
    out = validate_scores(resp, batch)
    assert len(out) == 1
    assert out[0].reject_reason == "suggestion_out_of_bounds"
    # Scores are KEPT
    assert out[0].hook == 9
    assert out[0].suggested_in == 1.999


def test_validate_scores_out_of_bounds_suggested_out_sets_reject_reason():
    batch = _oob_batch()
    resp = ScoringResponse(scores=[_score("A", in_=12.0, out_=38.001, hook=4)])
    out = validate_scores(resp, batch)
    assert out[0].reject_reason == "suggestion_out_of_bounds"
    assert out[0].hook == 4


def test_validate_scores_suggestion_at_slack_edge_is_in_bounds():
    batch = _oob_batch()
    resp = ScoringResponse(
        scores=[_score("A", in_=10.0 - SUGGEST_SLACK, out_=30.0 + SUGGEST_SLACK)]
    )
    out = validate_scores(resp, batch)
    assert out[0].reject_reason is None


def test_validate_scores_model_reject_reason_preserved():
    batch = _oob_batch()
    resp = ScoringResponse(
        scores=[_score("A", in_=12.0, out_=28.0, reject_reason="frag")]
    )
    out = validate_scores(resp, batch)
    assert out[0].reject_reason == "frag"
