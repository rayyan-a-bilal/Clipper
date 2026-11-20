"""
M5c cut refinement tests (tests/test_cuts.py) — CPU only, pure functions.

Golden cases against the STRICT M5c order (docs/spec/m5.md):

    clamp -> sentence snap -> word padding -> silence snap -> shot snap
    -> validate (start < end, 15..75s, >= 1 full sentence)

Fixture: one 1.6s sentence every 2s (sentence i = [2i, 2i+1.6]), each with
a single word [2i+0.25, 2i+1.5] - so word padding is exactly
start - 0.120 (PAD_START) and end + 0.250 (PAD_END) relative to the snapped
sentence.
"""

import pytest

from clipper.cuts import (
    PAD_END,
    PAD_START,
    SENTENCE_SNAP,
    SHOT_SNAP,
    SILENCE_SNAP,
    SUGGEST_SLACK,
    CutError,
    refine_cut,
)
from clipper.models import (
    Candidate,
    ScoredClip,
    Sentence,
    Shot,
    SilenceRegion,
    Word,
)


def _make_corpus(n: int = 30):
    sentences: list[Sentence] = []
    words: list[Word] = []
    for i in range(n):
        s_start = 2.0 * i
        s_end = 2.0 * i + 1.6
        words.append(
            Word(start=s_start + 0.25, end=s_end - 0.1, text=f"w{i}", prob=0.99)
        )
        sentences.append(
            Sentence(
                start=s_start,
                end=s_end,
                text=f"this is sentence number {i}",
                word_idx=(i, i),
            )
        )
    return sentences, words


def _cand(start: float, end: float) -> Candidate:
    return Candidate(
        cand_id="cut-1",
        start=start,
        end=end,
        sentence_range=(0, max(1, int(end // 2))),
        cheap_score=0.5,
    )


def _score(start: float, end: float) -> ScoredClip:
    return ScoredClip(
        cand_id="cut-1",
        hook=10,
        payoff=10,
        self_contained=10,
        emotion=0,
        clarity=0,
        suggested_in=start,
        suggested_out=end,
        reject_reason=None,
    )


def _refine(cand, score, *, silence=None, shots=None, duration=60.0,
            corpus=None):
    sentences, words = corpus or _make_corpus(30)
    shots = shots if shots is not None else [Shot(start=0.0, end=duration)]
    return refine_cut(
        cand, score, sentences, words,
        silence or [], shots, duration,
    )


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
def test_constants_match_spec():
    assert SUGGEST_SLACK == 8.0
    assert SENTENCE_SNAP == 1.0
    assert PAD_START == 0.120
    assert PAD_END == 0.250
    assert SILENCE_SNAP == 0.300
    assert SHOT_SNAP == 0.400


# ---------------------------------------------------------------------------
# golden cases
# ---------------------------------------------------------------------------
def test_sentence_snap_and_word_padding():
    """suggestion near sentence 3/14 boundaries -> snaps to s3 start and
    s14 end, then word padding (first word .start - PAD_START, last word
    .end + PAD_END)."""
    start, end = _refine(_cand(5.0, 30.0), _score(5.9, 29.0))
    # in 5.9 -> sentence s3 [6.0, 7.6]: word start 6.25 - 0.12 = 6.13
    assert start == pytest.approx(6.25 - PAD_START)
    # out 29.0 -> sentence s14 [28.0, 29.6]: word end 29.5 + 0.25 = 29.75
    assert end == pytest.approx(29.5 + PAD_END)


def test_silence_snap():
    """start/end are each moved to the nearby silence edge (<=0.3s)."""
    silence = [
        SilenceRegion(start=6.0, end=6.3),
        SilenceRegion(start=29.6, end=29.9),
    ]
    start, end = _refine(_cand(5.0, 30.0), _score(5.9, 29.0), silence=silence)
    assert start == pytest.approx(6.0)   # 6.13 -> edge 6.0 (d=0.13)
    assert end == pytest.approx(29.6)    # 29.75 -> edge 29.6 (d=0.15)


def test_shot_snap():
    """start moves to the nearby shot boundary (<=0.4s); end unaffected."""
    shots = [Shot(start=0.0, end=6.5), Shot(start=6.5, end=60.0)]
    start, end = _refine(_cand(5.0, 30.0), _score(5.9, 29.0), shots=shots)
    assert start == pytest.approx(6.5)   # 6.13 -> boundary 6.5 (d=0.37)
    assert end == pytest.approx(29.75)


def test_shot_beyond_tolerance_is_ignored():
    shots = [Shot(start=0.0, end=7.0), Shot(start=7.0, end=60.0)]
    start, end = _refine(_cand(5.0, 30.0), _score(5.9, 29.0), shots=shots)
    assert start == pytest.approx(6.13)  # |7.0 - 6.13| = 0.87 > 0.4 -> no snap
    assert end == pytest.approx(29.75)


def test_clamping_outside_candidate_slack():
    """suggestions beyond cand window +-8s clamp back to the window edge."""
    # sug_out 40.3 -> clamped to cand.end+8 = 38; sug_in 100 -> 38 too
    cand, score = _cand(5.0, 30.0), _score(100.0, 40.3)
    with pytest.raises(CutError) as exc:
        _refine(cand, score)
    assert "inverted" in str(exc.value)


def test_clamping_below_zero():
    """suggested_in = -10 clamps to 0 -> sentence s0 -> word padding, then
    the default [0, duration] shot boundary snaps it to exactly 0.0."""
    start, end = _refine(_cand(5.0, 30.0), _score(-10.0, 29.0))
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(29.75)


def test_clamping_word_padding_when_no_shot_edge():
    """same clamp, with shots starting at 1.0 (no boundary near 0.13): the
    word-padded value survives."""
    start, end = _refine(
        _cand(5.0, 30.0), _score(-10.0, 29.0),
        shots=[Shot(start=1.0, end=60.0)],
    )
    assert start == pytest.approx(0.25 - PAD_START)
    assert end == pytest.approx(29.75)


def test_mid_gap_suggestion_snaps_to_nearest_boundary():
    """suggestion in the gap between s2 [4, 5.6] and s3 [6, 7.6]."""
    start, end = _refine(_cand(5.0, 30.0), _score(5.8, 29.0))
    # nearest boundary within SENTENCE_SNAP: s3.start (d=0.2) -> word pad
    assert start == pytest.approx(6.25 - PAD_START)
    assert end == pytest.approx(29.75)


def test_too_short_rejected():
    cand, score = _cand(5.0, 30.0), _score(10.4, 11.0)
    with pytest.raises(CutError) as exc:
        _refine(cand, score)
    assert "too short" in str(exc.value)


def test_too_long_rejected():
    """suggestion clamp allows up to 75s of a 15s window; a 88s span after
    sentence snap exceeds MAX_CLIP_S and is dropped."""
    corpus = _make_corpus(60)  # covers to ~120s
    cand = Candidate(
        cand_id="cut-1", start=2.0, end=80.0, sentence_range=(0, 10),
        cheap_score=0.5,
    )
    score = _score(0.3, 100.0)
    with pytest.raises(CutError) as exc:
        _refine(cand, score, corpus=corpus, duration=120.0)
    assert "too long" in str(exc.value)


def test_inverted_suggestion_rejected():
    with pytest.raises(CutError) as exc:
        _refine(_cand(5.0, 30.0), _score(20.0, 10.0))
    assert "inverted" in str(exc.value)


def test_no_sentences_rejected():
    with pytest.raises(CutError) as exc:
        refine_cut(
            _cand(0.0, 10.0), _score(0.0, 10.0),
            [], [Word(start=0.1, end=0.5, text="x", prob=1.0)],
            [], [Shot(start=0.0, end=60.0)], 60.0,
        )
    assert "no sentences" in str(exc.value)


def test_result_bounds_and_full_sentence():
    """sanity invariants of a successful refine."""
    start, end = _refine(_cand(5.0, 30.0), _score(5.9, 29.0))
    sentences, words = _make_corpus(30)
    assert 0.0 <= start < end <= 60.0
    assert 15.0 <= end - start <= 75.0
    assert any(s.start >= start and s.end <= end for s in sentences)
    # start/end land at word edges (padding side), so both are inside
    # plausible word spans expanded by padding
    assert start >= 0.25 - PAD_START - 1e-9
    assert end <= 59.5 + PAD_END + 1e-9
    assert words  # corpus non-trivial
