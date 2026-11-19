"""
M4a candidate-generation foundations tests — CPU only.

No ffmpeg, no GPU, no LLM. Exercises the pure logic in
``clipper.candidates``, the new ``CandidatesArtifact`` contract, and the
committed golden fixture under ``tests/fixtures/m4/``.
"""

import json
import random
import re
import statistics
from pathlib import Path

from clipper.candidates import (
    MAX_CLIP_S,
    MAX_SENTENCES,
    MIN_CLIP_S,
    MIN_SENTENCES,
    NMS_IOU,
    TOP_K,
    W_DENSITY,
    W_ENERGY,
    W_HOOK,
    W_NICHE,
    W_SILENCE,
    _nms,
    cheap_score,
    density_score,
    energy_score,
    generate_candidates,
    hook_score,
    niche_score,
    silence_score,
    temporal_iou,
)
from clipper.models import (
    Candidate,
    CandidatesArtifact,
    EnergyWindow,
    Sentence,
    SilenceRegion,
)

FIXTURES = Path("tests/fixtures/m4")


def _s(start: float, end: float, text: str = "word") -> Sentence:
    return Sentence(start=start, end=end, text=text, word_idx=(0, 0))


def _load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _fixture_inputs():
    sents = [
        Sentence(**{**d, "word_idx": tuple(d["word_idx"])})
        for d in _load_json("sentences.json")["sentences"]
    ]
    windows = [
        EnergyWindow(**d) for d in _load_json("energy.json")["windows"]
    ]
    regions = [SilenceRegion(**d) for d in _load_json("silence.json")["regions"]]
    return sents, windows, regions


# ---------------------------------------------------------------------------
# Contract / JSON round-trip
# ---------------------------------------------------------------------------


def test_candidates_artifact_json_round_trip():
    cand = Candidate(
        cand_id="job-1-c0001-0003",
        start=10.0,
        end=40.0,
        sentence_range=(1, 3),
        cheap_score=0.42,
    )
    artifact = CandidatesArtifact(candidates=[cand])
    dumped = artifact.model_dump_json()
    assert json.loads(dumped) == {
        "schema_version": 1,
        "candidates": [
            {
                "cand_id": "job-1-c0001-0003",
                "start": 10.0,
                "end": 40.0,
                "sentence_range": [1, 3],
                "cheap_score": 0.42,
            }
        ],
    }
    restored = CandidatesArtifact.model_validate_json(dumped)
    assert restored == artifact
    assert restored.schema_version == 1


def test_candidates_artifact_explicit_empty():
    assert CandidatesArtifact(candidates=[
        Candidate(
            cand_id="j-c0000-0002",
            start=0.0,
            end=15.0,
            sentence_range=(0, 2),
            cheap_score=0.3,
        )
    ]).schema_version == 1


def test_candidates_artifact_empty_list():
    assert CandidatesArtifact(candidates=[]).candidates == []


def test_weights_sum_to_one():
    total = W_HOOK + W_DENSITY + W_ENERGY + W_SILENCE + W_NICHE
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Window invariants
# ---------------------------------------------------------------------------


def test_window_invariants_on_fixture():
    sents, windows, regions = _fixture_inputs()
    cands = generate_candidates(
        sents, windows, regions, job_id="job-x", niche_keywords=[]
    )
    assert cands, "expected at least one candidate from the fixture"
    seen_ids = set()
    for c in cands:
        lo, hi = c.sentence_range
        assert MIN_SENTENCES <= (hi - lo + 1) <= MAX_SENTENCES
        assert lo < hi  # inclusive, at least 2 indices span for 3..8 sentences
        assert MIN_CLIP_S <= (c.end - c.start) <= MAX_CLIP_S
        # start/end line up exactly with the sentence boundaries (inclusive).
        assert c.start == sents[lo].start
        assert c.end == sents[hi].end
        assert 0.0 <= c.cheap_score <= 1.0
        assert c.cand_id not in seen_ids
        seen_ids.add(c.cand_id)
    assert len(cands) <= TOP_K
    # ordering: non-increasing cheap_score
    scores = [c.cheap_score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_no_candidates_when_sentences_too_short():
    # 10 one-second sentences: even 8 contiguous sentences = 8s < 15s.
    sents = [_s(i, i + 1) for i in range(10)]
    assert generate_candidates(sents) == []


def test_no_candidates_when_sentences_too_long():
    # six 40s sentences: every valid window has >=3 sentences -> >=120s > 75s.
    sents = [_s(i * 40.0, (i + 1) * 40.0) for i in range(6)]
    assert generate_candidates(sents) == []


def test_empty_input_returns_empty():
    assert generate_candidates([]) == []


# ---------------------------------------------------------------------------
# Feature: hook
# ---------------------------------------------------------------------------


def test_hook_question():
    assert hook_score([_s(0, 1, "Why is this happening?")]) == 0.5


def test_hook_interrogative_lead():
    assert hook_score([_s(0, 1, "When should you ship?")]) == 0.5


def test_hook_negation():
    assert hook_score([_s(0, 1, "I do not like it.")]) == 0.25


def test_hook_digit():
    assert hook_score([_s(0, 1, "It took 5 minutes.")]) == 0.25


def test_hook_contrast():
    assert hook_score([_s(0, 1, "But it is cheap.")]) == 0.25


def test_hook_superlative():
    assert hook_score([_s(0, 1, "This is the best ever.")]) == 0.25


def test_hook_multiple_and_cap():
    # why + not + digit + '?' + but + best = 7 triggers -> capped at 1.0
    text = "Why did we not use 5? But actually the best."
    assert hook_score([_s(0, 1, text)]) == 1.0


def test_hook_uses_first_sentence_only():
    # second sentence is a strong hook, but only the first is inspected
    sents = [_s(0, 1, "Hello."), _s(1, 2, "Why did we not use 5? Best!")]
    assert hook_score(sents) == 0.0


def test_hook_word_boundary_no_false_positive():
    # "note", "number", "almost" must not trigger no / most
    assert hook_score([_s(0, 1, "I made a note of the number.")]) == 0.0


# ---------------------------------------------------------------------------
# Feature: density
# ---------------------------------------------------------------------------


def test_density_partially_filled_window():
    sents = [_s(0, 4), _s(6, 8)]  # 4s + 2s = 6s of speech
    assert density_score(sents, window_duration=10.0) == 0.6


def test_density_clamped_to_one():
    sents = [_s(0, 12)]
    assert density_score(sents, window_duration=10.0) == 1.0


def test_density_zero_when_no_speech():
    assert density_score([_s(0, 0)], window_duration=10.0) == 0.0


# ---------------------------------------------------------------------------
# Feature: energy
# ---------------------------------------------------------------------------


def test_energy_std_scaling():
    # two windows, midpoints inside [0,10], rms differing by 12 -> std 6
    windows = [
        EnergyWindow(start=0.0, end=0.5, rms_db=-40.0),
        EnergyWindow(start=4.5, end=5.0, rms_db=-28.0),
        EnergyWindow(start=8.5, end=9.0, rms_db=-40.0),
    ]
    # values = [-40, -28, -40] -> pstdev/12
    expected = statistics.pstdev([-40.0, -28.0, -40.0]) / 12.0
    assert abs(energy_score(windows, 0.0, 10.0) - expected) < 1e-9


def test_energy_fewer_than_two_windows_is_zero():
    windows = [EnergyWindow(start=0.0, end=0.5, rms_db=-30.0)]
    assert energy_score(windows, 0.0, 10.0) == 0.0


def test_energy_uses_midpoint_filter():
    # window midpoint 12.0 is outside [0,10]
    windows = [
        EnergyWindow(start=0.0, end=0.5, rms_db=-30.0),
        EnergyWindow(start=11.5, end=12.5, rms_db=10.0),
    ]
    assert energy_score(windows, 0.0, 10.0) == 0.0


# ---------------------------------------------------------------------------
# Feature: silence
# ---------------------------------------------------------------------------


def test_silence_full_penalty():
    regions = [SilenceRegion(start=0.0, end=2.5)]
    assert silence_score(regions, 0.0, 10.0) == 0.0


def test_silence_partial():
    regions = [SilenceRegion(start=0.0, end=1.0)]  # 1/10 = 0.1 fraction
    assert silence_score(regions, 0.0, 10.0) == 0.6


def test_silence_no_regions_is_full():
    assert silence_score([], 0.0, 10.0) == 1.0


# ---------------------------------------------------------------------------
# Feature: niche
# ---------------------------------------------------------------------------


def test_niche_fraction():
    assert niche_score("I ate a protein shake", ["protein", "workout"]) == 0.5
    assert niche_score("protein and workout", ["protein", "workout"]) == 1.0
    assert niche_score("nothing here", ["protein", "workout"]) == 0.0


def test_niche_neutral_when_unconfigured():
    assert niche_score("anything at all", []) == 0.5
    assert niche_score("anything at all", None) == 0.5


def test_niche_case_insensitive():
    assert niche_score("PROTEIN heavy", ["protein"]) == 1.0


# ---------------------------------------------------------------------------
# cheap_score
# ---------------------------------------------------------------------------


def test_cheap_score_bounds_and_rounding():
    assert cheap_score(1.0, 1.0, 1.0, 1.0, 1.0) == 1.0
    assert cheap_score(0.0, 0.0, 0.0, 0.0, 0.0) == 0.0
    # weights sum to 1, so a fully-featured window is exactly 1.0
    val = cheap_score(0.5, 0.5, 0.5, 0.5, 0.5)
    assert 0.0 <= val <= 1.0
    assert isinstance(val, float)
    # clamps even if features sneak outside [0,1]
    assert cheap_score(2.0, 2.0, 2.0, 2.0, 2.0) == 1.0
    # rounded to 6 decimals
    assert val == round(val, 6)


def test_cheap_score_deterministic_weighting():
    # 0.35*h + 0.2*d + 0.2*e + 0.1*s + 0.15*n
    expected = 0.35 * 0.5 + 0.2 * 0.5 + 0.2 * 0.5 + 0.1 * 0.5 + 0.15 * 0.5
    assert cheap_score(0.5, 0.5, 0.5, 0.5, 0.5) == round(expected, 6)


# ---------------------------------------------------------------------------
# IoU / NMS
# ---------------------------------------------------------------------------


def test_temporal_iou_values():
    assert temporal_iou(0, 10, 0, 10) == 1.0
    assert temporal_iou(0, 10, 10, 20) == 0.0
    # overlap 5, union 15 -> 1/3
    assert abs(temporal_iou(0, 10, 5, 15) - 1.0 / 3.0) < 1e-9


def test_nms_suppresses_high_overlap():
    ordered = [
        (0.9, 0, 2, 0.0, 10.0),
        (0.8, 3, 5, 2.0, 12.0),  # overlaps previous: IoU = 8/12 >= 0.5
        (0.7, 20, 22, 100.0, 200.0),  # disjoint -> kept
    ]
    kept = _nms(ordered, NMS_IOU)
    assert len(kept) == 2
    assert [t[1] for t in kept] == [0, 20]


def test_nms_keeps_disjoint_candidates():
    ordered = [
        (0.9, 0, 2, 0.0, 5.0),
        (0.8, 5, 7, 5.0, 10.0),
        (0.7, 8, 10, 15.0, 20.0),
    ]
    kept = _nms(ordered, NMS_IOU)
    assert len(kept) == 3


def test_nms_respects_top_k():
    # 30 disjoint windows -> capped at TOP_K (20)
    ordered = [
        (1.0 - i * 0.001, i, i + 2, float(i * 1000), float(i * 1000 + 100))
        for i in range(30)
    ]
    kept = _nms(ordered, NMS_IOU)
    assert len(kept) == TOP_K


# ---------------------------------------------------------------------------
# cand_id format / uniqueness
# ---------------------------------------------------------------------------


def test_cand_id_format_and_job_prefix():
    sents, windows, regions = _fixture_inputs()
    cands = generate_candidates(sents, windows, regions, job_id="abc123")
    pattern = re.compile(r"^abc123-c(\d{4})-(\d{4})$")
    for c in cands:
        m = pattern.match(c.cand_id)
        assert m, f"bad cand_id {c.cand_id}"
        lo, hi = c.sentence_range
        assert int(m.group(1)) == lo
        assert int(m.group(2)) == hi


def test_cand_id_global_uniqueness_across_jobs():
    sents, windows, regions = _fixture_inputs()
    a = generate_candidates(sents, windows, regions, job_id="job-A")
    b = generate_candidates(sents, windows, regions, job_id="job-B")
    all_ids = [c.cand_id for c in a] + [c.cand_id for c in b]
    assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Ordering / determinism
# ---------------------------------------------------------------------------


def test_ordering_is_deterministic():
    sents, windows, regions = _fixture_inputs()
    first = generate_candidates(sents, windows, regions, job_id="job")
    second = generate_candidates(sents, windows, regions, job_id="job")
    assert first == second
    assert [c.cand_id for c in first] == [c.cand_id for c in second]


def test_tie_break_by_position():
    # The fixture's first two candidates share a score; they must be ordered
    # by start position (14 < 22), not by score (equal).
    sents, windows, regions = _fixture_inputs()
    cands = generate_candidates(sents, windows, regions, job_id="job-m4-fixture")
    top2 = cands[:2]
    assert top2[0].cheap_score == top2[1].cheap_score
    assert top2[0].sentence_range[0] < top2[1].sentence_range[0]


# ---------------------------------------------------------------------------
# Golden (byte-identical)
# ---------------------------------------------------------------------------


def test_golden_candidates_byte_identical():
    sents, windows, regions = _fixture_inputs()
    artifact = CandidatesArtifact(
        candidates=generate_candidates(
            sents,
            windows,
            regions,
            job_id="job-m4-fixture",
            niche_keywords=[],
        )
    )
    expected = (FIXTURES / "candidates.json").read_text()
    assert artifact.model_dump_json(indent=2) + "\n" == expected


# ---------------------------------------------------------------------------
# Property test over seeded synthetic sentence lists
# ---------------------------------------------------------------------------


def _random_sentences(rng: random.Random) -> list[Sentence]:
    n = rng.randint(0, 60)
    sents = []
    t = 0.0
    for _ in range(n):
        dur = round(rng.uniform(0.3, 25.0), 2)
        text = rng.choice(
            [
                "why is this here",
                "no one knows",
                "3 things 5 ways",
                "but the best is worst",
                "hello world",
                "",
            ]
        )
        sents.append(_s(t, t + dur, text))
        t += dur + rng.uniform(0.0, 2.0)
    return sents


def test_property_invariants_seeded():
    for seed in range(8):
        rng = random.Random(seed)
        sents = _random_sentences(rng)
        energy = [
            EnergyWindow(start=i, end=i + 1.0, rms_db=float(rng.uniform(-60, -10)))
            for i in range(0, 600, 1)
        ]
        regions = [
            SilenceRegion(
                start=float(rng.uniform(0, 500)),
                end=float(rng.uniform(500, 600)),
            )
            for _ in range(rng.randint(0, 5))
        ]
        niche = rng.sample(["alpha", "beta", "gamma", "delta"], rng.randint(0, 3))
        job_id = f"prop-{seed}"

        cands = generate_candidates(
            sents, energy, regions, job_id=job_id, niche_keywords=niche
        )

        assert len(cands) <= TOP_K
        ids = set()
        prev_score = None
        for c in cands:
            lo, hi = c.sentence_range
            assert MIN_SENTENCES <= (hi - lo + 1) <= MAX_SENTENCES
            assert lo < hi
            assert MIN_CLIP_S <= (c.end - c.start) <= MAX_CLIP_S
            assert c.start == sents[lo].start
            assert c.end == sents[hi].end
            assert 0.0 <= c.cheap_score <= 1.0
            assert c.cand_id == f"{job_id}-c{lo:04d}-{hi:04d}"
            assert c.cand_id not in ids
            ids.add(c.cand_id)
            if prev_score is not None:
                assert c.cheap_score <= prev_score
            prev_score = c.cheap_score

        # pairwise IoU must be strictly below the NMS threshold
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                assert temporal_iou(a.start, a.end, b.start, b.end) < NMS_IOU
