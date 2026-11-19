"""M5a — new scoring contracts: JSON round-trip (additive, no changes)."""

import json

from clipper.models import (
    Candidate,
    CandidateContext,
    ScoredClip,
    ScoresArtifact,
    ScoringBatch,
    ScoringResponse,
)


def _scored(cand_id="abc-c0001-0005", reject=None) -> ScoredClip:
    return ScoredClip(
        cand_id=cand_id,
        hook=7,
        payoff=8,
        self_contained=7,
        emotion=6,
        clarity=9,
        suggested_in=1.25,
        suggested_out=30.5,
        reject_reason=reject,
    )


def test_candidate_context_roundtrip():
    cc = CandidateContext(
        cand_id="job1-c0003-0007",
        start=3.5,
        end=31.25,
        transcript="Stop scrolling. Here is the part nobody tells you about.",
        context_before="...introducing the topic a moment earlier",
        context_after="...and then the coach wraps up the segment",
    )
    restored = CandidateContext.model_validate_json(cc.model_dump_json())
    assert restored == cc


def test_scoring_batch_roundtrip():
    batch = ScoringBatch(
        candidates=[
            CandidateContext(cand_id="a", start=1.0, end=20.0, transcript="t1"),
            CandidateContext(cand_id="b", start=21.0, end=40.0, transcript="t2"),
        ]
    )
    data = json.loads(batch.model_dump_json())
    assert data["schema_version"] == 1
    restored = ScoringBatch.model_validate_json(batch.model_dump_json())
    assert restored == batch
    assert restored.candidates[0].transcript == "t1"


def test_scoring_batch_max_length():
    import pydantic

    seven = [
        CandidateContext(
            cand_id=f"c{i}", start=float(i), end=float(i + 1), transcript="t")
        for i in range(7)
    ]
    try:
        ScoringBatch(candidates=seven)
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("ScoringBatch must cap at 5 candidates")


def test_scoring_response_roundtrip():
    resp = ScoringResponse(scores=[_scored(), _scored("job1-c0009-0012")])
    rest = ScoringResponse.model_validate_json(resp.model_dump_json())
    assert rest == resp
    assert rest.scores[1].cand_id == "job1-c0009-0012"


def test_scores_artifact_roundtrip():
    art = ScoresArtifact(model="gemma4:12b", scores=[_scored(reject="fragment")])
    assert art.schema_version == 1
    rest = ScoresArtifact.model_validate_json(art.model_dump_json())
    assert rest == art
    assert rest.scores[0].reject_reason == "fragment"


def test_existing_candidate_unchanged():
    """Existing contracts MUST remain byte-identical (M5a is additive)."""
    c = Candidate(
        cand_id="j-c0001-0004",
        start=1.0,
        end=20.0,
        sentence_range=(1, 4),
        cheap_score=0.62,
    )
    d = json.loads(c.model_dump_json())
    assert set(d) == {"cand_id", "start", "end", "sentence_range", "cheap_score"}
    assert Candidate(**d) == c
