"""M5C-3 — metadata contracts, prompt, validation, fallback (CPU-only).

Covers the pure module `src/clipper/metadata.py` plus the existing
contracts (`ClipMetadata`, `MetadataResponse`) and the `METADATA_SYSTEM`
prompt. No LLM calls, no stage/script wiring — that is M5C-4/5 scope.
"""

import json
import re

import pytest
from pydantic import ValidationError

from clipper.metadata import (
    DESCRIPTION_MAX,
    FALLBACK_DESCRIPTION_CHARS,
    FALLBACK_HOOK_WORDS,
    FALLBACK_TITLE_WORDS,
    HASHTAG_MAX,
    HASHTAG_MIN,
    HOOK_MAX_CHARS,
    HOOK_MAX_WORDS,
    TITLE_MAX,
    build_metadata_request,
    fallback_metadata,
    validate_metadata,
)
from clipper.models import ClipMetadata, MetadataResponse
from clipper.prompts import METADATA_SYSTEM

CANDS = ["job1-c0001-0004", "job1-c0009-0015"]


def make_item(cand_id: str = CANDS[0], **overrides) -> ClipMetadata:
    base = {
        "cand_id": cand_id,
        "title": "An honest 60-second lesson",
        "hook_overlay": "The part nobody tells you",
        "description": (
            "The coach explains the first three weeks. "
            "It is boring work, but it works."
        ),
        "hashtags": ["#fitness", "#nutrition", "#coach"],
    }
    base.update(overrides)
    return ClipMetadata(**base)


def make_response(items: list[ClipMetadata]) -> MetadataResponse:
    return MetadataResponse(items=items)


# ---------------------------------------------------------------------------
# Contracts round-trip
# ---------------------------------------------------------------------------


def test_clip_metadata_round_trip():
    item = make_item()
    raw = item.model_dump_json()
    assert ClipMetadata.model_validate_json(raw) == item
    assert ClipMetadata.model_validate(json.loads(raw)) == item


def test_metadata_response_round_trip():
    response = make_response([make_item(cand_id=c) for c in CANDS])
    raw = response.model_dump_json(indent=2)
    assert MetadataResponse.model_validate_json(raw) == response
    assert len(response.items) == 2

    plain = json.loads(raw)
    assert set(plain) == {"items"}
    assert {frozenset(row) for row in plain["items"]} == {
        frozenset({"cand_id", "title", "hook_overlay", "description", "hashtags"})
    }


def test_metadata_response_requires_items_and_shaped_items():
    with pytest.raises(ValidationError):
        MetadataResponse()  # 'items' is required
    with pytest.raises(ValidationError):
        # Missing fields on one item must fail validation
        MetadataResponse(items=[{"cand_id": "x"}])


def test_clip_metadata_field_types():
    with pytest.raises(ValidationError):
        ClipMetadata(
            cand_id="c",
            title="t",
            hook_overlay="h",
            description="d",
            hashtags="not-a-list",
        )


# ---------------------------------------------------------------------------
# validate_metadata
# ---------------------------------------------------------------------------


def test_validate_accepts_exactly_one_item_per_selected():
    items = [make_item(cand_id=c) for c in CANDS]
    out = validate_metadata(make_response(items), list(CANDS))
    assert out == items
    assert [m.cand_id for m in out] == list(CANDS)


def test_validate_zero_selected_zero_items():
    assert validate_metadata(MetadataResponse(items=[]), []) == []


@pytest.mark.parametrize(
    "items,cands",
    [
        ([make_item(cand_id=CANDS[0])], list(CANDS)),  # missing second
        ([], list(CANDS)),  # missing all
        ([
            make_item(cand_id=c) for c in CANDS
        ] + [make_item(cand_id="job1-c0020-0099")],
            list(CANDS),
        ),  # extra
        (
            [make_item(cand_id="other-a"), make_item(cand_id="other-b")],
            list(CANDS),
        ),  # wrong ids entirely
        (
            [make_item(cand_id=CANDS[0]), make_item(cand_id=CANDS[0])],
            list(CANDS),
        ),  # duplicate
    ],
)
def test_validate_rejects_id_mismatches(items, cands):
    with pytest.raises(ValueError, match="mismatch|expected|duplicate"):
        validate_metadata(make_response(items), cands)


def test_validate_rejects_empty_fields():
    for field in ("title", "hook_overlay", "description"):
        with pytest.raises(ValueError, match=field):
            validate_metadata(
                make_response([make_item(**{field: "   "})]), list(CANDS[:1])
            )


def test_validate_rejects_over_length_fields():
    cases = [
        ("title", "x" * (TITLE_MAX + 1)),
        ("title", "word " * 40 + "x" * TITLE_MAX),  # long title
        ("hook_overlay", "one two three four five six seven"),  # 7 words
        ("hook_overlay", "y" * (HOOK_MAX_CHARS + 1)),  # 1 word, 41 chars
        ("description", "z" * (DESCRIPTION_MAX + 1)),
    ]
    for field, value in cases:
        with pytest.raises(ValueError, match=field):
            validate_metadata(
                make_response([make_item(**{field: value})]), list(CANDS[:1])
            )


def test_validate_accepts_boundary_lengths():
    item = make_item(
        cand_id=CANDS[0],
        title="t" * TITLE_MAX,
        hook_overlay="w" * HOOK_MAX_CHARS,
        description="d" * DESCRIPTION_MAX,
    )
    assert validate_metadata(make_response([item]), list(CANDS[:1])) == [item]


@pytest.mark.parametrize(
    "hashtags",
    [
        ["#a"],  # too few (< 3)
        ["#a", "#b"],
        ["#a", "#b", "#c", "#d", "#e", "#f", "#g"],  # too many (> 6)
        ["#a", "#b c", "#c"],  # space inside a tag
        ["a", "#b", "#c"],  # missing '#'
        ["#", "#b", "#c"],  # bare '#'
        ["#a", "a#b", "#c"],  # tag not starting with '#'
    ],
)
def test_validate_rejects_bad_hashtags(hashtags):
    with pytest.raises(ValueError, match="hashtag"):
        validate_metadata(
            make_response([make_item(hashtags=hashtags)]), list(CANDS[:1])
        )


def test_validate_accepts_boundary_hashtag_counts():
    low = make_item(hashtags=["#a", "#b", "#c"])
    high = make_item(
        hashtags=["#a1", "#b2", "#c3", "#d4", "#e5", "#f6"]
    )
    assert validate_metadata(make_response([low]), list(CANDS[:1])) == [low]
    assert validate_metadata(make_response([high]), list(CANDS[:1])) == [high]


def test_validate_error_names_the_cand_id():
    with pytest.raises(ValueError, match=CANDS[0]):
        validate_metadata(
            make_response([make_item(cand_id=CANDS[0], title="")]),
            list(CANDS[:1]),
        )


# ---------------------------------------------------------------------------
# fallback_metadata — deterministic and within limits
# ---------------------------------------------------------------------------

TRANSCRIPT = (
    "The first three weeks are the part nobody tells you about. "
    "Protein first, then sleep, then the rest. It is not glamorous, "
    "it is just the baseline that makes everything else possible."
)


def test_fallback_values():
    item = fallback_metadata(CANDS[0], TRANSCRIPT, "fitness")
    assert item.cand_id == CANDS[0]
    words = TRANSCRIPT.split()
    assert item.title == " ".join(words[:FALLBACK_TITLE_WORDS])
    assert item.hook_overlay == " ".join(words[:FALLBACK_HOOK_WORDS])
    assert item.description == TRANSCRIPT[:FALLBACK_DESCRIPTION_CHARS]
    assert item.hashtags == ["#fitness"]


def test_fallback_niche_default():
    assert fallback_metadata("c", "a b c", "").hashtags == ["#clip"]
    assert fallback_metadata("c", "a b c", "   ").hashtags == ["#clip"]
    assert fallback_metadata("c", "a b c", "ai").hashtags == ["#ai"]


def test_fallback_deterministic():
    a = fallback_metadata(CANDS[1], TRANSCRIPT, "fitness")
    b = fallback_metadata(CANDS[1], TRANSCRIPT, "fitness")
    assert a == b
    assert a.model_dump_json() == b.model_dump_json()


def test_fallback_within_limits_normal():
    item = fallback_metadata(CANDS[0], TRANSCRIPT, "fitness")
    assert len(item.title) <= TITLE_MAX
    assert 1 <= len(item.hook_overlay.split()) <= HOOK_MAX_WORDS
    assert len(item.hook_overlay) <= HOOK_MAX_CHARS
    assert len(item.description) <= DESCRIPTION_MAX
    assert len(item.hashtags) >= 1
    assert all(t.startswith("#") and not any(c.isspace() for c in t)
               for t in item.hashtags)


def test_fallback_within_limits_hostile_inputs():
    # One enormous word and a very long transcript: still within limits
    # and still validatable.
    hostile = " ".join(["word"] * 3) + "X" * 500
    item = fallback_metadata(CANDS[0], hostile, "niche")
    assert len(item.title) <= TITLE_MAX
    assert len(item.hook_overlay) <= HOOK_MAX_CHARS
    assert len(item.description) <= DESCRIPTION_MAX
    assert item.description.startswith("word word wordX")
    assert fallback_metadata(CANDS[0], "", "fitness").title == ""
    # Empty transcript: title/hook/description degrade to empty strings
    # (degenerate clip); the niche tag still survives.
    empty = fallback_metadata(CANDS[0], "", "fitness")
    assert empty.hashtags == ["#fitness"]


# ---------------------------------------------------------------------------
# build_metadata_request — compact, whitelisted, full-transcript-free
# ---------------------------------------------------------------------------


def _clip_dict(**extra) -> dict:
    return {
        "cand_id": CANDS[0],
        "transcript": "Selected clip transcript only.",
        "niche": "fitness",
        "duration_s": 34.5,
        **extra,
    }


def test_request_whitelisted_keys_and_order():
    a, b = _clip_dict(), _clip_dict(cand_id=CANDS[1])
    out = json.loads(build_metadata_request([a, b]))
    assert out == {
        "clips": [
            {
                "cand_id": CANDS[0],
                "duration_s": 34.5,
                "niche": "fitness",
                "transcript": "Selected clip transcript only.",
            },
            {
                "cand_id": CANDS[1],
                "duration_s": 34.5,
                "niche": "fitness",
                "transcript": "Selected clip transcript only.",
            },
        ]
    }
    # Input order preserved
    assert [c["cand_id"] for c in out["clips"]] == list(CANDS)
    # Compact: no `", "` / `": "` separators outside string values
    raw = build_metadata_request([a])
    assert '", "' not in raw and '": ' not in raw


def test_request_carries_no_full_video_transcript():
    full_video = "FULL VIDEO TRANSCRIPT MARKER never meant for the metadata call."
    clip = _clip_dict(full_transcript=full_video)  # caller passes the full text
    raw = build_metadata_request([clip])
    assert "FULL VIDEO TRANSCRIPT MARKER" not in raw
    parsed = json.loads(raw)
    assert set(parsed["clips"][0]) == {"cand_id", "transcript", "niche", "duration_s"}


def test_request_zero_clips():
    raw = build_metadata_request([])
    assert json.loads(raw) == {"clips": []}
    assert raw == '{"clips":[]}'


def test_request_ignores_unknown_keys():
    raw = build_metadata_request(
        [_clip_dict(sentence_range=(3, 8), context_before="nope", secret=123)]
    )
    parsed = json.loads(raw)
    assert set(parsed["clips"][0]) <= {"cand_id", "transcript", "niche", "duration_s"}
    assert "sentence_range" not in raw and "secret" not in raw


# ---------------------------------------------------------------------------
# METADATA_SYSTEM prompt — limits match spec (docs/spec/m5.md via task card)
# ---------------------------------------------------------------------------


def test_prompt_limits_match_module_constants():
    p = METADATA_SYSTEM
    assert f"title: at most {TITLE_MAX} characters" in p  # 80
    assert (
        f"hook_overlay: at most {HOOK_MAX_WORDS} words and at most "
        f"{HOOK_MAX_CHARS} characters" in p
    )  # 6 words / 40 chars
    assert f"description: at most {DESCRIPTION_MAX} characters" in p  # 300
    assert f"exactly {HASHTAG_MIN} to {HASHTAG_MAX} hashtags" in p  # 3-6
    assert "MUST start with '#'" in p
    assert "MUST contain no spaces" in p


def test_prompt_ground_rules():
    p = METADATA_SYSTEM.lower()
    assert "no clickbait" in p
    assert re.search(r"no clickbait lies|do not invent facts", p) is not None
    assert "return json only" in p
    assert '"items"' in p
    assert '"cand_id"' in p
    # Exactly one item per clip, same order
    assert "one item per input clip" in p
    assert "same order as the input list" in p
    # Selected clips only, never the full video
    assert "full transcript" in p


def test_prompt_input_is_selected_clips_only():
    # The input to the metadata call is the selected clips' final transcript
    # (plus cand_id / times); the prompt must never demand the full video
    # transcript.
    p = METADATA_SYSTEM.lower()
    assert "you receive the selected clips" in p
    assert "full transcript" in p
    assert "one item per input clip" in p
    assert "same order as the input list" in p
