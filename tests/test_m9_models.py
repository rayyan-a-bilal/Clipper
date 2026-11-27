"""M9a — M9 contract tests (CPU-only, no I/O beyond dataclasses).

Covers docs/spec/m9.md decisions m9-1 … m9-11 as implemented in
`src/clipper/models/m9_models.py` (M9a: contracts + assets only).
"""

import json

import pytest
from pydantic import ValidationError

from clipper.models import (
    CaptionOverrides,
    ClipPlan,
    FontEntry,
    FontInventory,
    JobOverrides,
    LayoutOverrides,
    RenderHistory,
    RenderVersion,
    ResolvedCaption,
    ResolvedLayout,
    StylePreset,
)

# --------------------------------------------------------------------------
# round-trips
# --------------------------------------------------------------------------

def test_all_m9_models_json_round_trip():
    caption = CaptionOverrides(
        enabled=True,
        hook_enabled=True,
        burn_in="full",
        font="Montserrat ExtraBold",
        size=88,
        margin_v=450,
        margin_l=150,
        margin_r=150,
        safe_top=220,
        safe_right=150,
        safe_zone_platform="tiktok",
        max_words=4,
        max_line_chars=20,
        max_lines=2,
        primary="&HFFFFFF&",
        highlight="&H00FFFF&",
        outline=4,
        shadow=0,
        bold=True,
        karaoke=True,
    )
    layout = LayoutOverrides(
        score_threshold=0.6, face_follow_enabled=True, follow_persistence_min=0.3
    )
    assert json.loads(caption.model_dump_json()) == caption.model_dump()
    assert json.loads(layout.model_dump_json()) == layout.model_dump()

    resolved_c = ResolvedCaption(
        enabled=True, hook_enabled=True, burn_in="full",
        font="Montserrat ExtraBold", size=88, margin_v=450, margin_l=150,
        margin_r=150, safe_top=220, safe_right=150,
        safe_zone_platform="tiktok", max_words=4, max_line_chars=20,
        max_lines=2, primary="&HFFFFFF&", highlight="&H00FFFF&",
        outline=4, shadow=0, bold=True, karaoke=True,
        sources={
            "size": "preset",
            "margin_v": "clip",
            "max_words": "job",
            "max_line_chars": "config",
        },
    )
    rt = ResolvedCaption.model_validate_json(resolved_c.model_dump_json())
    assert rt == resolved_c
    assert rt.sources["size"] == "preset"

    resolved_l = ResolvedLayout(
        score_threshold=0.5, face_follow_enabled=True,
        follow_persistence_min=0.3,
        sources={"score_threshold": "clip", "face_follow_enabled": "config"},
    )
    assert ResolvedLayout.model_validate_json(
        resolved_l.model_dump_json()) == resolved_l

    preset = StylePreset(id="default", label="Default", caption=caption, layout=layout)
    assert StylePreset.model_validate(preset.model_dump()) == preset
    assert preset.schema_version == 1

    job = JobOverrides(style_id="default", caption_overrides=caption,
                       layout_overrides=layout)
    assert JobOverrides.model_validate(job.model_dump()) == job
    assert job.schema_version == 1

    entry = FontEntry(file="assets/fonts/Montserrat-ExtraBold.ttf",
                      family="Montserrat ExtraBold",
                      stem="Montserrat-ExtraBold", fallback_used=False)
    assert FontEntry.model_validate(entry.model_dump()) == entry
    inventory = FontInventory(fonts=[entry, FontEntry(
        file="x.ttf", family="Inter", stem="x", fallback_used=True)])
    assert FontInventory.model_validate(inventory.model_dump()) == inventory
    assert inventory.schema_version == 1


def test_render_version_and_history_round_trip():
    version = RenderVersion(
        n=2,
        final_path="data/jobs/j/clips/c/final.v2.mp4",
        created_at="2026-12-02T12:00:00Z",
        dirty=["caption"],
        resolved_caption=CaptionOverrides(size=64),
        resolved_layout=LayoutOverrides(score_threshold=0.5),
        render_seconds=12.5,
    )
    rt = RenderVersion.model_validate(version.model_dump())
    assert rt == version
    assert rt.model_dump_json() and version.model_dump_json()

    empty = RenderVersion(n=1, final_path="f.mp4", created_at="t")
    assert empty.dirty == []
    assert empty.resolved_caption is None
    assert empty.resolved_layout is None
    assert empty.render_seconds == 0.0

    history = RenderHistory(clip_id="job-clip01", versions=[version, empty])
    rt = RenderHistory.model_validate(history.model_dump())
    assert rt == history
    assert rt.schema_version == 1
    parsed = RenderHistory.model_validate_json(
        json.dumps(RenderHistory(clip_id="c", versions=[]).model_dump())
    )
    assert parsed.versions == [] and parsed.schema_version == 1


# --------------------------------------------------------------------------
# extra="forbid"
# --------------------------------------------------------------------------

def test_caption_overrides_forbids_unknown_keys():
    with pytest.raises(ValidationError):
        CaptionOverrides(size=80, unknown_knob=1)
    with pytest.raises(ValidationError, match="extra"):
        CaptionOverrides(fonts=["x"])


def test_layout_overrides_forbids_unknown_keys():
    with pytest.raises(ValidationError):
        LayoutOverrides(score_threshold=0.5, layout="face_track")
    with pytest.raises(ValidationError):
        LayoutOverrides(enabled=True)


# --------------------------------------------------------------------------
# bounds (every bounded field rejects out-of-range)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("size", 39), ("size", 161), ("size", -1),
        ("margin_v", -1), ("margin_v", 901),
        ("margin_l", 401), ("margin_l", -1),
        ("margin_r", 401), ("margin_r", -10),
        ("safe_top", 601), ("safe_top", -0.5),
        ("safe_right", 401), ("safe_right", -1),
        ("max_words", 0), ("max_words", 9),
        ("max_line_chars", 7), ("max_line_chars", 41),
        ("max_lines", 0), ("max_lines", 4),
        ("outline", 9), ("outline", -1),
        ("shadow", 9), ("shadow", -1),
    ],
)
def test_caption_overrides_bounds_reject_out_of_range(field, bad_value):
    with pytest.raises(ValidationError):
        CaptionOverrides(**{field: bad_value})


@pytest.mark.parametrize(
    ("field", "good_value"),
    [
        ("size", 40), ("size", 160),
        ("margin_v", 0), ("margin_v", 900),
        ("margin_l", 400), ("margin_r", 0),
        ("safe_top", 600), ("safe_right", 400),
        ("max_words", 1), ("max_words", 8),
        ("max_line_chars", 8), ("max_line_chars", 40),
        ("max_lines", 1), ("max_lines", 3),
        ("outline", 0), ("outline", 8), ("shadow", 0), ("shadow", 8),
    ],
)
def test_caption_overrides_bounds_accept_boundaries(field, good_value):
    assert CaptionOverrides(**{field: good_value}).model_dump()[field] == good_value


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("score_threshold", 0.05), ("score_threshold", 0.96),
        ("follow_persistence_min", -0.01), ("follow_persistence_min", 1.01),
    ],
)
def test_layout_overrides_bounds_reject_out_of_range(field, bad_value):
    with pytest.raises(ValidationError):
        LayoutOverrides(**{field: bad_value})


@pytest.mark.parametrize(
    ("field", "good_value"),
    [
        ("score_threshold", 0.1), ("score_threshold", 0.95),
        ("follow_persistence_min", 0.0), ("follow_persistence_min", 1.0),
    ],
)
def test_layout_overrides_bounds_accept_boundaries(field, good_value):
    assert LayoutOverrides(**{field: good_value}).model_dump()[field] == good_value


# --------------------------------------------------------------------------
# ASS colour regex
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad", ["yellow", "#FFFFFF", "&HFFFFFF", "&HFFFFFFF&", "&XFFFFFF00&",
            "", "&HGGGGGGGG&"]
)
@pytest.mark.parametrize("field", ["primary", "highlight"])
def test_ass_colour_regex_rejects_malformed(field, bad):
    with pytest.raises(ValidationError):
        CaptionOverrides(**{field: bad})


@pytest.mark.parametrize("good", ["&H00FFFF&", "&HFFFFFF&", "&H80AABBCC&",
                                  "&H00FFFFFF&"])
@pytest.mark.parametrize("field", ["primary", "highlight"])
def test_ass_colour_regex_accepts_valid(field, good):
    # 6-digit hex (current repo values, e.g. &H00FFFF&) and 8-digit hex
    # (e.g. &H00FFFFFF&) ASS BGR forms — M9a-J2.
    assert CaptionOverrides(**{field: good}).model_dump()[field] == good


# --------------------------------------------------------------------------
# literals
# --------------------------------------------------------------------------

def test_burn_in_literal_rejects_unknown():
    with pytest.raises(ValidationError):
        CaptionOverrides(burn_in="half")


def test_burn_in_literals_accepted():
    for value in ("full", "captions_only", "none"):
        assert CaptionOverrides(burn_in=value).burn_in == value


def test_safe_zone_platform_literal_accepts_all_four_names():
    for name in ("tiktok", "shorts", "reels", "none"):
        assert CaptionOverrides(safe_zone_platform=name).safe_zone_platform == name


def test_safe_zone_platform_literal_rejects_unknown():
    with pytest.raises(ValidationError):
        CaptionOverrides(safe_zone_platform="youtube")


def test_resolved_sources_keys_restricted_to_layers():
    with pytest.raises(ValidationError):
        ResolvedLayout(
            score_threshold=0.6, face_follow_enabled=True,
            follow_persistence_min=0.3, sources={"score_threshold": "web"},
        )


# --------------------------------------------------------------------------
# CaptionOverrides all-None default
# --------------------------------------------------------------------------

def test_caption_overrides_all_defaults_none_except_no_defaults():
    dumped = CaptionOverrides().model_dump()
    assert set(dumped) == {
        "enabled", "hook_enabled", "burn_in", "font", "size", "margin_v",
        "margin_l", "margin_r", "safe_top", "safe_right",
        "safe_zone_platform", "max_words", "max_line_chars", "max_lines",
        "primary", "highlight", "outline", "shadow", "bold", "karaoke",
    }
    assert all(v is None for v in dumped.values())


def test_layout_overrides_all_defaults_none():
    dumped = LayoutOverrides().model_dump()
    assert set(dumped) == {"score_threshold", "face_follow_enabled",
                           "follow_persistence_min"}
    assert all(v is None for v in dumped.values())


# --------------------------------------------------------------------------
# ClipPlan — additive, backward compatible
# --------------------------------------------------------------------------

def _pre_m9_clip_plan_json() -> dict:
    """A ClipPlan document as M5C-5/M6e wrote it before M9 (no M9 fields)."""
    return {
        "clip_id": "job-clip01",
        "job_id": "job",
        "start": 14.0,
        "end": 41.0,
        "layout": "center_crop",
        "crop_path": [],
        "weighted_score": 9.5,
        "title": "t",
        "hook_overlay": "h",
        "description": "d",
        "hashtags": ["#x"],
        "status": "pending_qc",
    }


def test_clip_plan_pre_m9_json_still_parses():
    plan = ClipPlan.model_validate(_pre_m9_clip_plan_json())
    assert plan.status == "pending_qc"
    assert plan.layout == "center_crop"


def test_clip_plan_new_fields_default_correctly():
    plan = ClipPlan.model_validate(_pre_m9_clip_plan_json())
    assert plan.caption_overrides is None
    assert plan.layout_overrides is None
    assert plan.manual_start is None
    assert plan.manual_end is None
    assert plan.forced is False
    assert plan.force_reason is None
    assert plan.style_id is None


def test_clip_plan_m9_fields_reject_bad_values():
    base = _pre_m9_clip_plan_json()
    with pytest.raises(ValidationError):
        ClipPlan.model_validate({**base, "caption_overrides": {
            "size": 39}})
    with pytest.raises(ValidationError):
        ClipPlan.model_validate({**base, "layout_overrides": {
            "score_threshold": 0.01}})
    # "not_a_bool" is genuinely invalid (pydantic v2 lax mode would still
    # coerce "yes"/"true"/"off" strings to bool — use a non-coercible value)
    with pytest.raises(ValidationError):
        ClipPlan.model_validate({**base, "forced": "not_a_bool"})


def test_clip_plan_new_fields_round_trip():
    base = _pre_m9_clip_plan_json()
    base.update(
        caption_overrides=CaptionOverrides(size=64).model_dump(),
        layout_overrides=LayoutOverrides(score_threshold=0.5).model_dump(),
        manual_start=15.0,
        manual_end=40.5,
        forced=True,
        force_reason="user picked rejected candidate",
        style_id="compact",
    )
    plan = ClipPlan.model_validate(base)
    assert plan.model_dump() == base
    assert ClipPlan.model_validate_json(plan.model_dump_json()).forced is True


def test_clip_plan_field_set_preserved_and_schema_version_stays_1():
    pre_m9 = {
        "clip_id", "job_id", "start", "end", "layout", "crop_path",
        "weighted_score", "title", "hook_overlay", "description",
        "hashtags", "status",
    }
    added_m9 = {
        "caption_overrides", "layout_overrides", "manual_start",
        "manual_end", "forced", "force_reason", "style_id",
    }
    assert set(ClipPlan.model_fields) == pre_m9 | added_m9
    # existing field defaults unchanged
    assert ClipPlan.model_fields["status"].default == "pending_qc"
    assert pre_m9 <= set(ClipPlan.model_fields)
    # ClipPlan carries no schema_version of its own (the artifact does);
    # the card's "schema_version STAYS 1" is honoured by ClipPlansArtifact:
    assert "schema_version" not in ClipPlan.model_fields
    from clipper.models import ClipPlansArtifact
    assert ClipPlansArtifact.model_fields["schema_version"].default == 1
