"""M9b — CPU-only tests for the pure override-resolution surface.

Covers docs/spec/m9.md m9-9 (precedence + provenance), m9-2 (effective
cut), m9-5 (dirty set) and m9-11 (platform expansion) via
``src/clipper/overrides.py``.  All fixtures are inline — no files, no
GPU, no network.
"""

import ast
import inspect

import pytest

from clipper.config import CaptionsConfig, FacesConfig
from clipper.cuts import CutError
from clipper.models import (
    CaptionOverrides,
    ClipPlan,
    JobOverrides,
    LayoutOverrides,
    ResolvedCaption,
    ResolvedLayout,
    StylePreset,
)
from clipper.overrides import (
    dirty_set,
    effective_cut,
    resolve_caption,
    resolve_layout,
)

CFG = CaptionsConfig()
FACES = FacesConfig()
# Inline stand-in for assets/safe_zones.json (resolver takes the dict as a
# pure argument — no loader in M9b): tiktok + none per M9a, plus a test-
#only "shorts" bundle.
SAFE_ZONES = {
    "tiktok": {"margin_v": 450, "safe_top": 220, "safe_right": 150},
    "none": {"margin_v": 0, "safe_top": 0, "safe_right": 0},
    "shorts": {"margin_v": 300, "safe_top": 100, "safe_right": 80},
}

# (field, preset_value, job_value) — one case per CaptionOverrides field,
# values chosen so the winning layer is unambiguous.
MATRIX = [
    ("enabled", True, False),
    ("hook_enabled", True, False),
    ("burn_in", "full", "none"),
    ("font", "Montserrat ExtraBold", "JobFont"),
    ("size", 88, 96),
    ("margin_v", 450, 500),
    ("margin_l", 150, 120),
    ("margin_r", 150, 120),
    ("safe_top", 220, 240),
    ("safe_right", 150, 160),
    ("max_words", 4, 5),
    ("max_line_chars", 20, 24),
    ("max_lines", 2, 3),
    ("primary", "&HFFFFFF&", "&H0000FF&"),
    ("highlight", "&H00FFFF&", "&HFF0000&"),
    ("outline", 4, 2),
    ("shadow", 0, 1),
    ("bold", True, False),
    ("karaoke", True, False),
]

LAYOUT_MATRIX = [
    ("score_threshold", 0.6, 0.55),
    ("face_follow_enabled", True, False),
    ("follow_persistence_min", 0.3, 0.45),
]


def _preset(caption=None, layout=None) -> StylePreset:
    return StylePreset(id="preset-x", label="Preset X",
                       caption=caption, layout=layout)


def _job(caption=None, layout=None) -> JobOverrides:
    return JobOverrides(style_id="preset-x",
                        caption_overrides=caption, layout_overrides=layout)


def _plan(**kw) -> ClipPlan:
    base = {
        "clip_id": "job1-clip01",
        "job_id": "job1",
        "start": 10.0,
        "end": 40.0,
        "layout": "center_crop",
        "crop_path": [],
        "weighted_score": 8.0,
        "title": "t",
        "hook_overlay": "h",
        "description": "d",
        "hashtags": ["#test"],
        "status": "pending_qc",
    }
    base.update(kw)
    return ClipPlan(**base)


def _resolve(preset=None, job=None, clip=None, safe_zones=SAFE_ZONES):
    return resolve_caption(CFG, preset, job, clip, safe_zones)


def _resolve_l(preset=None, job=None, clip=None):
    return resolve_layout(FACES, preset, job, clip)


# ---------------------------------------------------------------------------
# config-layer baseline (defaults when unset everywhere, per card / m9-11)
# ---------------------------------------------------------------------------

def test_config_baseline_values_and_all_sources_config():
    res = _resolve()
    assert res.enabled is True
    assert res.hook_enabled is True
    assert res.burn_in == "full"
    assert res.font == CFG.font
    assert res.size == CFG.font_size
    assert res.margin_v == CFG.margin_v
    assert res.margin_l == CFG.safe_zone_right  # M7a: both margins from it
    assert res.margin_r == CFG.safe_zone_right
    assert res.safe_top == CFG.safe_zone_top
    assert res.safe_right == CFG.safe_zone_right
    assert res.max_words == CFG.max_words_per_group
    assert res.max_line_chars == CFG.max_chars_per_line
    assert res.max_lines == CFG.max_lines
    assert res.primary == "&HFFFFFF&"
    assert res.highlight == "&H00FFFF&"
    assert res.bold is True
    assert res.outline == 4
    assert res.shadow == 0
    assert res.karaoke is True
    assert res.safe_zone_platform == "tiktok"
    assert res.sources and all(v == "config" for v in res.sources.values())


def test_sources_cover_every_resolved_key():
    res = _resolve()
    expected = set(ResolvedCaption.model_fields) - {"sources"}
    assert set(res.sources) == expected


def test_resolved_caption_is_typed_model():
    res = _resolve(clip=CaptionOverrides(karaoke=False))
    assert isinstance(res, ResolvedCaption)
    assert isinstance(res.model_validate(res.model_dump()), ResolvedCaption)
    assert res.karaoke is False
    assert res.sources["karaoke"] == "clip"


def test_resolve_layout_config_baseline_sources():
    res = _resolve_l()
    assert res.model_validate(res.model_dump()) is not None
    assert res.score_threshold == FACES.score_threshold
    assert res.face_follow_enabled is FACES.face_follow_enabled
    assert res.follow_persistence_min == FACES.follow_persistence_min
    assert set(res.sources) == set(ResolvedLayout.model_fields) - {"sources"}
    assert all(v == "config" for v in res.sources.values())


# ---------------------------------------------------------------------------
# m9-9 precedence matrix — each layer wins over the lower ones, per key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,pv,jv", MATRIX)
def test_preset_layer_wins_over_config(field, pv, jv):
    res = _resolve(preset=_preset(caption=CaptionOverrides(**{field: pv})))
    assert getattr(res, field) == pv
    assert res.sources[field] == "preset"


@pytest.mark.parametrize("field,pv,jv", MATRIX)
def test_job_layer_wins_over_preset_and_config(field, pv, jv):
    preset = _preset(caption=CaptionOverrides(**{field: pv}))
    job = _job(caption=CaptionOverrides(**{field: jv}))
    res = _resolve(preset=preset, job=job)
    assert getattr(res, field) == jv
    assert res.sources[field] == "job"


@pytest.mark.parametrize("field,pv,jv", MATRIX)
def test_clip_layer_wins_over_job_preset_config(field, pv, jv):
    # clip re-uses the preset value — the winning layer is proven by both
    # the value AND the explicit provenance label.
    preset = _preset(caption=CaptionOverrides(**{field: pv}))
    job = _job(caption=CaptionOverrides(**{field: jv}))
    clip = CaptionOverrides(**{field: pv})
    res = _resolve(preset=preset, job=job, clip=clip)
    assert getattr(res, field) == pv
    assert res.sources[field] == "clip"


def test_preset_wins_over_config_for_other_fields_not_in_matrix():
    # safe_zone_platform itself also rides the same precedence chain
    preset = _preset(caption=CaptionOverrides(safe_zone_platform="shorts"))
    res = _resolve(preset=preset)
    assert res.safe_zone_platform == "shorts"
    assert res.sources["safe_zone_platform"] == "preset"
    # its bundle wins over the config-layer pixels
    assert res.margin_v == 300 and res.safe_top == 100 and res.safe_right == 80
    assert res.sources["margin_v"] == "preset"


# ---------------------------------------------------------------------------
# m9-9 "None means inherit" (reset-to-inherited at the clip layer)
# ---------------------------------------------------------------------------

def test_none_at_clip_layer_inherits_from_job_and_preset():
    preset = _preset(caption=CaptionOverrides(size=72, margin_v=80))
    job = _job(caption=CaptionOverrides(size=80))
    clip = CaptionOverrides(size=None, karaoke=False)  # explicit None
    res = _resolve(preset=preset, job=job, clip=clip)
    assert res.size == 80 and res.sources["size"] == "job"
    assert res.margin_v == 80 and res.sources["margin_v"] == "preset"
    assert res.karaoke is False and res.sources["karaoke"] == "clip"


def test_absent_layers_are_simple_inheritance():
    job = _job(caption=CaptionOverrides(max_words=6))
    res = _resolve(job=job, clip=CaptionOverrides())
    assert res.max_words == 6 and res.sources["max_words"] == "job"
    assert res.size == CFG.font_size and res.sources["size"] == "config"


@pytest.mark.parametrize("field,pv,jv", LAYOUT_MATRIX)
def test_layout_none_at_clip_inherits_job(field, pv, jv):
    preset = _preset(layout=LayoutOverrides(**{field: pv}))
    job = _job(layout=LayoutOverrides(**{field: jv}))
    res = _resolve_l(preset=preset, job=job, clip=LayoutOverrides())
    assert getattr(res, field) == jv
    assert res.sources[field] == "job"


@pytest.mark.parametrize("field,pv,jv", LAYOUT_MATRIX)
def test_layout_clip_wins_over_job_preset_config(field, pv, jv):
    preset = _preset(layout=LayoutOverrides(**{field: pv}))
    job = _job(layout=LayoutOverrides(**{field: jv}))
    if field == "score_threshold":
        clip_value = 0.5
    elif field == "face_follow_enabled":
        clip_value = not jv
    else:
        clip_value = 0.8
    clip_ovr = LayoutOverrides(**{field: clip_value})
    res = _resolve_l(preset=preset, job=job, clip=clip_ovr)
    assert getattr(res, field) == clip_value
    assert res.sources[field] == "clip"


# ---------------------------------------------------------------------------
# m9-11 platform expansion (within-layer rule + layer beating)
# ---------------------------------------------------------------------------

def test_same_layer_explicit_pixel_key_beats_platform_expansion():
    clip = CaptionOverrides(safe_zone_platform="tiktok", margin_v=600)
    res = _resolve(clip=clip)
    assert res.margin_v == 600            # explicit pixel key wins…
    assert res.sources["margin_v"] == "clip"
    assert res.safe_top == 220            # …bundle still fills the rest…
    assert res.safe_right == 150
    assert res.sources["safe_top"] == "clip"
    assert res.sources["safe_right"] == "clip"


def test_clip_platform_expansion_overrides_preset_pixels():
    preset = _preset(caption=CaptionOverrides(
        margin_v=350, safe_top=250, safe_right=90))
    clip = CaptionOverrides(safe_zone_platform="shorts")
    res = _resolve(preset=preset, clip=clip)
    # clip-layer bundle beats preset-layer explicit pixels for ALL bundle
    # keys, even where preset had its own explicit value
    assert res.margin_v == 300
    assert res.safe_top == 100
    assert res.safe_right == 80
    assert res.sources["margin_v"] == "clip"
    assert res.sources["safe_top"] == "clip"
    assert res.sources["safe_right"] == "clip"


def test_job_platform_beats_preset_explicit_and_clip_inherits_rest():
    preset = _preset(caption=CaptionOverrides(
        safe_zone_platform="tiktok", margin_v=100))
    job = _job(caption=CaptionOverrides(safe_zone_platform="shorts"))
    res = _resolve(preset=preset, job=job)
    # job layer: bundle first (300), then its (absent) explicit keys
    assert res.margin_v == 300 and res.sources["margin_v"] == "job"
    assert res.safe_top == 100 and res.sources["safe_top"] == "job"
    assert res.safe_right == 80 and res.sources["safe_right"] == "job"
    # preset's same-layer explicit margin_v was overtaken by job; the
    # preset's OTHER keys would still win where no higher layer set them
    assert res.margin_l == CFG.safe_zone_right  # untouched → config


def test_platform_none_expands_zero_bundle():
    clip = CaptionOverrides(safe_zone_platform="none", safe_top=120)
    res = _resolve(clip=clip)
    assert res.margin_v == 0
    assert res.safe_right == 0
    assert res.safe_top == 120  # explicit key above the "none" zeros
    assert res.safe_zone_platform == "none"


def test_platform_not_in_safe_zones_is_concrete_error():
    with pytest.raises(ValueError, match=r"platform preset not installed: reels"):
        _resolve(clip=CaptionOverrides(safe_zone_platform="reels"))
    with pytest.raises(ValueError, match=r"platform preset not installed: reels"):
        _resolve(preset=_preset(
            caption=CaptionOverrides(safe_zone_platform="reels")))
    with pytest.raises(ValueError, match=r"platform preset not installed: reels"):
        _resolve(job=_job(
            caption=CaptionOverrides(safe_zone_platform="reels")))


def test_platforms_are_checked_per_layer_that_sets_them():
    # unset platform at any layer never triggers a safe_zones lookup
    res = _resolve(
        preset=_preset(caption=CaptionOverrides(size=64)),
        safe_zones={},
    )
    assert res.size == 64
    assert res.margin_v == CFG.margin_v  # config pixels, no lookup needed


# ---------------------------------------------------------------------------
# provenance: sources reports the correct winning layer for every key
# ---------------------------------------------------------------------------

def test_full_provenance_sweep():
    preset = _preset(caption=CaptionOverrides(
        font="PF", size=60, margin_v=111, safe_top=77, safe_right=55,
        outline=1))
    job = _job(caption=CaptionOverrides(
        burn_in="captions_only", enabled=False, hook_enabled=False,
        margin_l=22, max_words=3))
    clip = CaptionOverrides(karaoke=False)
    res = _resolve(preset=preset, job=job, clip=clip)
    assert res.sources == {
        "enabled": "job",
        "hook_enabled": "job",
        "burn_in": "job",
        "font": "preset",
        "size": "preset",
        "margin_v": "preset",
        "margin_l": "job",
        "margin_r": "config",
        "safe_top": "preset",
        "safe_right": "preset",
        "safe_zone_platform": "config",
        "max_words": "job",
        "max_line_chars": "config",
        "max_lines": "config",
        "primary": "config",
        "highlight": "config",
        "outline": "preset",
        "shadow": "config",
        "bold": "config",
        "karaoke": "clip",
    }
    assert res.enabled is False
    assert res.margin_r == CFG.safe_zone_right


# ---------------------------------------------------------------------------
# m9-2 — effective cut
# ---------------------------------------------------------------------------

def test_effective_cut_no_manual_uses_original():
    assert effective_cut(_plan()) == (10.0, 40.0)


def test_effective_cut_manual_when_set():
    assert effective_cut(_plan(manual_start=12.0, manual_end=50.0)) == (12.0, 50.0)


def test_effective_cut_partial_manual_independently():
    assert effective_cut(_plan(manual_start=12.0)) == (12.0, 40.0)
    assert effective_cut(_plan(manual_end=30.0)) == (10.0, 30.0)


def test_effective_cut_manual_can_widen_beyond_original_end():
    assert effective_cut(_plan(manual_start=20.0, manual_end=50.0)) == (20.0, 50.0)


def test_effective_cut_rejects_short_with_actual_duration():
    err = pytest.raises(CutError, effective_cut,
                        _plan(manual_start=20.0, manual_end=30.0))
    assert "too short: 10.000s < 15" in str(err.value)


def test_effective_cut_rejects_long_with_actual_duration():
    err = pytest.raises(CutError, effective_cut,
                        _plan(manual_start=5.0, manual_end=88.0))
    assert "too long: 83.000s > 75" in str(err.value)


def test_effective_cut_rejects_inverted():
    with pytest.raises(CutError, match="inverted cut"):
        effective_cut(_plan(manual_start=45.0))


def test_effective_cut_uses_existing_cuts_validation_surface():
    from clipper import cuts
    err = pytest.raises(CutError, effective_cut,
                        _plan(manual_start=20.0, manual_end=30.0))
    assert "15" in str(err.value)  # bound names the real constant value
    assert f"{cuts.MIN_CLIP_S:.1f}s" in str(err.value)
    assert f"{cuts.MAX_CLIP_S:.1f}s" in str(
        pytest.raises(CutError, effective_cut,
                      _plan(manual_start=0.0, manual_end=100.0)).value)


# ---------------------------------------------------------------------------
# m9-5 — dirty set truth table
# ---------------------------------------------------------------------------

def test_dirty_caption_only():
    a = _plan()
    b = _plan(caption_overrides=CaptionOverrides(size=64))
    assert dirty_set(a, b) == {"caption"}


def test_dirty_style_id_only_counts_as_caption():
    assert dirty_set(_plan(), _plan(style_id="default")) == {"caption"}


def test_dirty_trim_only():
    assert dirty_set(_plan(), _plan(manual_start=15.0)) == {"trim"}
    assert dirty_set(_plan(), _plan(manual_end=35.0)) == {"trim"}


def test_dirty_layout_overrides_only():
    assert dirty_set(_plan(), _plan(
        layout_overrides=LayoutOverrides(score_threshold=0.5))) == {"layout"}


def test_dirty_layout_selection_only():
    assert dirty_set(_plan(), _plan(layout="face_follow")) == {"layout"}


def test_dirty_combinations():
    a = _plan()
    b = _plan(
        caption_overrides=CaptionOverrides(size=64),
        manual_start=15.0,
        layout_overrides=LayoutOverrides(face_follow_enabled=False),
    )
    assert dirty_set(a, b) == {"caption", "trim", "layout"}


def test_dirty_forced_only_is_empty_metadata():
    assert dirty_set(_plan(), _plan(forced=True)) == set()
    assert dirty_set(_plan(forced=False), _plan(forced=True,
                                                force_reason="user pick")) == set()


def test_dirty_no_change_is_empty():
    assert dirty_set(_plan(), _plan()) == set()
    same = CaptionOverrides(size=64)
    assert dirty_set(_plan(caption_overrides=same),
                     _plan(caption_overrides=CaptionOverrides(size=64))) == set()


# ---------------------------------------------------------------------------
# module hygiene — pure, no filesystem/network imports
# ---------------------------------------------------------------------------

def test_overrides_module_imports_no_filesystem_or_network_modules():
    import clipper.overrides as O

    forbidden = {
        "os", "sys", "pathlib", "io", "fcntl", "shutil", "subprocess",
        "socket", "http", "httpx", "urllib", "requests", "sqlite3",
        "wave", "threading", "multiprocessing", "asyncio", "ftplib",
    }
    tree = ast.parse(inspect.getsource(O))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names if a.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden), imported & forbidden
    # and it never opens files itself
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "open" for n in ast.walk(tree))
