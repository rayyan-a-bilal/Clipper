"""M9a — asset tests: assets/safe_zones.json + assets/styles/*.json.

CPU-only; reads the committed repo assets and the CURRENT config defaults
(`clipper.config`) + ASS style constants (`clipper.captions`) so the
"tiktok" safe zone and the "default" preset are pinned to real repo values
(Judgment notes M9a-J1 / M9a-J3, docs/spec/m9.md).
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.config import CaptionsConfig, FacesConfig
from clipper.models import FontInventory, LayoutOverrides, StylePreset

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# assets/safe_zones.json
# --------------------------------------------------------------------------

def test_safe_zones_tiktok_equals_current_cfg_values_at_runtime():
    zones = _load(ASSETS / "safe_zones.json")
    cap = CaptionsConfig()
    tiktok = zones["tiktok"]
    assert tiktok == {"margin_v": cap.margin_v,
                      "safe_top": cap.safe_zone_top,
                      "safe_right": cap.safe_zone_right}
    # the verified repo numbers (docs/spec/m9.md, Judgment note M9a-J1):
    assert tiktok == {"margin_v": 450, "safe_top": 220, "safe_right": 150}


def test_safe_zones_none_is_all_zeros():
    zones = _load(ASSETS / "safe_zones.json")
    assert zones["none"] == {"margin_v": 0, "safe_top": 0, "safe_right": 0}


def test_safe_zones_v1_ships_only_tiktok_and_none():
    zones = _load(ASSETS / "safe_zones.json")
    assert set(zones) == {"tiktok", "none"}
    for value in zones.values():
        assert set(value) == {"margin_v", "safe_top", "safe_right"}


# --------------------------------------------------------------------------
# assets/styles/default.json — provably a no-op against current config
# --------------------------------------------------------------------------

def _default_preset() -> StylePreset:
    raw = _load(ASSETS / "styles" / "default.json")
    return StylePreset.model_validate(raw)


def test_default_preset_validates_as_style_preset():
    preset = _default_preset()
    assert preset.schema_version == 1
    assert preset.id == "default"
    assert preset.caption is not None
    assert preset.layout is not None
    # applies cleanly on top of a ClipPlan (later resolver consumes it):
    assert LayoutOverrides(**preset.layout.model_dump()) is not None


def test_default_preset_caption_fields_equal_current_cfg_values():
    preset = _default_preset()
    cap = preset.caption
    assert cap is not None
    cfg = CaptionsConfig()
    # job-level cfg knobs (current behaviour, verbatim):
    assert cap.font == cfg.font
    assert cap.size == cfg.font_size
    assert cap.margin_v == cfg.margin_v
    # captions.py applies safe_zone_right to BOTH margin_l and margin_r:
    assert cap.margin_l == cfg.safe_zone_right
    assert cap.margin_r == cfg.safe_zone_right
    assert cap.safe_top == cfg.safe_zone_top
    assert cap.safe_right == cfg.safe_zone_right
    assert cap.max_words == cfg.max_words_per_group
    assert cap.max_line_chars == cfg.max_chars_per_line
    assert cap.max_lines == cfg.max_lines
    # ASS style constants (M7a/M7h — sung white, unsung yellow):
    from clipper.captions import HIGHLIGHT_COLOR, SUNG_COLOR
    assert cap.primary == SUNG_COLOR
    assert cap.highlight == HIGHLIGHT_COLOR
    assert cap.outline == 4
    assert cap.shadow == 0
    assert cap.bold is True
    # on + full burn-in = the render stage's current unconditional behaviour:
    assert cap.enabled is True
    assert cap.hook_enabled is True
    assert cap.burn_in == "full"
    # the tiktok bundle expands to the identical pixels (same-layer explicit
    # keys win, m9-11 — so the expansion cannot change anything):
    zones = _load(ASSETS / "safe_zones.json")["tiktok"]
    assert cap.margin_v == zones["margin_v"]
    assert cap.safe_top == zones["safe_top"]
    assert cap.safe_right == zones["safe_right"]


def test_default_preset_layout_fields_equal_current_cfg_values():
    preset = _default_preset()
    assert preset.layout is not None
    faces = FacesConfig()
    assert preset.layout.score_threshold == faces.score_threshold
    assert preset.layout.face_follow_enabled == faces.face_follow_enabled
    assert preset.layout.follow_persistence_min == faces.follow_persistence_min


# --------------------------------------------------------------------------
# assets/styles/ — the shipped set
# --------------------------------------------------------------------------

def test_shipped_style_presets_are_valid_and_stable():
    styles = sorted(ASSETS.glob("styles/*.json"))
    assert {p.stem for p in styles} == {"compact", "default"}
    seen_ids = set()
    for path in styles:
        preset = StylePreset.model_validate(_load(path))
        assert preset.model_dump() == StylePreset.model_validate_json(
            preset.model_dump_json()).model_dump()
        seen_ids.add(preset.id)
        assert preset.id == path.stem  # file name == id (m9-9 "presets are data")
    assert seen_ids == {"default", "compact"}


def test_compact_preset_differs_from_default():
    compact = StylePreset.model_validate(_load(ASSETS / "styles" / "compact.json"))
    default = _default_preset()
    assert compact.caption is not None and default.caption is not None
    assert compact.caption.size != default.caption.size
    assert compact.layout is not None
    assert compact.layout.score_threshold != default.layout.score_threshold


# --------------------------------------------------------------------------
# FontInventory / FontEntry shape (M9a only ships the models)
# --------------------------------------------------------------------------

def test_font_inventory_accepts_installed_font_entry():
    inventory = FontInventory.model_validate(
        {
            "schema_version": 1,
            "fonts": [
                {
                    "file": "assets/fonts/Montserrat-ExtraBold.ttf",
                    "family": "Montserrat ExtraBold",
                    "stem": "Montserrat-ExtraBold",
                    "fallback_used": False,
                }
            ],
        }
    )
    assert inventory.fonts[0].family == "Montserrat ExtraBold"


def test_font_inventory_rejects_bad_entry_shape():
    with pytest.raises(ValidationError):
        FontInventory.model_validate({"schema_version": 1,
                                      "fonts": [{"file": "f"}]})
