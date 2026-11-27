"""M9a — M9 QC-dashboard contracts (docs/spec/m9.md decisions m9-1 … m9-11).

Data contracts ONLY. No resolution logic, no I/O, no web — the pure
resolvers (`resolve_caption` / `resolve_layout`) and the asset loaders come
in a later M9 milestone. All models are plain pydantic v2, additive to the
existing M0–M8 contracts. Mirror: docs/spec/contracts.md.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# m9-1: colours stay ASS-native BGR hex, validated by regex, no hex
# conversion in v1. ASS allows both the 6-digit `&HBBGGRR&` and the
# 8-digit `&HAABBGGRR&` form; the CURRENT repo values (captions.py
# SUNG_COLOR/HIGHLIGHT_COLOR: &HFFFFFF& / &H00FFFF&, 6-digit) must stay
# valid, so BOTH forms are accepted (docs/spec/m9.md, M9a-J2).
_ASS_COLOR = re.compile(r"^&H[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?&$")


def _validated_ass_color(value: str) -> str:
    if not _ASS_COLOR.fullmatch(value):
        raise ValueError(
            f"{value!r} is not a valid ASS colour; expected &HBBGGRR& or "
            "&HAABBGGRR& (BGR hex, e.g. &H00FFFF&)"
        )
    return value


class CaptionOverrides(BaseModel):
    """Per-clip caption knobs (m9-1).

    `extra="forbid"` — any key not listed here is a hard error. Every field
    is optional; `None` means "inherit from the layer below". Defaults
    preserve current behaviour when such an object is absent
    (`ClipPlan.caption_overrides is None`).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    hook_enabled: bool | None = None
    burn_in: Literal["full", "captions_only", "none"] | None = None
    font: str | None = None
    size: int | None = None  # 40..160
    margin_v: int | None = None  # 0..900
    margin_l: int | None = None  # 0..400
    margin_r: int | None = None  # 0..400
    safe_top: int | None = None  # 0..600
    safe_right: int | None = None  # 0..400
    safe_zone_platform: Literal["tiktok", "shorts", "reels", "none"] | None = None
    max_words: int | None = None  # 1..8
    max_line_chars: int | None = None  # 8..40
    max_lines: int | None = None  # 1..3
    primary: str | None = None  # ASS BGR &HBBGGRR&/&HAABBGGRR&
    highlight: str | None = None  # ASS BGR, same form
    outline: int | None = None  # 0..8
    shadow: int | None = None  # 0..8
    bold: bool | None = None
    karaoke: bool | None = None

    @field_validator("size")
    @classmethod
    def _bounds_size(cls, v: int | None) -> int | None:
        if v is not None and not (40 <= v <= 160):
            raise ValueError("size must be in 40..160")
        return v

    @field_validator("margin_v")
    @classmethod
    def _bounds_margin_v(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 900):
            raise ValueError("margin_v must be in 0..900")
        return v

    @field_validator("margin_l", "margin_r")
    @classmethod
    def _bounds_margin_lr(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 400):
            raise ValueError("margin_l/margin_r must be in 0..400")
        return v

    @field_validator("safe_top")
    @classmethod
    def _bounds_safe_top(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 600):
            raise ValueError("safe_top must be in 0..600")
        return v

    @field_validator("safe_right")
    @classmethod
    def _bounds_safe_right(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 400):
            raise ValueError("safe_right must be in 0..400")
        return v

    @field_validator("max_words")
    @classmethod
    def _bounds_max_words(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 8):
            raise ValueError("max_words must be in 1..8")
        return v

    @field_validator("max_line_chars")
    @classmethod
    def _bounds_max_line_chars(cls, v: int | None) -> int | None:
        if v is not None and not (8 <= v <= 40):
            raise ValueError("max_line_chars must be in 8..40")
        return v

    @field_validator("max_lines")
    @classmethod
    def _bounds_max_lines(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 3):
            raise ValueError("max_lines must be in 1..3")
        return v

    @field_validator("outline", "shadow")
    @classmethod
    def _bounds_outline_shadow(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 8):
            raise ValueError("outline/shadow must be in 0..8")
        return v

    @field_validator("primary", "highlight")
    @classmethod
    def _bounds_colors(cls, v: str | None) -> str | None:
        if v is not None:
            _validated_ass_color(v)
        return v


class LayoutOverrides(BaseModel):
    """Per-clip face/layout knobs (m9-8).

    `extra="forbid"`. Layout SELECTION is NOT in this model — the UI sets
    the existing `ClipPlan.layout` and marks `layout_source="manual"`
    (forced_layout precedent, m9-8).
    """

    model_config = ConfigDict(extra="forbid")

    score_threshold: float | None = None  # 0.1..0.95
    face_follow_enabled: bool | None = None
    follow_persistence_min: float | None = None  # 0.0..1.0

    @field_validator("score_threshold")
    @classmethod
    def _bounds_score_threshold(cls, v: float | None) -> float | None:
        if v is not None and not (0.1 <= v <= 0.95):
            raise ValueError("score_threshold must be in 0.1..0.95")
        return v

    @field_validator("follow_persistence_min")
    @classmethod
    def _bounds_follow_persistence_min(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("follow_persistence_min must be in 0.0..1.0")
        return v


class ResolvedCaption(BaseModel):
    """Result of `resolve_caption` (m9-9): the CONCRETE per-clip values
    after the four-layer precedence (config -> preset -> job -> clip),
    plus `sources` mapping each field name to the layer that won."""

    enabled: bool
    hook_enabled: bool
    burn_in: Literal["full", "captions_only", "none"]
    font: str
    size: int
    margin_v: int
    margin_l: int
    margin_r: int
    safe_top: int
    safe_right: int
    safe_zone_platform: Literal["tiktok", "shorts", "reels", "none"]
    max_words: int
    max_line_chars: int
    max_lines: int
    primary: str
    highlight: str
    outline: int
    shadow: int
    bold: bool
    karaoke: bool
    sources: dict[str, Literal["config", "preset", "job", "clip"]]


class ResolvedLayout(BaseModel):
    """Result of `resolve_layout` (m9-9): concrete per-clip face/layout
    values after the four-layer precedence + winning layer per field."""

    score_threshold: float
    face_follow_enabled: bool
    follow_persistence_min: float
    sources: dict[str, Literal["config", "preset", "job", "clip"]]


class StylePreset(BaseModel):
    """Named style preset (m9-9), stored as `assets/styles/*.json`."""

    schema_version: int = 1
    id: str
    label: str
    caption: CaptionOverrides | None = None
    layout: LayoutOverrides | None = None


class JobOverrides(BaseModel):
    """Job-level override layer (m9-9), stored as
    `data/jobs/<job_id>/job_overrides.json`. An override channel, NOT a
    stage artifact."""

    schema_version: int = 1
    style_id: str | None = None
    caption_overrides: CaptionOverrides | None = None
    layout_overrides: LayoutOverrides | None = None


class FontEntry(BaseModel):
    """One installed font family (m9-3). `family` = the libass/internal
    family name; `stem` = filename stem; `fallback_used` = True when the
    family came from the filename-stem fallback instead of `fc-scan`."""

    file: str
    family: str
    stem: str
    fallback_used: bool = False


class FontInventory(BaseModel):
    """installed font families (m9-3), `assets/fonts/inventory.json`."""

    schema_version: int = 1
    fonts: list[FontEntry] = []


class RenderVersion(BaseModel):
    """One rerendered version (m9-5), recorded in `render_history.json`.
    `resolved_caption` / `resolved_layout` are the RESOLVED snapshot used
    to render version `n` (the card types them as the override models of
    record, per m9-5 "each version stores its resolved override snapshot
    + dirty set")."""

    n: int
    final_path: str
    created_at: str
    dirty: list[str] = []
    resolved_caption: CaptionOverrides | None = None
    resolved_layout: LayoutOverrides | None = None
    render_seconds: float = 0.0


class RenderHistory(BaseModel):
    """Rerender version history for one clip (m9-5),
    `clips/<clip_id>/render_history.json`. Latest versions first; the m9-5
    retain-last-3 pruning is a later-milestone behaviour rule."""

    schema_version: int = 1
    clip_id: str
    versions: list[RenderVersion] = []
