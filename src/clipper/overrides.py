"""M9b — pure override resolution, effective cut, dirty set (M9, no I/O).

Implements the M9 pure-resolution surface from ``docs/spec/m9.md``:

- ``resolve_caption`` / ``resolve_layout`` (m9-9): four-layer precedence
  ``config -> preset -> job -> clip`` plus provenance (``sources``:
  field -> winning layer) for EVERY resolved key.  A ``None`` field in
  any layer means "inherit" — never "set to null".
- (m9-11) within each layer, a ``safe_zone_platform`` value set AT THAT
  LAYER first expands its ``{margin_v, safe_top, safe_right}`` bundle;
  that layer's explicit pixel keys then win over the expansion.  A
  platform whose bundle is absent from the passed ``safe_zones`` dict is
  a hard error: ``"platform preset not installed: <name>"``.
- ``effective_cut`` (m9-2): ``manual_start``/``manual_end`` when set,
  else the plan's own ``start``/``end``; validated against the EXISTING
  cuts validation surface (``CutError``, ``MIN_CLIP_S``, ``MAX_CLIP_S``
  from ``clipper.cuts``) — duration MUST stay 15–75 s, NO clamping.
- ``dirty_set`` (m9-5): rerender invalidation tokens ``"caption"`` /
  ``"trim"`` / ``"layout"`` (``forced``/``force_reason`` alone are
  metadata-only and produce an EMPTY set).

PURE module: takes already-parsed objects (config models, ``StylePreset``,
``JobOverrides``, ``ClipPlan``, plain ``safe_zones`` dict).  Never paths,
never files, never the network — the asset loaders (``assets/safe_zones.json``,
``assets/styles/*.json``) are a later M9 concern.

Judgment notes (no-drift rule; recorded here and in docs/spec/m9.md):
- M9b-J1: the m9-2 digest's ``cuts.validate`` does not exist as a
  standalone function — the m5.md "validate LAST" step lives inline at
  the end of ``cuts.refine_cut`` (start < end, 15.0 <= dur <= 75.0,
  >= 1 full sentence) raising ``cuts.CutError``.  ``effective_cut``
  therefore validates through that same real surface (``CutError`` +
  the same constants + the same "too short/too long" message shape
  naming the ACTUAL duration and the violated bound).  The full-sentence
  sub-check needs the words/sentences artifacts and belongs to
  ``refine_cut`` only; a manual trim cannot create the condition the
  original refine_cut already excluded, and the render stage re-
  validates the effective cut with the artifacts it has.
- M9b-J2: the CONFIG layer is never expanded through a platform bundle —
  its pixels come straight from ``CaptionsConfig`` (m9-11: "Platform
  presets never change cfg defaults"; margin_l/margin_r both map to
  ``safe_zone_right`` per M7a/M9a-J1).  ``safe_zone_platform`` at the
  config layer is the provenance label ``"tiktok"`` whose digitised
  bundle reproduces those defaults exactly (M9a-J1).  Expansion happens
  only for layers that explicitly set a platform, and those MUST find
  their bundle in the passed ``safe_zones`` dict (else the concrete
  "platform preset not installed: <name>" error — M9a-J1 behaviour).
- M9b-J3: style-level knobs with no ``CaptionsConfig`` field today
  (bold / outline / shadow / karaoke, sung+unsung colours, and the
  card-mandated enabled / hook_enabled / burn_in defaults) resolve to the
  CURRENT behaviour: colours are imported from ``clipper.captions``
  (``SUNG_COLOR`` / ``HIGHLIGHT_COLOR``, single source of truth),
  Bold 1 / Outline 4 / Shadow 0 mirror ``captions._style_line`` (M7a),
  and enabled=True / hook_enabled=True / burn_in="full" come from the
  task card + m9-11 (full = current behaviour).
"""

from __future__ import annotations

from typing import Any

from .captions import HIGHLIGHT_COLOR, SUNG_COLOR
from .config import CaptionsConfig, FacesConfig
from .cuts import MAX_CLIP_S, MIN_CLIP_S, CutError
from .models import (
    CaptionOverrides,
    ClipPlan,
    JobOverrides,
    LayoutOverrides,
    ResolvedCaption,
    ResolvedLayout,
    StylePreset,
)

__all__ = [
    "effective_cut",
    "resolve_caption",
    "resolve_layout",
    "dirty_set",
]

# m9-11 / card defaults when unset everywhere (see M9b-J3): full == the
# current always-captions-always-hook behaviour.
DEFAULT_ENABLED = True
DEFAULT_HOOK_ENABLED = True
DEFAULT_BURN_IN = "full"
# M9b-J2: provenance label for the config layer's default pixels (the
# tiktok bundle reproduces CaptionsConfig defaults exactly, M9a-J1).
DEFAULT_SAFE_ZONE_PLATFORM = "tiktok"
# M9b-J3: Caption-style behaviour constants from captions._style_line
# (Bold 1, Outline 4, Shadow 0) + karaoke always on (M7a/M7h).
CFG_BOLD = True
CFG_OUTLINE = 4
CFG_SHADOW = 0
CFG_KARAOKE = True

# m9-11: the keys a platform bundle expands (bundle = {margin_v, safe_top,
# safe_right}); margin_l/margin_r are NOT part of any bundle.
_ZONE_KEYS = ("margin_v", "safe_top", "safe_right")

_CAPTION_FIELDS = tuple(CaptionOverrides.model_fields)
_LAYOUT_FIELDS = tuple(LayoutOverrides.model_fields)


# ---------------------------------------------------------------------------
# m9-9 — caption / layout resolution with provenance
# ---------------------------------------------------------------------------

def _config_caption_layer(
    cfg: CaptionsConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The config-default layer: concrete values + all-sources-"config"."""
    values: dict[str, Any] = {
        "enabled": DEFAULT_ENABLED,
        "hook_enabled": DEFAULT_HOOK_ENABLED,
        "burn_in": DEFAULT_BURN_IN,
        "font": cfg.font,
        "size": cfg.font_size,
        "margin_v": cfg.margin_v,
        # M7a / M9a-J1: captions.py applies safe_zone_right as BOTH
        # margin_l and margin_r (1080x1080-safe 150/150 margins).
        "margin_l": cfg.safe_zone_right,
        "margin_r": cfg.safe_zone_right,
        "safe_top": cfg.safe_zone_top,
        "safe_right": cfg.safe_zone_right,
        "safe_zone_platform": DEFAULT_SAFE_ZONE_PLATFORM,
        "max_words": cfg.max_words_per_group,
        "max_line_chars": cfg.max_chars_per_line,
        "max_lines": cfg.max_lines,
        "primary": SUNG_COLOR,
        "highlight": HIGHLIGHT_COLOR,
        "bold": CFG_BOLD,
        "outline": CFG_OUTLINE,
        "shadow": CFG_SHADOW,
        "karaoke": CFG_KARAOKE,
    }
    return values, dict.fromkeys(values, "config")


def _apply_caption_layer(
    layer: str,
    ovr: CaptionOverrides,
    values: dict[str, Any],
    sources: dict[str, str],
    safe_zones: dict,
) -> None:
    """Fold one override layer into the running resolution (m9-9 + m9-11).

    Within the layer: (1) if ``safe_zone_platform`` is set here, expand
    its bundle FIRST; (2) then apply this layer's explicit keys on top —
    so an explicit pixel key in the same layer beats the expansion.
    ``None`` fields are skipped (inherit from the layer below).
    """
    platform = ovr.safe_zone_platform
    if platform is not None:
        bundle = safe_zones.get(platform)
        if not isinstance(bundle, dict):
            raise ValueError(f"platform preset not installed: {platform}")
        for key in _ZONE_KEYS:
            if key in bundle:
                values[key] = bundle[key]
                sources[key] = layer
    for key in _CAPTION_FIELDS:
        value = getattr(ovr, key)
        if value is not None:
            values[key] = value
            sources[key] = layer


def resolve_caption(
    cfg_captions: CaptionsConfig,
    preset: StylePreset | None,
    job: JobOverrides | None,
    clip: CaptionOverrides | None,
    safe_zones: dict,
) -> ResolvedCaption:
    """Resolve the per-clip caption knobs (m9-9, m9-11).

    Precedence, lowest to highest: ``cfg_captions`` -> ``preset.caption``
    -> ``job.caption_overrides`` -> ``clip``.  A ``None`` field means
    "inherit", never "set to null".  Returns CONCRETE values for every
    key plus ``sources`` (field -> ``"config"|"preset"|"job"|"clip"``).
    A platform set at any layer whose bundle is missing from
    ``safe_zones`` raises ``ValueError`` ("platform preset not
    installed: <name>").
    """
    values, sources = _config_caption_layer(cfg_captions)
    layers = (
        ("preset", preset.caption if preset is not None else None),
        ("job", job.caption_overrides if job is not None else None),
        ("clip", clip),
    )
    for name, ovr in layers:
        if ovr is not None:
            _apply_caption_layer(name, ovr, values, sources, safe_zones)
    return ResolvedCaption(**values, sources=sources)


def resolve_layout(
    cfg_faces: FacesConfig,
    preset: StylePreset | None,
    job: JobOverrides | None,
    clip: LayoutOverrides | None,
) -> ResolvedLayout:
    """Resolve the per-clip face/layout knobs (m9-8, m9-9).

    Same four-layer precedence, no platform expansion.  Keys:
    ``score_threshold``, ``face_follow_enabled``,
    ``follow_persistence_min``; returns concrete values + ``sources``.
    (Layout SELECTION is not resolved here — m9-8: the UI sets
    ``ClipPlan.layout`` + ``layout_source="manual"``.)
    """
    values: dict[str, Any] = {
        "score_threshold": cfg_faces.score_threshold,
        "face_follow_enabled": cfg_faces.face_follow_enabled,
        "follow_persistence_min": cfg_faces.follow_persistence_min,
    }
    sources = dict.fromkeys(values, "config")
    layers = (
        ("preset", preset.layout if preset is not None else None),
        ("job", job.layout_overrides if job is not None else None),
        ("clip", clip),
    )
    for name, ovr in layers:
        if ovr is not None:
            for key in _LAYOUT_FIELDS:
                value = getattr(ovr, key)
                if value is not None:
                    values[key] = value
                    sources[key] = name
    return ResolvedLayout(**values, sources=sources)


# ---------------------------------------------------------------------------
# m9-2 — effective cut (manual trims, validated, no clamping)
# ---------------------------------------------------------------------------

def effective_cut(plan: ClipPlan) -> tuple[float, float]:
    """The cut that will actually be rendered (m9-2).

    ``manual_start`` / ``manual_end`` when set, else the plan's own
    ``start`` / ``end`` (each independently).  Validated through the
    existing cuts validation surface (M9b-J1): ``start < end`` and
    ``MIN_CLIP_S <= end - start <= MAX_CLIP_S`` — otherwise :class:`
    CutError` with a concrete message naming the ACTUAL duration and the
    violated bound.  NO clamping, ever.
    """
    start = plan.manual_start if plan.manual_start is not None else plan.start
    end = plan.manual_end if plan.manual_end is not None else plan.end
    if not start < end:
        raise CutError(f"inverted cut: start ({start}) >= end ({end})")
    duration = end - start
    if duration < MIN_CLIP_S:
        raise CutError(f"too short: {duration:.3f}s < {MIN_CLIP_S:.1f}s")
    if duration > MAX_CLIP_S:
        raise CutError(f"too long: {duration:.3f}s > {MAX_CLIP_S:.1f}s")
    return (start, end)


# ---------------------------------------------------------------------------
# m9-5 — rerender dirty set (minimal-recompute invalidation tokens)
# ---------------------------------------------------------------------------

def dirty_set(before: ClipPlan, after: ClipPlan) -> set[str]:
    """Which render passes must re-run (m9-5).

    - ``"caption"`` -> ``caption_overrides`` or ``style_id`` changed
    - ``"trim"``   -> ``manual_start`` or ``manual_end`` changed
      (=> mezzanine -> audio -> ASS -> compose -> thumbnail)
    - ``"layout"`` -> ``layout_overrides`` or ``layout`` changed
      (=> layout -> mezzanine -> ASS -> compose -> thumbnail)

    ``forced`` / ``force_reason`` changes alone are metadata-only: EMPTY
    set.  No change -> empty set.
    """
    dirty: set[str] = set()
    if (
        before.caption_overrides != after.caption_overrides
        or before.style_id != after.style_id
    ):
        dirty.add("caption")
    if (
        before.manual_start != after.manual_start
        or before.manual_end != after.manual_end
    ):
        dirty.add("trim")
    if (
        before.layout_overrides != after.layout_overrides
        or before.layout != after.layout
    ):
        dirty.add("layout")
    return dirty
