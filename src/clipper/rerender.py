"""M9e — single-clip rerender + authoritative preview (m9-5, m9-10, m9-12).

Service layer for the CLI commands `clipper rerender CLIP [--relayout]
[--cpu-render]` and `clipper preview CLIP --at T`, per `docs/spec/m9.md`:

- **m9-5 invalidation.** `rerender_clip` holds the EXISTING worker lock
  (`worker.worker_lock`) for its full duration — contended lock → the
  clean `WorkerError` refusal (never queues silently) — then re-runs ONLY
  the passes `overrides.dirty_set` says must re-run:

    - `caption` only (caption/style change)     → ASS → compose → thumbnail
                                                 (REUSES mezzanine.mp4 + audio.m4a)
    - `trim` (manual cut change)              → mezzanine → audio
                                                    → ASS → compose → thumbnail
    - `layout` (layout/face-knob change, or --relayout)
                                                 → layout (run_layout, THIS clip only)
                                                   → mezzanine → audio
                                                    → ASS → compose → thumbnail

  Multiple tokens take the UNION. The only stage artifact rerender may
  rewrite is THAT clip's `layout.json`, and only via the EXISTING
  `stages.layout.run_layout(clip_id=...)`. `words.json`, `sentences.json`,
  `scores.json`, `selection.json`, `candidates.json` are NEVER written, and
  other clips are never touched.
- **m9-12 "before" state.** The dirty set is `dirty_set(before, after)`
  with `before` = the LATEST `render_history.json` version's
  `plan_snapshot` (m9-12) — the recorded state at the last render. A clip
  with no recorded snapshot yet (never rerendered) falls back to the
  "first render" before-state: the current plan with every M9 override /
  cut knob cleared (M9e-J2, see `docs/spec/m9.md`).
- **m9-5 version history.** Each render writes `final.v<N>.mp4` (N
  increments), `final.mp4` becomes a COPY of the latest version (never a
  render target), `render.json` keeps pointing at the latest, and
  `render_history.json` gains one `RenderVersion` (dirty set + RESOLVED
  override snapshot + m9-12 `plan_snapshot` + timing). Retain the LAST 3
  versions; prune older ones (their `final.v<N>.mp4` files too).
- **m9-10 Tier 2.** `preview_frame` burns the REAL ASS onto ONE frame of
  the EXISTING mezzanine via the additive `RenderAdapter.preview` method
  (`adapters/render.py`, the canonical m9-10 ffmpeg command) →
  `clips/<clip_id>/preview.png` (overwritten each call, not versioned),
  T clamped into the effective cut. A missing or stale mezzanine (missing
  file, or a pending `trim`/`layout` change) is a CONCRETE error telling
  the user to `clipper rerender` first. Preview holds the worker lock only
  for the duration of the frame burn.
- **No-op.** Empty dirty set AND `final.mp4` present → clean success,
  NOTHING written. Empty dirty set without `final.mp4` → concrete error
  (the clip was never rendered with the current state).

Adapter discipline: ALL ffmpeg I/O goes through `clipper.adapters.render`
(M7 surfaces + additive `preview`); file I/O reuses `clipper.adapters.qc`
helpers. Pure logic reuses the M9b resolvers (`dirty_set` /
`resolve_caption` / `resolve_layout` / `effective_cut`), M7a
`captions.words_to_ass`, and the existing `run_layout` — NO M0–M8 stage
logic is changed, re-implemented, or "improved" here.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import clipper.adapters.qc as qc
from clipper import worker
from clipper.adapters import assets as m9_assets
from clipper.adapters.faces import get_face_detector
from clipper.adapters.render import (
    CPU_RENDER_WARNING,
    FFmpegRenderAdapter,
    get_render_adapter,
)
from clipper.adapters.vision import get_vision_adapter
from clipper.captions import words_to_ass
from clipper.config import Config
from clipper.cuts import CutError
from clipper.db import get_job
from clipper.models import (
    CaptionOverrides,
    ClipLayoutArtifact,
    ClipPlan,
    LayoutOverrides,
    ProbeResult,
    RenderArtifact,
    RenderHistory,
    RenderVersion,
    ResolvedCaption,
    WordsArtifact,
)
from clipper.overrides import (
    dirty_set,
    effective_cut,
    resolve_caption,
    resolve_layout,
)
from clipper.stages.layout import run_layout

logger = logging.getLogger("clipper.rerender")

__all__ = [
    "RENDER_HISTORY_FILENAME",
    "RerenderError",
    "preview_frame",
    "rerender_clip",
]


class RerenderError(Exception):
    """Rerender/preview failed (unknown clip/job, missing input, invalid
    cut, stale preview inputs). `str(error)` always carries the concrete
    detail + the next command to run."""


# Per-clip output file names (docs/spec/m7.md decision 4, unchanged).
MEZZANINE_FILENAME = "mezzanine.mp4"
AUDIO_FILENAME = "audio.m4a"
CAPTIONS_FILENAME = "captions.ass"
FINAL_FILENAME = "final.mp4"
THUMBNAIL_FILENAME = "thumbnail.jpg"
RENDER_JSON_FILENAME = "render.json"
RENDER_HISTORY_FILENAME = "render_history.json"
PREVIEW_FILENAME = "preview.png"

VERSION_FILE_RE = re.compile(r"^final\.v(\d+)\.mp4$")
RETAIN_VERSIONS = 3  # m9-5: retain the last 3 versions, prune older

# M9e: this module lives in `src/clipper/` (one level shallower than
# `adapters/`, whose files use `parents[3]` for the repo root), so here
# the repo root is `parents[2]` (src/clipper/rerender.py → clipper → src → root).
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Shared lookups / artifact readers (table access via the caller's conn;
# file I/O through `clipper.adapters.qc` / concrete errors naming producer)
# ---------------------------------------------------------------------------

def _clip_from_conn(conn, clip_id: str) -> tuple[str, ClipPlan]:
    """Resolve `(job_id, ClipPlan)` from the `clips` table (the plan is
    the CURRENT state — the CLI edit channel's target)."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT job_id, plan_json FROM clips WHERE clip_id = ?", (clip_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise RerenderError(
            f"unknown clip {clip_id!r}: no row in the clips table (a "
            "completed clip_plans stage or `clipper force-select` creates one)"
        )
    try:
        plan = ClipPlan.model_validate_json(row[1])
    except Exception as exc:
        raise RerenderError(f"invalid clip row for {clip_id!r}: {exc}") from exc
    return row[0], plan


def _job_dir(cfg: Config, job_id: str) -> Path:
    return cfg.resolved_data_dir / "jobs" / job_id


def _read_words(job_dir: Path) -> list:
    path = job_dir / "words.json"
    if not path.is_file():
        raise RerenderError(
            f"missing {path} — produced by the M2b transcribe stage "
            "(scripts/transcribe_job.py); run that stage first"
        )
    try:
        artifact = WordsArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RerenderError(f"invalid {path}: {exc}") from exc
    return artifact.words


def _read_probe(job_dir: Path) -> ProbeResult:
    path = job_dir / "probe.json"
    if not path.is_file():
        raise RerenderError(
            f"missing {path} — produced by the M1 ingest stage "
            "(python -m clipper ingest)"
        )
    try:
        return ProbeResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RerenderError(f"invalid {path}: {exc}") from exc


def _read_layout(clip_dir: Path, clip_id: str) -> ClipLayoutArtifact:
    path = clip_dir / "layout.json"
    if not path.is_file():
        raise RerenderError(
            f"missing {path} — produced by the M6e layout stage "
            "(scripts/layout_job.py); run that stage first (or use "
            f" `clipper rerender {clip_id} --relayout`)"
        )
    try:
        artifact = ClipLayoutArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RerenderError(f"invalid {path}: {exc}") from exc
    if artifact.clip_id != clip_id:
        msg = f"{path} belongs to clip {artifact.clip_id!r}, not {clip_id!r}"
        raise RerenderError(msg)
    return artifact


def _read_history(clip_dir: Path, clip_id: str) -> RenderHistory | None:
    """`render_history.json` if present (missing = valid: never rerendered)."""
    path = clip_dir / RENDER_HISTORY_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        history = RenderHistory.model_validate(data)
    except Exception as exc:
        raise RerenderError(f"invalid {path}: {exc}") from exc
    if history.clip_id != clip_id:
        msg = f"{path} belongs to clip {history.clip_id!r}, not {clip_id!r}"
        raise RerenderError(msg)
    return history


def _write_history(
    clip_dir: Path, clip_id: str, versions: list[RenderVersion]
) -> Path:
    history = RenderHistory(
        schema_version=1, clip_id=clip_id, versions=list(versions[:RETAIN_VERSIONS])
    )
    path = clip_dir / RENDER_HISTORY_FILENAME
    # Atomic tmp+replace via the shared adapter helper (M9d discipline).
    return qc.write_text_file(path, history.model_dump_json(indent=2) + "\n")


def _write_render_json(clip_dir: Path, artifact: RenderArtifact) -> Path:
    path = clip_dir / RENDER_JSON_FILENAME
    return qc.write_text_file(
        path, artifact.model_dump_json(indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# m9-12 — the recorded "before" side of the dirty set
# ---------------------------------------------------------------------------

# The M9 additive ClipPlan knobs a rerender is responsible for re-deriving
# (m9-1 / m9-2 / m9-8 / m9-9). `forced` / `force_reason` are metadata-only
# and never dirty anything (m9-5 / M9b `dirty_set`).
_FIRST_RENDER_CLEARED_FIELDS: dict[str, object] = {
    "caption_overrides": None,
    "layout_overrides": None,
    "manual_start": None,
    "manual_end": None,
    "style_id": None,
}


def _before_plan(current: ClipPlan, history: RenderHistory | None) -> ClipPlan:
    """The `before` plan for `dirty_set` (m9-12).

    The LATEST version's `plan_snapshot` (the authoritative state at the
    last render) when recorded; else the "first render" fallback — the
    current plan with ALL M9 override/cut knobs cleared (M9e-J2: such a
    clip's M7d render ran on exactly the non-overridden state).
    """
    if history is not None:
        for version in history.versions:  # latest first (M9a model rule)
            if version.plan_snapshot is not None:
                return version.plan_snapshot
    return current.model_copy(update=dict(_FIRST_RENDER_CLEARED_FIELDS))


# ---------------------------------------------------------------------------
# m9-9/m9-11 — the resolved override stack (snapshots + resolve-aware ASS)
# ---------------------------------------------------------------------------

def _style_preset_for(style_id: str | None):
    """The style preset for `plan.style_id` (m9-9 layer; `None` = unset).

    A SET `style_id` whose preset is not installed is a concrete error —
    never a silent default (m9-3/m9-9: no silent fallback of any kind).
    """
    if style_id is None:
        return None
    styles = m9_assets.load_styles(_REPO_ROOT / "assets" / "styles")
    if style_id not in styles:
        raise RerenderError(
            f"unknown style_id {style_id!r} on the clip — no matching preset "
            f"in assets/styles/ (available: {', '.join(sorted(styles)) or 'none'})"
        )
    return styles[style_id]


def _resolved(
    cfg: Config,
    plan: ClipPlan,
    job_dir: Path,
):
    """(ResolvedCaption, ResolvedLayout) — four layers config → preset →
    job overrides → clip (M9b `overrides.resolve_caption/resolve_layout`),
    using the M9c asset loaders + `job_overrides.json` channel."""
    try:
        safe_zones_path = _REPO_ROOT / "assets" / "safe_zones.json"
        safe_zones = m9_assets.load_safe_zones(safe_zones_path)
    except ValueError as exc:
        raise RerenderError(f"safe-zone assets unreadable: {exc}") from exc
    preset = _style_preset_for(plan.style_id)
    job_ovr = m9_assets.read_job_overrides(job_dir)  # missing = valid empty
    try:
        resolved_cap = resolve_caption(
            cfg.captions, preset, job_ovr, plan.caption_overrides, safe_zones
        )
        resolved_lov = resolve_layout(
            cfg.faces, preset, job_ovr, plan.layout_overrides
        )
    except ValueError as exc:
        # e.g. `platform preset not installed: <name>` (m9-11 / M9a-J1)
        raise RerenderError(f"unresolvable override stack for the clip: {exc}") from exc
    return resolved_cap, resolved_lov


def _ass_text(
    cfg: Config,
    plan: ClipPlan,
    words: list,
    resolved_cap,
) -> str:
    """The ASS document for THIS render (m9-5: every render path ends at a
    regenerated ASS, from the CURRENT resolved caption values — never a
    stale `captions.ass` file).

    Built on the EXISTING M7a engine (`captions.words_to_ass`) with a
    shadow `Config` carrying the RESOLVED caption values (M9e-J1 — the
    engine reads knobs from `cfg.captions`, so resolution is applied
    through that same surface; no engine changes), then the m9-11
    `burn_in` / `enabled` / `hook_enabled` switches are applied by
    dropping the affected `Dialogue:` style lines:

    - `enabled=False` or `burn_in="none"`  → no word captions
    - `burn_in in {"none","captions_only"}` or `hook_enabled=False` → no hook

    (m9-11: enabled/hook_enabled are the finer-grained switches WITHIN
    full/captions_only; `full` with both on == unchanged behaviour.)
    """
    cap = cfg.captions
    # The M7a engine applies ONE side margin (safe_zone_right → both
    # margin_l AND margin_r). Use the resolved margin when the clip
    # resolved symmetric margins; else fall back to safe_right (M9e-J1).
    side = (
        resolved_cap.margin_l
        if resolved_cap.margin_l == resolved_cap.margin_r
        else resolved_cap.safe_right
    )
    shadow_cfg = cfg.model_copy(
        update={
            "captions": cap.model_copy(
                update={
                    "font": resolved_cap.font,
                    "font_size": resolved_cap.size,
                    "margin_v": resolved_cap.margin_v,
                    "safe_zone_top": resolved_cap.safe_top,
                    "safe_zone_right": side,
                    "max_words_per_group": resolved_cap.max_words,
                    "max_chars_per_line": resolved_cap.max_line_chars,
                    "max_lines": resolved_cap.max_lines,
                }
            )
        }
    )
    base = words_to_ass(words, plan, shadow_cfg)

    drop_captions = (not resolved_cap.enabled) or (resolved_cap.burn_in == "none")
    drop_hook = (
        resolved_cap.burn_in in ("none", "captions_only")
        or not resolved_cap.hook_enabled
    )
    if not (drop_captions or drop_hook):
        return base
    lines = []
    for line in base.split("\n"):
        if line.startswith("Dialogue:"):
            parts = line.split(",")
            # Dialogue: 0,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            style = parts[4] if len(parts) > 4 else ""
            if drop_captions and style != "Hook":
                continue
            if drop_hook and style == "Hook":
                continue
        lines.append(line)
    return "\n".join(lines)


def _override_snapshot(resolved) -> CaptionOverrides | LayoutOverrides:
    """The concrete resolved values (all keys) as the override model of
    record — the M9a-J4 snapshot type on `RenderVersion."""
    if isinstance(resolved, ResolvedCaption):
        model: type = CaptionOverrides
    else:
        model = LayoutOverrides
    return model(
        **{key: getattr(resolved, key) for key in model.model_fields}
    )


# ---------------------------------------------------------------------------
# m9-5 — version numbering / history / pruning
# ---------------------------------------------------------------------------

def _next_version(clip_dir: Path, history: RenderHistory | None) -> int:
    """The next `final.v<N>` number: one above the highest existing
    version (history entries AND on-disk `final.v*.mp4` files)."""
    highest = 0
    for path in clip_dir.glob("final.v*.mp4"):
        match = VERSION_FILE_RE.fullmatch(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    if history is not None:
        for version in history.versions:
            highest = max(highest, int(version.n))
    return highest + 1


def _prune_old_versions(
    clip_dir: Path, kept_versions: list[RenderVersion]
) -> list[Path]:
    """Delete `final.v<N>.mp4` files older than the retained set (m9-5:
    retain the LAST 3, prune older — files AND history entries)."""
    keep = {int(v.n) for v in kept_versions[:RETAIN_VERSIONS]}
    removed: list[Path] = []
    for path in sorted(clip_dir.glob("final.v*.mp4")):
        match = VERSION_FILE_RE.fullmatch(path.name)
        if match and int(match.group(1)) not in keep:
            path.unlink()
            removed.append(path)
    return removed


def _render_adapter(cfg: Config, cpu_render: bool):
    """The render adapter for this rerender.

    - `cpu_render=True` (the explicit `--cpu-render` flag, M7 decision 2):
      a FRESH `FFmpegRenderAdapter(cpu_render=True)` + the loud
      CPU_RENDER_WARNING — direct construction, because the adapter
      SINGLETON may hold an earlier non-CPU instance and must never mask
      an explicit flag (and there is NO silent libx264/fallback path).
    - default: the existing `adapters.render.get_render_adapter` factory
      (production path / `FixtureRenderAdapter` under fixture cfgs).
    """
    if cpu_render:
        logger.warning(CPU_RENDER_WARNING)
        return FFmpegRenderAdapter(cpu_render=True)
    return get_render_adapter(cfg)


# ---------------------------------------------------------------------------
# m9-5 — the only stage rerender may run (layout, THIS clip only)
# ---------------------------------------------------------------------------

def _run_layout_for_clip(
    job_id: str, cfg: Config, conn, clip_id: str, resolved_lov
) -> None:
    """Re-run the EXISTING layout stage for THIS clip only (m9-5: the sole
    stage artifact rerender may rewrite is this clip's `layout.json`, and
    only via `run_layout(clip_id=...)`).

    Face knobs apply at their RESOLVED value: `run_layout` reads the
    selection knobs from `cfg.faces`, so we pass a shadow config carrying
    the `ResolvedLayout` concrete values (M9e-J3 — same shadow-cfg
    discipline as the ASS build; `run_layout` itself is unchanged)."""
    faces = cfg.faces
    shadow_cfg = cfg.model_copy(
        update={
            "faces": faces.model_copy(
                update={
                    "score_threshold": resolved_lov.score_threshold,
                    "face_follow_enabled": resolved_lov.face_follow_enabled,
                    "follow_persistence_min": resolved_lov.follow_persistence_min,
                }
            )
        }
    )
    vision = get_vision_adapter(cfg)
    detector = get_face_detector(cfg)
    try:
        run_layout(job_id, shadow_cfg, conn, vision, detector, clip_id=clip_id)
    except Exception as exc:
        raise RerenderError(
            f"layout re-run for {clip_id} failed: {exc} — "
            "(the layout stage owns its own failure bookkeeping; no "
            "partial state)"
        ) from exc


# ---------------------------------------------------------------------------
# m9-5 — rerender_clip
# ---------------------------------------------------------------------------

def rerender_clip(
    clip_id: str,
    cfg: Config,
    conn,
    relayout: bool = False,
    cpu_render: bool = False,
) -> dict:
    """Rerender ONE clip with dirty-set invalidation (m9-5, m9-12).

    - Holds the EXISTING `worker.worker_lock` for the full duration
      (contention → the clean `WorkerError` refusal; never queues).
    - `dirty = dirty_set(before, current)` (m9-12 `before`; `--relayout`
      adds the `layout` token). Empty dirty + `final.mp4` present → clean
      NO-OP success (nothing written). Empty dirty without `final.mp4` →
      concrete error (never rendered yet).
    - Else exactly the passes the matrix says (union of tokens), in order:
      [layout] → [mezzanine → audio] → ASS → compose (`final.v<N>.mp4`)
      → thumbnail (`thumbnail.jpg`) → `render.json` (latest) → copy to
      `final.mp4` → `render_history.json` (append + prune to last 3,
      pruning older `final.v<N>.mp4` too).
    - `words.json` / `sentences.json` / `scores.json` / `selection.json` /
      `candidates.json` are NEVER written; other clips are never touched;
      the clip's `ClipPlan.status` (QC state) is left as-is — the version
      history is the authoritative render-state record (M9e-J4).

    Returns a summary dict: `clip_id`, `job_id`, `no_op`, `dirty`,
    `relayout`, `version`, `final_path`, `versioned_path`,
    `thumbnail_path`, `removed_versions`, `render_seconds`.
    """
    started = time.monotonic()
    with worker.worker_lock(cfg):
        job_id, plan = _clip_from_conn(conn, clip_id)
        job = get_job(conn, job_id)
        if job is None:
            raise RerenderError(f"unknown job {job_id!r} for clip {clip_id!r}")
        job_dir = _job_dir(cfg, job_id)
        clip_dir = job_dir / "clips" / clip_id

        history = _read_history(clip_dir, clip_id)
        before = _before_plan(plan, history)
        dirty = dirty_set(before, plan)
        if relayout:
            dirty.add("layout")

        # ---- no-op success (m9-5): nothing changed, output current ------
        if not dirty:
            final = clip_dir / FINAL_FILENAME
            if not final.is_file():
                raise RerenderError(
                    f"no rerender work for {clip_id} (nothing changed since "
                    "the last render) AND no existing render to keep: the "
                    f"clips/{clip_id}/final.mp4 is missing. Complete the "
                    f"M7 render stage first (`clipper run {job_id}`) or force "
                    f"a full re-render: `clipper rerender {clip_id} --relayout`"
                )
            return {
                "clip_id": clip_id,
                "job_id": job_id,
                "no_op": True,
                "dirty": [],
                "relayout": False,
                "version": None,
                "final_path": str(final.resolve()),
                "versioned_path": None,
                "thumbnail_path": None,
                "removed_versions": [],
                "render_seconds": 0.0,
            }

        # ---- resolved override stack (snapshots + ASS + layout knobs) ---
        resolved_cap, resolved_lov = _resolved(cfg, plan, job_dir)

        # ---- effective cut (m9-2, existing validated surface) ----------
        try:
            cut_start, cut_end = effective_cut(plan)
        except CutError as exc:
            raise RerenderError(
                f"invalid cut for {clip_id}: {exc} — fix the manual trim and "
                "rerender"
            ) from exc
        # Render + ASS against the EFFECTIVE window (plan fields intact
        # otherwise — the clip's status/QC state is never rewritten here).
        render_plan = plan.model_copy(update={"start": cut_start, "end": cut_end})

        adapter = _render_adapter(cfg, cpu_render)

        # ---- layout pass (only if `layout` dirty; may rewrite this
        # clip's layout.json + the clip's layout/crop_path via the
        # existing stage) -------------------------------------------------
        layout_ran = False
        if "layout" in dirty:
            _run_layout_for_clip(job_id, cfg, conn, clip_id, resolved_lov)
            # The decision may have changed the plan (layout/crop_path).
            job_id, plan = _clip_from_conn(conn, clip_id)
            render_plan = plan.model_copy(
                update={"start": cut_start, "end": cut_end}
            )
            layout_ran = True

        # ---- video pass (trim | layout): mezzanine + audio --------------
        if "trim" in dirty or "layout" in dirty:
            source_path = job.get("source_path") or ""
            if not source_path or not Path(source_path).is_file():
                raise RerenderError(
                    f"cannot (re)render the mezzanine for {clip_id}: the "
                    f"source video {source_path!r} is missing from the "
                    "jobs table/disk"
                )
            words = _read_words(job_dir)
            probe = _read_probe(job_dir)
            layout_art = _read_layout(clip_dir, clip_id)
            adapter.render_mezzanine(
                source_path,
                render_plan,
                layout_art.layout,
                str(clip_dir / MEZZANINE_FILENAME),
                cfg,
                native_fps=probe.fps,
                tracks=layout_art.tracks,
            )
            adapter.extract_audio(
                source_path, cut_start, cut_end, str(clip_dir / AUDIO_FILENAME)
            )
        else:
            words = _read_words(job_dir)
            for filename, label in (
                (MEZZANINE_FILENAME, "mezzanine"),
                (AUDIO_FILENAME, "audio"),
            ):
                if not (clip_dir / filename).is_file():
                    raise RerenderError(
                        f"caption-only rerender for {clip_id} needs the "
                        f"existing {label} ({clip_dir / filename}), which is "
                        "missing — the clip was probably never rendered "
                        f"with this cut/layout. Full pass: `clipper rerender "
                        f"{clip_id} --relayout`"
                    )

        # ---- ASS (ALWAYS regenerated on a render, m9-5) ------------------
        ass_path = clip_dir / CAPTIONS_FILENAME
        try:
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            ass_path.write_text(
                _ass_text(cfg, render_plan, words, resolved_cap),
                encoding="utf-8",
            )
        except OSError as exc:
            raise RerenderError(f"failed to write {ass_path}: {exc}") from exc

        # ---- compose the NEXT version (final.mp4 never a render target) --
        version = _next_version(clip_dir, history)
        final_version_path = clip_dir / f"final.v{version}.mp4"
        partial = adapter.compose_final(
            str(clip_dir / MEZZANINE_FILENAME),
            str(clip_dir / AUDIO_FILENAME),
            str(ass_path),
            str(final_version_path),
            cfg,
        )

        # ---- thumbnail (from the NEW versioned final) --------------------
        thumbnail = adapter.make_thumbnail(
            str(final_version_path), str(clip_dir / THUMBNAIL_FILENAME)
        )

        # ---- render.json continues to point at the latest ---------------
        # (M7 contract unchanged: `final_path` is the clip's STABLE
        # `final.mp4`; its CONTENT is now the copy of the latest version)
        artifact = partial.model_copy(
            update={
                "clip_id": clip_id,
                "final_path": str((clip_dir / FINAL_FILENAME).resolve()),
                "thumbnail_path": str(thumbnail),
            }
        )
        try:
            _write_render_json(clip_dir, artifact)
        except OSError as exc:
            msg = f"failed to write {clip_dir / RENDER_JSON_FILENAME}: {exc}"
            raise RerenderError(msg) from exc

        # ---- final.mp4 = copy of the latest version ----------------------
        try:
            qc.copy_file(final_version_path, clip_dir / FINAL_FILENAME)
        except (OSError, ValueError) as exc:
            msg = f"failed to copy {final_version_path} to final.mp4: {exc}"
            raise RerenderError(msg) from exc

        # ---- render_history.json: append (latest first) + prune ----------
        version_rec = RenderVersion(
            n=version,
            final_path=str(final_version_path.resolve()),
            created_at=datetime.now().isoformat(),
            dirty=sorted(dirty),
            resolved_caption=_override_snapshot(resolved_cap),
            resolved_layout=_override_snapshot(resolved_lov),
            render_seconds=round(time.monotonic() - started, 3),
            plan_snapshot=plan,
        )
        kept = [version_rec, *(history.versions if history is not None else [])]
        kept = kept[:RETAIN_VERSIONS]
        try:
            _write_history(clip_dir, clip_id, kept)
            removed = _prune_old_versions(clip_dir, kept)
        except OSError as exc:
            raise RerenderError(f"failed to update the version history: {exc}") from exc

        return {
            "clip_id": clip_id,
            "job_id": job_id,
            "no_op": False,
            "dirty": sorted(dirty),
            "relayout": layout_ran,
            "version": version,
            "final_path": str((clip_dir / FINAL_FILENAME).resolve()),
            "versioned_path": str(final_version_path.resolve()),
            "thumbnail_path": str(Path(thumbnail).resolve()),
            "removed_versions": [str(p) for p in removed],
            "render_seconds": round(time.monotonic() - started, 3),
        }


# ---------------------------------------------------------------------------
# m9-10 Tier 2 — preview_frame
# ---------------------------------------------------------------------------

def _stale_reason(
    clip_id: str,
    clip_dir: Path,
    plan: ClipPlan,
    history: RenderHistory | None,
) -> str | None:
    """m9-10: "tier 2 ... if the mezzanine is stale relative to a pending
    trim/layout change, disable tier 2 and say why." Returns the concrete
    reason when stale, else `None`."""
    mezzanine = clip_dir / MEZZANINE_FILENAME
    ass = clip_dir / CAPTIONS_FILENAME
    if not mezzanine.is_file():
        return (
            f"the mezzanine is stale (clips/{clip_id}/mezzanine.mp4 is "
            "missing — the clip was never rendered here or the "
            "intermediate was removed)"
        )
    if not ass.is_file():
        return (
            "clips/"
            f"{clip_id}/captions.ass is missing — the clip was never rendered here"
        )
    # Pending trim/layout change relative to the recorded before state
    # (m9-12 snapshot) → the on-disk mezzanine does not reflect it anymore.
    before = _before_plan(plan, history)
    try:
        pending = dirty_set(before, plan)
    except Exception:
        pending = set()
    if "trim" in pending or "layout" in pending:
        return (
            f"the mezzanine is stale relative to a pending change "
            f"(dirty={sorted(pending)})"
        )
    return None


def preview_frame(
    clip_id: str,
    at_s: float,
    cfg: Config,
    conn,
) -> Path:
    """Burn the REAL ASS onto ONE frame of the EXISTING mezzanine (m9-10
    tier 2) and write `clips/<clip_id>/preview.png` (overwritten each
    call, never versioned). Returns the written path.

    - T = `at_s` (seconds from job start), clamped into the effective cut
      `[start, end]` (existing `overrides.effective_cut` validation).
    - Stale inputs (missing mezzanine/ASS, or a pending `trim`/`layout`
      change) → CONCRETE `RerenderError` telling the user to
      `clipper rerender CLIP_ID` first (tier 2 is disabled, with a reason).
    - The worker lock is held ONLY for the duration of the frame burn
      (m9-10) — the ffmpeg call goes through `RenderAdapter.preview`
      (the canonical m9-10 command in `adapters/render.py`).
    """
    with worker.worker_lock(cfg):
        job_id, plan = _clip_from_conn(conn, clip_id)
        clip_dir = _job_dir(cfg, job_id) / "clips" / clip_id
        history = _read_history(clip_dir, clip_id)

        reason = _stale_reason(clip_id, clip_dir, plan, history)
        if reason is not None:
            raise RerenderError(
                f"preview for {clip_id} unavailable: {reason} — run "
                f"`clipper rerender {clip_id}` first"
            )

        try:
            cut_start, cut_end = effective_cut(plan)
        except CutError as exc:
            raise RerenderError(
                f"preview for {clip_id}: the pending cut is invalid ({exc}) — "
                "fix the manual trim first"
            ) from exc

        # Clamp the requested timestamp INTO the effective cut [start, end].
        t = min(max(at_s - cut_start, 0.0), cut_end - cut_start)

        adapter = get_render_adapter(cfg)
        dst = clip_dir / PREVIEW_FILENAME
        try:
            out = adapter.preview(
                str(clip_dir / MEZZANINE_FILENAME),
                str(clip_dir / CAPTIONS_FILENAME),
                float(t),
                str(dst),
            )
        except RerenderError:
            raise
        except Exception as exc:
            raise RerenderError(
                f"preview frame burn for {clip_id} failed: {exc}"
            ) from exc
        return Path(out)
