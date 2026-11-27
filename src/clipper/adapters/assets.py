"""M9c — M9 asset loaders + job-override persistence (ALL M9 file I/O).

Implements the I/O side of the M9 decisions in ``docs/spec/m9.md``:

- ``load_safe_zones``   (m9-11): ``assets/safe_zones.json`` -> plain dict
- ``load_styles``       (m9-9):  ``assets/styles/*.json`` -> ``{id: StylePreset}``
- ``load_font_inventory`` (m9-3): ``assets/fonts/inventory.json`` -> ``FontInventory``
- ``build_font_inventory`` (m9-3): rebuild ``FontInventory`` from the font
  files on disk using fontconfig's ``fc-scan`` — the family is the libass
  INTERNAL family name, never the filename stem.  Missing ``fc-scan`` (or an
  empty scan) falls back to the filename stem with ``fallback_used=True``
  (m9-3).  ``scripts/install_fonts.sh`` calls this and stays the ONLY font
  add path — no auto-download, ever.
- ``read_job_overrides`` / ``write_job_overrides`` (m9-9):
  ``data/jobs/<job_id>/job_overrides.json`` — an OVERRIDE CHANNEL, NOT a
  stage artifact.  Missing file on read is a valid empty ``JobOverrides``,
  not an error.  Writes are atomic (tmp file + ``os.replace``).

This module is I/O ONLY: the pure resolution logic stays in
``clipper.overrides`` (M9b) — assets.py imports ``clipper.models`` for the
types and NOTHING else from ``clipper`` (no ``clipper.overrides``, no
config, no captions, no cuts).  It never writes stage artifacts such as
``words.json`` / ``sentences.json`` / ``scores.json`` / ``selection.json`` /
``candidates.json``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from ..models import FontEntry, FontInventory, JobOverrides, StylePreset

__all__ = [
    "JOB_OVERRIDES_FILENAME",
    "FONT_FILE_EXTS",
    "load_safe_zones",
    "load_styles",
    "load_font_inventory",
    "build_font_inventory",
    "read_job_overrides",
    "write_job_overrides",
    "validate_font",
]

JOB_OVERRIDES_FILENAME = "job_overrides.json"
FONT_FILE_EXTS = (".ttf", ".otf", ".ttc")
_FSCAN_TIMEOUT_S = 60


def _read_json(path: Path | str) -> object:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc


def load_safe_zones(path: Path | str) -> dict:
    """Load the platform safe-zone bundles (m9-11).

    Returns the plain ``{platform: {margin_v, safe_top, safe_right}}`` dict —
    exactly the shape the M9b `resolve_caption(safe_zones=...)` argument
    expects.  Raises ``ValueError`` on a non-object JSON or a non-object
    bundle.
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object of platform bundles")
    for name, bundle in data.items():
        if not isinstance(bundle, dict):
            msg = f"{path}: safe-zone bundle for {name!r} must be a JSON object"
            raise ValueError(msg)
    return data


def load_styles(styles_dir: Path | str) -> dict[str, StylePreset]:
    """Load every ``assets/styles/*.json`` into a ``{id: StylePreset}`` map.

    Each file is validated by the ``StylePreset`` contract (``schema_version
    == 1``) — a malformed preset is a hard error naming the file.
    """
    styles: dict[str, StylePreset] = {}
    directory = Path(styles_dir)
    if not directory.is_dir():
        raise ValueError(f"no such styles directory: {directory}")
    for path in sorted(directory.glob("*.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        preset = StylePreset.model_validate(data)
        if preset.schema_version != 1:
            msg = f"{path}: schema_version must be 1 (got {preset.schema_version})"
            raise ValueError(msg)
        if preset.id in styles:
            raise ValueError(f"duplicate style id {preset.id!r} in {styles_dir}")
        styles[preset.id] = preset
    return styles


def load_font_inventory(path: Path | str) -> FontInventory:
    """Load ``assets/fonts/inventory.json`` into a ``FontInventory``.

    m9-3: the inventory lists REAL FONT FAMILY NAMES (libass-internal), one
    entry per font file, with ``fallback_used`` flagged when the family came
    from the filename-stem fallback instead of ``fc-scan``.
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    inventory = FontInventory.model_validate(data)
    if inventory.schema_version != 1:
        msg = f"{path}: schema_version must be 1 (got {inventory.schema_version})"
        raise ValueError(msg)
    return inventory


def _pick_family(line: str) -> str:
    """fc-scan ``%{family}`` may report several comma-separated names (CSS
    family-name + full name, e.g. ``Montserrat, Montserrat ExtraBold``).
    Return the MOST SPECIFIC (longest; last on ties) — the libass-matching
    name the picker shows (m9-3; judgment note M9c-J1)."""
    parts = [p.strip() for p in line.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    best = parts[0]
    for part in parts[1:]:
        if len(part) >= len(best):
            best = part
    return best


def _scan_family(path: Path, fc_scan_bin: str = "fc-scan") -> str | None:
    """Run ``fc-scan --format '%{family}\\n'`` on one font file.

    Returns the first non-empty family line, or ``None`` when the binary is
    unavailable, fails, or reports no family — the caller applies the m9-3
    stem fallback (flagged ``fallback_used=True``).
    """
    try:
        proc = subprocess.run(
            [fc_scan_bin, "--format", "%{family}\n", str(path)],
            capture_output=True,
            text=True,
            timeout=_FSCAN_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            return _pick_family(line)
    return None


def build_font_inventory(fonts_dir: Path | str) -> FontInventory:
    """Rebuild the `FontInventory` from the font files in ``fonts_dir`` (m9-3).

    Idempotent: the result depends only on the font files present.  Each
    ``.ttf``/``.otf``/``.ttc`` file (sorted by name; ``inventory.json`` and
    other files ignored) yields one ``FontEntry``:

    - ``family`` = ``fc-scan --format '%{family}\\n'`` first family line —
      the libass-internal name the picker and ``caption_overrides.font``
      must use (m9-3: filename stems SILENTLY fall back to another face in
      libass, which is why the inventory is family-based);
    - on missing ``fc-scan`` / failed scan: ``family = filename stem`` and
      ``fallback_used = True`` (the installer prints this LOUD).
    """
    directory = Path(fonts_dir)
    if not directory.is_dir():
        raise ValueError(f"no such fonts directory: {directory}")
    entries: list[FontEntry] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in FONT_FILE_EXTS:
            continue
        stem = path.stem
        family = _scan_family(path)
        fallback = False
        if family is None:
            family, fallback = stem, True
        entries.append(
            FontEntry(file=path.name, family=family, stem=stem, fallback_used=fallback)
        )
    return FontInventory(fonts=entries)


def read_job_overrides(job_dir: Path | str) -> JobOverrides:
    """Read ``job_overrides.json`` from a job directory (m9-9).

    A MISSING file is a valid state — an empty ``JobOverrides`` (all fields
    ``None``), NOT an error.  A present file is contract-validated
    (``schema_version == 1``).  This is an override channel, never a stage
    artifact.
    """
    path = Path(job_dir) / JOB_OVERRIDES_FILENAME
    if not path.is_file():
        return JobOverrides()
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    overrides = JobOverrides.model_validate(data)
    if overrides.schema_version != 1:
        msg = f"{path}: schema_version must be 1 (got {overrides.schema_version})"
        raise ValueError(msg)
    return overrides


def write_job_overrides(job_dir: Path | str, overrides: JobOverrides) -> None:
    """Atomically write ``job_overrides.json`` into the job directory (m9-9).

    Creates parent directories as needed; writes to a temp file in the SAME
    directory and replaces (``os.replace``), so concurrent readers never see
    a partial file.  Only THIS channel file is ever written — never the
    words/sentences/scores/selection/candidates stage artifacts.
    """
    directory = Path(job_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / JOB_OVERRIDES_FILENAME
    payload = overrides.model_dump_json(indent=2) + "\n"
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=".job_overrides.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def validate_font(family: str, inventory: FontInventory) -> None:
    """Check that ``family`` is an installed inventory family (m9-3).

    ``caption_overrides.font`` must match an inventory family; an unknown
    family raises a concrete error listing the available families so the
    user can pick one (no silent libass fallback).
    """
    available = sorted({entry.family for entry in inventory.fonts})
    if family not in available:
        listing = ", ".join(available) if available else "(none installed)"
        raise ValueError(
            f"unknown font family: {family!r}; available families: {listing}"
        )
