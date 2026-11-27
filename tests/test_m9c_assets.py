"""M9c — asset loaders + job-override persistence (clipper.adapters.assets).

CPU-only: tmp dirs + the real repo assets (assets/safe_zones.json,
assets/styles/*.json) + the M9c inventory fixture. install_fonts.sh is
NEVER executed by any test (meta-assertion at the bottom). The fc-scan
missing-binary path is simulated by monkeypatching subprocess — no binary
invocation, no font download, no root.
"""

import ast
import json
import subprocess
from pathlib import Path

import pytest

from clipper.adapters import assets as assets_mod
from clipper.adapters.assets import (
    build_font_inventory,
    load_font_inventory,
    load_safe_zones,
    load_styles,
    read_job_overrides,
    validate_font,
    write_job_overrides,
)
from clipper.models import (
    CaptionOverrides,
    FontEntry,
    FontInventory,
    JobOverrides,
    LayoutOverrides,
    StylePreset,
)

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "m9c"
FIXTURE_INVENTORY = FIXTURE_DIR / "inventory.json"


# --------------------------------------------------------------------------
# load_safe_zones / load_styles — the real repo assets from M9a
# --------------------------------------------------------------------------

def test_load_safe_zones_parses_repo_asset():
    zones = load_safe_zones(ASSETS / "safe_zones.json")
    assert set(zones) == {"tiktok", "none"}
    assert zones["tiktok"] == {"margin_v": 450, "safe_top": 220, "safe_right": 150}
    assert zones["none"] == {"margin_v": 0, "safe_top": 0, "safe_right": 0}


def test_load_safe_zones_rejects_non_object_and_non_object_bundle(tmp_path):
    with pytest.raises(ValueError):
        load_safe_zones(tmp_path / "missing.json")
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('["not", "an", "object"]')
    with pytest.raises(ValueError, match="JSON object"):
        load_safe_zones(bad_file)
    bad_bundle = tmp_path / "bad_bundle.json"
    bad_bundle.write_text('{"tiktok": 42}')
    with pytest.raises(ValueError, match="bundle"):
        load_safe_zones(bad_bundle)


def test_load_styles_parses_repo_assets_keyed_by_id():
    styles = load_styles(ASSETS / "styles")
    assert set(styles) == {"default", "compact"}
    assert all(isinstance(p, StylePreset) for p in styles.values())
    assert styles["default"].label.startswith("Default")
    assert styles["compact"].caption is not None
    assert styles["compact"].caption.size == 64
    # preset -> resolver-shape: caption/layout nested models intact
    assert styles["default"].caption.font == "Montserrat ExtraBold"
    assert styles["compact"].layout.score_threshold == 0.5


def test_load_styles_rejects_bad_files(tmp_path):
    bad_json = '{"id": "x", "label": "y", "caption": {"size": 999}}'
    (tmp_path / "bad.json").write_text(bad_json)
    with pytest.raises(ValueError):
        load_styles(tmp_path)
    with pytest.raises(ValueError):
        load_styles(tmp_path / "does-not-exist")
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"id": "ok", "label": "Ok"}))
    (tmp_path / "bad.json").unlink()
    assert set(load_styles(tmp_path)) == {"ok"}

# --------------------------------------------------------------------------
# font inventory — fixture + fc-scan build (missing binary simulated)
# --------------------------------------------------------------------------

def test_load_font_inventory_parses_fixture_with_fallback_flags():
    inv = load_font_inventory(FIXTURE_INVENTORY)
    assert inv.schema_version == 1
    assert {e.file: e.fallback_used for e in inv.fonts} == {
        "Montserrat-ExtraBold.ttf": False,
        "Roboto-Regular.ttf": True,
    }
    by_family = {e.family: e for e in inv.fonts}
    assert by_family["Montserrat ExtraBold"].stem == "Montserrat-ExtraBold"
    # stem fallback entry keeps family == stem (m9-3)
    assert by_family["Roboto-Regular"].family == by_family["Roboto-Regular"].stem


def test_load_font_inventory_rejects_bad_schema(tmp_path):
    bad = tmp_path / "inv.json"
    bad.write_text('{"schema_version": 2, "fonts": []}')
    with pytest.raises(ValueError, match="schema_version"):
        load_font_inventory(bad)


def _completed(cmd, stdout):
    return subprocess.CompletedProcess(
        cmd, 0, stdout=stdout, stderr=""
    )


def _write_font_file(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"\x00")  # content irrelevant — fc-scan is monkeypatched
    return path


def test_build_font_inventory_uses_fc_scan_family(tmp_path, monkeypatch):
    _write_font_file(tmp_path, "Montserrat-ExtraBold.ttf")

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "fc-scan"
        assert "--format" in cmd and "%{family}" in cmd[cmd.index("--format") + 1]
        # real fc-scan output: comma-separated family-name + full name
        return _completed(cmd, "Montserrat, Montserrat ExtraBold\n")

    monkeypatch.setattr(assets_mod.subprocess, "run", fake_run)
    inv = build_font_inventory(tmp_path)
    assert len(inv.fonts) == 1
    (entry,) = inv.fonts
    assert entry.family == "Montserrat ExtraBold"  # libass name, NOT the stem
    assert entry.stem == "Montserrat-ExtraBold"
    assert entry.fallback_used is False
    assert entry.file == "Montserrat-ExtraBold.ttf"


def test_build_font_inventory_picks_most_specific_family(tmp_path, monkeypatch):
    _write_font_file(tmp_path, "Inter-Bold.otf")
    # alias before full name — order must not matter (M9c-J1)
    monkeypatch.setattr(
        assets_mod.subprocess,
        "run",
        lambda cmd, **kwargs: _completed(cmd, "Inter, Inter Bold\n"),
    )
    (entry,) = build_font_inventory(tmp_path).fonts
    assert entry.family == "Inter Bold"
    assert entry.fallback_used is False


def test_build_font_inventory_fc_scan_absent_uses_stem_fallback(tmp_path, monkeypatch):
    _write_font_file(tmp_path, "Roboto-Regular.ttf")

    def missing_binary(cmd, **kwargs):
        raise FileNotFoundError(f"fc-scan not found: {cmd}")

    monkeypatch.setattr(assets_mod.subprocess, "run", missing_binary)
    inv = build_font_inventory(tmp_path)
    (entry,) = inv.fonts
    assert entry.family == "Roboto-Regular"  # stem fallback (m9-3)
    assert entry.fallback_used is True  # flagged, never silent


def test_build_font_inventory_fc_scan_empty_output_falls_back(tmp_path, monkeypatch):
    _write_font_file(tmp_path, "Noto-Sans-Bold.ttf")
    monkeypatch.setattr(
        assets_mod.subprocess,
        "run",
        lambda cmd, **kwargs: _completed(cmd, ""),
    )
    (entry,) = build_font_inventory(tmp_path).fonts
    assert entry.family == "Noto-Sans-Bold"
    assert entry.fallback_used is True


def test_build_font_inventory_is_idempotent_and_skips_non_fonts(tmp_path, monkeypatch):
    _write_font_file(tmp_path, "A-Regular.ttf")
    _write_font_file(tmp_path, "B-Bold.otf")
    (tmp_path / "inventory.json").write_text("{}")
    (tmp_path / "readme.txt").write_text("not a font")

    def fake_run(cmd, **kwargs):
        path = Path(cmd[-1])
        return _completed(cmd, f"FamilyOf{path.stem}\n")

    monkeypatch.setattr(assets_mod.subprocess, "run", fake_run)
    first = build_font_inventory(tmp_path)
    second = build_font_inventory(tmp_path)
    assert first == second
    assert [e.file for e in first.fonts] == ["A-Regular.ttf", "B-Bold.otf"]
    assert {e.family for e in first.fonts} == {"FamilyOfA-Regular", "FamilyOfB-Bold"}


# --------------------------------------------------------------------------
# job overrides — m9-9 override channel (NOT a stage artifact)
# --------------------------------------------------------------------------

def test_read_job_overrides_missing_file_is_valid_empty(tmp_path):
    ovr = read_job_overrides(tmp_path)
    assert ovr == JobOverrides()
    assert ovr.schema_version == 1
    assert ovr.style_id is None
    assert ovr.caption_overrides is None
    assert ovr.layout_overrides is None


def test_write_job_overrides_creates_parent_dirs_and_round_trips(tmp_path):
    job_dir = tmp_path / "data" / "jobs" / "abc123"  # parent dirs do not exist
    ovr = JobOverrides(
        style_id="compact",
        caption_overrides=CaptionOverrides(size=64, margin_v=300, max_words=3),
        layout_overrides=LayoutOverrides(score_threshold=0.5),
    )
    write_job_overrides(job_dir, ovr)
    assert (job_dir / "job_overrides.json").is_file()
    assert read_job_overrides(job_dir) == ovr
    # no stray temp files left behind (atomic write cleaned up)
    assert [p.name for p in job_dir.iterdir()] == ["job_overrides.json"]


def test_write_then_read_preserves_nested_overrides_exactly(tmp_path):
    job_dir = tmp_path / "job"
    ovr = JobOverrides(
        style_id="default",
        caption_overrides=CaptionOverrides(
            enabled=True,
            hook_enabled=False,
            burn_in="captions_only",
            font="Montserrat ExtraBold",
            safe_zone_platform="tiktok",
            primary="&HFFFFFF&",
            highlight="&H00FFFF&",
            outline=3,
        ),
        layout_overrides=LayoutOverrides(
            score_threshold=0.55,
            face_follow_enabled=False,
            follow_persistence_min=0.35,
        ),
    )
    write_job_overrides(job_dir, ovr)
    back = read_job_overrides(job_dir)
    assert back.caption_overrides == ovr.caption_overrides
    assert back.layout_overrides == ovr.layout_overrides
    assert back.style_id == ovr.style_id
    # the on-disk file is the canonical channel form and carries schema_version
    data = json.loads((job_dir / "job_overrides.json").read_text())
    assert data["schema_version"] == 1
    assert data["caption_overrides"]["hook_enabled"] is False
    assert data["layout_overrides"]["follow_persistence_min"] == 0.35


def test_write_job_overrides_atomic_replace(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    ovr = JobOverrides(style_id="compact")

    real_replace = assets_mod.os.replace

    def boom(src, dst):
        raise OSError("disk error during replace")

    monkeypatch.setattr(assets_mod.os, "replace", boom)
    with pytest.raises(OSError, match="disk error"):
        write_job_overrides(job_dir, ovr)
    # failed replace: no target file, no temp litter (atomic semantics)
    assert not (job_dir / "job_overrides.json").exists()
    assert [p.name for p in job_dir.iterdir() if p.name != "job_overrides.json"] == []

    monkeypatch.setattr(assets_mod.os, "replace", real_replace)
    write_job_overrides(job_dir, ovr)
    assert read_job_overrides(job_dir) == ovr


def test_write_empty_overrides_round_trips(tmp_path):
    job_dir = tmp_path / "job"
    write_job_overrides(job_dir, JobOverrides())
    assert read_job_overrides(job_dir) == JobOverrides()


def test_read_job_overrides_rejects_bad_schema(tmp_path):
    (tmp_path / "job_overrides.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="schema_version"):
        read_job_overrides(tmp_path)


# --------------------------------------------------------------------------
# validate_font — m9-3 picker bound
# --------------------------------------------------------------------------

def _fixture_inventory() -> FontInventory:
    return FontInventory(
        fonts=[
            FontEntry(
                file="Montserrat-ExtraBold.ttf",
                family="Montserrat ExtraBold",
                stem="Montserrat-ExtraBold",
            ),
            FontEntry(file="Inter-Bold.ttf", family="Inter", stem="Inter-Bold"),
        ]
    )


def test_validate_font_known_family_passes():
    validate_font("Montserrat ExtraBold", _fixture_inventory())
    validate_font("Inter", _fixture_inventory())


def test_validate_font_unknown_family_lists_available():
    with pytest.raises(ValueError) as exc:
        validate_font("Comic Sans", _fixture_inventory())
    msg = str(exc.value)
    assert "unknown font family" in msg and "Comic Sans" in msg
    assert "Montserrat ExtraBold" in msg and "Inter" in msg  # concrete available list


def test_validate_font_empty_inventory_names_none():
    with pytest.raises(ValueError, match="\\(none installed\\)"):
        validate_font("Anything", FontInventory())


# --------------------------------------------------------------------------
# module discipline — assets.py is I/O only, no resolution logic
# --------------------------------------------------------------------------

def test_assets_module_has_no_resolution_logic():
    source = Path(assets_mod.__file__).read_text()
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
    # no resolution/config/caption/cuts surface at all — models only
    forbidden = {
        "clipper.overrides",
        "overrides",
        "clipper.captions",
        "clipper.config",
        "clipper.cuts",
        "clipper.selection",
    }
    assert not (imported_modules & forbidden)
    # and no pure-logic / resolver symbols in the public surface
    assert not set(assets_mod.__all__) & {
        "resolve_caption", "resolve_layout", "effective_cut", "dirty_set"
    }
    assert "StylePreset" in source  # models imported for the types it returns


# --------------------------------------------------------------------------
# meta — install_fonts.sh is a MANUAL installer, never executed by tests
# --------------------------------------------------------------------------

def test_no_test_executes_install_fonts_sh():
    """install_fonts.sh is manual-only (m9-3: no auto-download, never in tests).

    AST check: no call in the test suite passes an "install_fonts" argument
    to a shell/subprocess mechanism. String mentions (e.g. the render
    error-message constant) remain allowed.
    """
    test_dir = Path(__file__).resolve().parent
    exec_kinds = ("run", "call", "Popen", "system", "check_call", "check_output")
    offenders = []
    for path in sorted(test_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else ""
            )
            if name not in exec_kinds:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for literal in ast.walk(arg):
                    if isinstance(literal, ast.Constant):
                        value = literal.value
                        if isinstance(value, str) and "install_fonts" in value:
                            offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"tests must never execute install_fonts.sh: {offenders}"
