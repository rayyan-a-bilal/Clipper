"""
M7d script tests for `scripts/render_job.py` — CPU only.

Covers: success printout (per clip: clip_id, final path, duration,
width/height, encoder, render_seconds) + exit 0; ``--clip`` limits to ONE
clip; ``--cpu-render`` prints the LOUD warning AND forwards the explicit
flag; missing job -> clean error + exit 1.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from clipper.adapters.render import FixtureRenderAdapter
from clipper.config import Config

sys.path.insert(0, str(Path(__file__).parent))

from test_render_stage import JOB_ID, _setup  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent


def _load_script() -> object:
    path = REPO_ROOT / "scripts" / "render_job.py"
    name = f"clipper_render_job_script_{JOB_ID}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


def _script(monkeypatch):
    """Load the script + patch its adapter factory to the CPU fixture."""
    script = _load_script()
    calls: list[bool | None] = []

    def fake_get_render_adapter(cfg: Config | None = None,
                                cpu_render: bool | None = None):
        calls.append(cpu_render)
        return FixtureRenderAdapter()

    monkeypatch.setattr(script, "get_render_adapter", fake_get_render_adapter)
    return script, calls


def test_script_success_prints_per_clip(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    script, calls = _script(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main([JOB_ID])

    assert rc == 0
    assert calls == [None]  # default = GPU path (no explicit cpu_render)
    out = capsys.readouterr().out
    cid = f"{JOB_ID}-clip01"
    assert cid in out
    assert f"clips/{cid}/final.mp4" in out
    assert "duration=15.0s" in out
    assert "1080x1920" in out
    assert "encoder=h264_nvenc" in out
    assert "render_seconds=0.0" in out
    assert "exit 0" in out


def test_script_clip_flags_forwarded(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    script, calls = _script(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main([JOB_ID, "--cpu-render"])

    assert rc == 0
    assert calls == [True]  # explicit cpu render forwarded
    out = capsys.readouterr().out
    assert "WARNING" in out and "cpu_render" in out  # LOUD warning printed

    # and the --clip flag selects exactly one clip
    rc2 = script.main([JOB_ID, "--clip", f"{JOB_ID}-clip01"])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert f"{JOB_ID}-clip01" in out2
    # second invocation: stage already done -> skipped
    assert "skipped" in out2


def test_script_missing_job_exit_1(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    script, _ = _script(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main(["no-such-job"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and "no-such-job" in out
