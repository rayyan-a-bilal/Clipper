"""
M5C-5 script tests for `scripts/select_job.py` — CPU only.

Covers: `--no-llm` uses fallback metadata WITHOUT spawning managed Ollama
(exit 0); zero selected -> exit 0, reason breakdown printed, Ollama NEVER
spawned; default path where `managed_ollama` raises LLMError -> warning +
fallback metadata, exit 0 (job does not fail); `--mock` canned metadata
without spawn; missing job -> exit 1.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clipper.adapters.llm import LLMError
from clipper.config import Config
from clipper.db import Database

sys.path.insert(0, str(Path(__file__).parent))

from test_clip_plans_stage import JOB_ID, _failing_score, _setup  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent


def _load_script() -> object:
    path = REPO_ROOT / "scripts" / "select_job.py"
    name = f"clipper_select_job_script_{JOB_ID}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _select_job_dir(cfg: Config) -> Path:
    return cfg.resolved_data_dir / "jobs" / JOB_ID


def _spies(monkeypatch):
    """Patch the script's managed_ollama so SPAWN ATTEMPTS are provable."""
    script = _load_script()
    spawned: list = []

    def boom(cfg_):
        spawned.append(cfg_)
        raise LLMError("port 11435 already in use (spawn attempt spy)")

    monkeypatch.setattr(script, "managed_ollama", boom)
    return script, spawned


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


# ---------------------------------------------------------------------------
# --no-llm: fallback metadata, NO spawn, exit 0 (selected > 0)
# ---------------------------------------------------------------------------
def test_script_no_llm_uses_fallback_without_spawning(
    tmp_path, monkeypatch, capsys,
):
    cfg = _setup(tmp_path, 3)
    script, spawned = _spies(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main(JOB_ID, no_llm=True)

    assert rc == 0
    assert spawned == []  # managed Ollama NEVER entered
    out = capsys.readouterr().out
    assert "--no-llm" in out and "fallback" in out.lower()
    assert f"{JOB_ID}-clip01" in out
    assert "gate failures" not in out  # all 3 selected
    assert "rejections: none (all candidates selected)" in out

    plans = json.loads(
        (_select_job_dir(cfg) / "clip_plans.json").read_text("utf-8")
    )["plans"]
    assert len(plans) == 3
    assert all(p["status"] == "pending_qc" for p in plans)

    job = Database(cfg).get_job(JOB_ID)
    assert job["state"] == "queued"


# ---------------------------------------------------------------------------
# Zero selected: exit 0, reason breakdown printed, Ollama NEVER spawned
# ---------------------------------------------------------------------------
def test_script_zero_selected_exits_0_with_reason_breakdown(
    tmp_path, monkeypatch, capsys,
):
    cfg = _setup(tmp_path, 3, score_for=_failing_score)
    script, spawned = _spies(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main(JOB_ID)  # default path, but zero selected -> no spawn

    assert rc == 0
    assert spawned == []
    out = capsys.readouterr().out
    assert "gate failures" in out
    assert "0 clip(s)" in out
    assert "VALID success" in out
    assert "--no-llm" not in out

    data = json.loads(
        (_select_job_dir(cfg) / "clip_plans.json").read_text("utf-8")
    )
    assert data == {"schema_version": 1, "plans": []}

    job = Database(cfg).get_job(JOB_ID)
    assert job["state"] == "queued"


# ---------------------------------------------------------------------------
# Default path: managed_ollama raises LLMError -> fallback, exit 0
# ---------------------------------------------------------------------------
def test_script_managed_ollama_failure_falls_back_exit_0(
    tmp_path, monkeypatch, capsys,
):
    cfg = _setup(tmp_path, 3)
    script, spawned = _spies(monkeypatch)  # boom() records + raises LLMError
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main(JOB_ID)  # NOT --no-llm -> attempts managed Ollama

    assert rc == 0
    assert len(spawned) == 1  # exactly ONE spawn attempt
    out = capsys.readouterr().out
    assert "fallback metadata" in out
    assert "clip_plans" in out

    plans = json.loads(
        (_select_job_dir(cfg) / "clip_plans.json").read_text("utf-8")
    )["plans"]
    assert len(plans) == 3  # deterministic fallback, job succeeded

    job = Database(cfg).get_job(JOB_ID)
    assert job["state"] == "queued"


# ---------------------------------------------------------------------------
# --mock: canned metadata, NO spawn, exit 0
# ---------------------------------------------------------------------------
def test_script_mock_canned_metadata_without_spawning(
    tmp_path, monkeypatch, capsys,
):
    cfg = _setup(tmp_path, 3)
    script, spawned = _spies(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main(JOB_ID, use_mock=True)

    assert rc == 0
    assert spawned == []
    out = capsys.readouterr().out
    assert "--mock" in out and "wiring test only" in out

    plans = json.loads(
        (_select_job_dir(cfg) / "clip_plans.json").read_text("utf-8")
    )["plans"]
    assert len(plans) == 3
    # Canned metadata: fallback-shaped text + the fixed 3-tag set.
    for p in plans:
        assert p["hashtags"] == ["#clipper", "#shorts", "#demo"]
        assert p["title"].startswith("Sentence")

    job = Database(cfg).get_job(JOB_ID)
    assert job["state"] == "queued"


# ---------------------------------------------------------------------------
# Missing job -> exit 1
# ---------------------------------------------------------------------------
def test_script_missing_job_exits_1(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path, 3)
    script, spawned = _spies(monkeypatch)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    rc = script.main("no-such-job", no_llm=True)

    assert rc == 1
    assert spawned == []
    assert "not found" in capsys.readouterr().out
