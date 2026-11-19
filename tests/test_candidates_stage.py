"""
M4b candidates stage tests — CPU only, temp DB, synthetic artifacts.

Covers: artifact + stage row + candidate rows, skip on second run,
byte-identical determinism across fresh runs, clean failures (missing
inputs with milestone hints, schema_version mismatch, missing job),
zero-candidate success, and retry after failure.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import clipper.stages.candidates as cands_mod
from clipper.config import ClipsConfig, Config, PathsConfig
from clipper.db import Database
from clipper.stages import StageError
from clipper.stages.candidates import run_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
M4_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "m4"
JOB_ID = "job-m4b"
ZERO_JOB_ID = "job-m4b-zero"

ZERO_SENTENCES = [
    {"start": 0.0, "end": 2.0, "text": "Hi there.", "word_idx": [0, 0]},
    {"start": 3.0, "end": 4.0, "text": "A short one.", "word_idx": [1, 1]},
    {"start": 5.0, "end": 7.0, "text": "And another.", "word_idx": [2, 2]},
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_CLIPS_NICHE_KEYWORDS", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, niche_keywords: dict | None = None) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    clips = ClipsConfig(niche_keywords=niche_keywords or {})
    return Config(paths=PathsConfig(data_dir=str(data_dir)), clips=clips)


def _write_fixture_inputs(job_dir: Path) -> None:
    """Copy the deterministic M4a fixture inputs into the job dir."""
    for name in ("sentences.json", "energy.json", "silence.json"):
        shutil.copy(M4_FIXTURES / name, job_dir / name)


def _setup(tmp_path: Path, niche: str = "saas") -> tuple[Database, Config, Path]:
    cfg = _cfg(tmp_path)
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", niche)
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_fixture_inputs(job_dir)
    return db, cfg, job_dir


def _stages(db, job_id) -> dict:
    return {s["name"]: s for s in db.get_all_stages_for_job(job_id)}


def _candidate_rows(db, job_id) -> list[dict]:
    conn = db.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM candidates WHERE job_id = ? ORDER BY cand_id",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_candidates_creates_artifact_stage_and_rows(tmp_path):
    db, cfg, job_dir = _setup(tmp_path)
    conn = db.get_connection()
    try:
        summary = run_candidates(JOB_ID, cfg, conn)
    finally:
        conn.close()

    # Summary
    assert summary["job_id"] == JOB_ID
    assert summary["skipped"] is False
    assert 0 < summary["candidates"] <= 20

    # Artifact
    artifact_path = job_dir / "candidates.json"
    assert artifact_path.exists()
    data = json.loads(artifact_path.read_text())
    assert data["schema_version"] == 1
    cands = data["candidates"]
    assert len(cands) == summary["candidates"]
    assert len(cands) == 20  # M4a fixture golden count
    for c in cands:
        assert c["cand_id"].startswith(f"{JOB_ID}-c")
        assert c["start"] < c["end"]
        assert 0.0 <= c["cheap_score"] <= 1.0
        assert 0 <= c["sentence_range"][0] < c["sentence_range"][1] < 30

    # Stage row
    stages = _stages(db, JOB_ID)
    assert stages["candidates"]["state"] == "done"
    assert stages["candidates"]["artifact_path"] == str(artifact_path)
    assert stages["candidates"]["started_at"] is not None
    assert stages["candidates"]["finished_at"] is not None
    assert stages["candidates"]["error"] is None

    # Candidates table: one row per candidate, score_json still NULL (M5)
    rows = _candidate_rows(db, JOB_ID)
    assert len(rows) == len(cands)
    for row, cand in zip(rows, sorted(cands, key=lambda c: c["cand_id"]), strict=True):
        assert row["cand_id"] == cand["cand_id"]
        assert row["job_id"] == JOB_ID
        parsed = json.loads(row["candidate_json"])
        assert parsed == cand
        assert row["score_json"] is None

    # Job returns to queued
    assert db.get_job(JOB_ID)["state"] == "queued"


def test_second_run_skips_completed_stage(tmp_path):
    db, cfg, job_dir = _setup(tmp_path)

    conn = db.get_connection()
    run_candidates(JOB_ID, cfg, conn)
    conn.close()

    before = (job_dir / "candidates.json").read_bytes()
    stages_before = _stages(db, JOB_ID)

    conn = db.get_connection()
    summary = run_candidates(JOB_ID, cfg, conn)
    conn.close()

    assert summary["skipped"] is True
    assert summary["candidates"] == 20
    assert (job_dir / "candidates.json").read_bytes() == before
    assert _stages(db, JOB_ID) == stages_before
    assert db.get_job(JOB_ID)["state"] == "queued"


def test_deterministic_byte_identical_across_fresh_runs(tmp_path, tmp_path_factory):
    db_a, cfg_a, job_dir_a = _setup(tmp_path)

    b_root = tmp_path_factory.mktemp("fresh_b")
    cfg_b = _cfg(b_root)
    db_b = Database(cfg_b)
    assert db_b.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")
    job_dir_b = cfg_b.resolved_data_dir / "jobs" / JOB_ID
    job_dir_b.mkdir(parents=True, exist_ok=True)
    _write_fixture_inputs(job_dir_b)

    for db, cfg, _job_dir in ((db_a, cfg_a, job_dir_a), (db_b, cfg_b, job_dir_b)):
        conn = db.get_connection()
        try:
            run_candidates(JOB_ID, cfg, conn)
        finally:
            conn.close()

    a_bytes = (job_dir_a / "candidates.json").read_bytes()
    b_bytes = (job_dir_b / "candidates.json").read_bytes()
    assert a_bytes == b_bytes
    # And the DB rows are identical too
    assert _candidate_rows(db_a, JOB_ID) == _candidate_rows(db_b, JOB_ID)


def test_niche_keywords_resolved_from_config(tmp_path, tmp_path_factory):
    """cfg.clips.niche_keywords[job.niche] must drive the niche feature."""
    b_root = tmp_path_factory.mktemp("niche")
    cfg_a = _cfg(tmp_path)  # empty mapping -> neutral 0.5 niche score
    db_a = Database(cfg_a)
    assert db_a.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")
    job_dir_a = cfg_a.resolved_data_dir / "jobs" / JOB_ID
    job_dir_a.mkdir(parents=True, exist_ok=True)
    _write_fixture_inputs(job_dir_a)

    cfg_b = _cfg(
        b_root, niche_keywords={"saas": ["onboarding", "churn"]}
    )
    db_b = Database(cfg_b)
    assert db_b.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")
    job_dir_b = cfg_b.resolved_data_dir / "jobs" / JOB_ID
    job_dir_b.mkdir(parents=True, exist_ok=True)
    _write_fixture_inputs(job_dir_b)

    for db, cfg in ((db_a, cfg_a), (db_b, cfg_b)):
        conn = db.get_connection()
        try:
            run_candidates(JOB_ID, cfg, conn)
        finally:
            conn.close()

    a = json.loads((job_dir_a / "candidates.json").read_text())
    b = json.loads((job_dir_b / "candidates.json").read_text())
    assert a != b  # keyword presence changes niche scores -> different pre-rank


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing,hint",
    [
        ("sentences.json", "M2b transcribe stage (scripts/transcribe_job.py)"),
        ("energy.json", "M3b analysis stage (scripts/analyze_job.py)"),
        ("silence.json", "M3b analysis stage (scripts/analyze_job.py)"),
    ],
    ids=["sentences", "energy", "silence"],
)
def test_missing_input_fails_cleanly_with_milestone_hint(
    tmp_path, missing, hint
):
    db, cfg, job_dir = _setup(tmp_path)
    (job_dir / missing).unlink()

    conn = db.get_connection()
    try:
        with pytest.raises(StageError) as excinfo:
            run_candidates(JOB_ID, cfg, conn)
    finally:
        conn.close()

    message = str(excinfo.value)
    assert missing in message
    assert hint in message
    assert db.get_job(JOB_ID)["state"] == "failed"
    assert _stages(db, JOB_ID)["candidates"]["state"] == "failed"
    assert missing in _stages(db, JOB_ID)["candidates"]["error"]
    assert not (job_dir / "candidates.json").exists()
    assert _candidate_rows(db, JOB_ID) == []


def test_schema_version_mismatch_fails_cleanly(tmp_path):
    db, cfg, job_dir = _setup(tmp_path)
    data = json.loads((job_dir / "sentences.json").read_text())
    data["schema_version"] = 2
    (job_dir / "sentences.json").write_text(json.dumps(data))

    conn = db.get_connection()
    try:
        with pytest.raises(StageError, match="schema_version"):
            run_candidates(JOB_ID, cfg, conn)
    finally:
        conn.close()

    assert db.get_job(JOB_ID)["state"] == "failed"
    assert _stages(db, JOB_ID)["candidates"]["state"] == "failed"
    assert not (job_dir / "candidates.json").exists()


def test_missing_job_fails_cleanly(tmp_path):
    db, cfg, job_dir = _setup(tmp_path)

    conn = db.get_connection()
    try:
        with pytest.raises(StageError, match="not found"):
            run_candidates("no-such-job", cfg, conn)
    finally:
        conn.close()

    assert db.get_all_stages_for_job("no-such-job") == []
    assert db.get_job(JOB_ID)["state"] == "queued"


def test_zero_candidates_succeeds_with_empty_artifact(tmp_path):
    db, cfg, job_dir = _setup(tmp_path)

    (job_dir / "sentences.json").write_text(
        json.dumps({"schema_version": 1, "sentences": ZERO_SENTENCES})
    )
    (job_dir / "energy.json").write_text(
        json.dumps({"schema_version": 1, "window_s": 0.5, "windows": []})
    )
    (job_dir / "silence.json").write_text(
        json.dumps(
            {"schema_version": 1, "noise_db": -35.0, "min_dur": 0.5, "regions": []}
        )
    )

    conn = db.get_connection()
    try:
        summary = run_candidates(JOB_ID, cfg, conn)
    finally:
        conn.close()

    assert summary["candidates"] == 0
    assert summary["skipped"] is False
    data = json.loads((job_dir / "candidates.json").read_text())
    assert data["schema_version"] == 1
    assert data["candidates"] == []
    assert _candidate_rows(db, JOB_ID) == []
    assert _stages(db, JOB_ID)["candidates"]["state"] == "done"
    assert db.get_job(JOB_ID)["state"] == "queued"


def test_rerun_after_failure_retries(tmp_path, monkeypatch):
    db, cfg, job_dir = _setup(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("generator exploded")

    monkeypatch.setattr(cands_mod, "generate_candidates", boom)

    conn = db.get_connection()
    try:
        with pytest.raises(StageError, match="generator exploded"):
            run_candidates(JOB_ID, cfg, conn)
    finally:
        conn.close()

    stages = _stages(db, JOB_ID)
    assert db.get_job(JOB_ID)["state"] == "failed"
    assert stages["candidates"]["state"] == "failed"
    assert "generator exploded" in stages["candidates"]["error"]
    assert _candidate_rows(db, JOB_ID) == []

    # Unpatch -> the failed stage is retried and now succeeds.
    monkeypatch.undo()
    conn = db.get_connection()
    try:
        summary = run_candidates(JOB_ID, cfg, conn)
    finally:
        conn.close()

    assert summary["candidates"] == 20
    assert summary["skipped"] is False
    assert _stages(db, JOB_ID)["candidates"]["state"] == "done"
    assert len(_candidate_rows(db, JOB_ID)) == 20
    assert db.get_job(JOB_ID)["state"] == "queued"


# ---------------------------------------------------------------------------
# Script (scripts/candidates_job.py)
# ---------------------------------------------------------------------------


def _load_candidates_job_script() -> object:
    path = REPO_ROOT / "scripts" / "candidates_job.py"
    spec = importlib.util.spec_from_file_location(
        "clipper_candidates_job_script", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidates_job_script_success(tmp_path, monkeypatch, capsys):
    db, cfg, job_dir = _setup(tmp_path)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_candidates_job_script()

    rc = script.main(JOB_ID)

    assert rc == 0
    out = capsys.readouterr().out
    assert "job:" in out and JOB_ID in out
    assert "candidates: 20" in out
    assert "top 5:" in out
    # [start-end] duration score preview lines
    lines = [
        ln for ln in out.splitlines()
        if ln.strip().startswith("[") and "]" in ln
    ]
    assert len(lines) == 5
    assert all("." in ln.split()[0] for ln in lines)
    assert "How many of your visitors could" in "".join(lines)
    assert str(job_dir / "candidates.json") in out


def test_candidates_job_script_missing_job_exits_1(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    Database(cfg)  # initialize the DB file
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_candidates_job_script()

    rc = script.main("no-such-job")

    assert rc == 1
    assert "not found" in capsys.readouterr().out


def test_candidates_job_script_failure_exits_1(tmp_path, monkeypatch, capsys):
    db, cfg, job_dir = _setup(tmp_path)
    (job_dir / "energy.json").unlink()
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_candidates_job_script()

    rc = script.main(JOB_ID)

    assert rc == 1
    out = capsys.readouterr().out
    assert "energy.json" in out
    assert "scripts/analyze_job.py" in out
