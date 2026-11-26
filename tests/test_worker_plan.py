"""M8a worker plan tests — CPU only, temp DB, no adapters/GPU/network.

Covers: STAGE_PLAN order exact, required_artifacts, coarse done detection
(done/missing/invalid artifacts), next_coarse_stage skipping completed
coarse stages, failed_fine_stages (failed only).
"""

import json
from pathlib import Path

import pytest

from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.worker_plan import (
    DONE_STATES,
    STAGE_PLAN,
    failed_fine_stages,
    is_coarse_done,
    next_coarse_stage,
    required_artifacts,
)

JOB_ID = "m8a-job"

PLAN_NAMES = (
    "prep",
    "transcribe",
    "analyze",
    "candidates",
    "scoring",
    "selection",
    "clip_plans",
    "layout",
    "render",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


@pytest.fixture
def fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(data_dir)))
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/src.mp4", "client", "niche")
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    conn = db.get_connection()
    yield {"db": db, "conn": conn, "cfg": cfg, "job_dir": job_dir, "tmp": tmp_path}
    conn.close()


def _mark_stage(
    conn, name: str, state: str, artifact: Path | None = None, error: str | None = None
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO stages "
        "(job_id, name, state, artifact_path, error) "
        "VALUES (?, ?, ?, ?, ?)",
        (JOB_ID, name, state, str(artifact) if artifact else None, error),
    )
    conn.commit()


def _write_json_artifact(job_dir: Path, filename: str, payload: dict) -> Path:
    path = job_dir / filename
    path.write_text(json.dumps(payload) + "\n")
    return path


def _mark_done(conn, name: str, artifact: Path, state: str = "done") -> None:
    _mark_stage(conn, name, state, artifact)


# ---------------------------------------------------------------------------
# STAGE_PLAN shape and order
# ---------------------------------------------------------------------------


def test_stage_plan_order_is_exact():
    assert tuple(spec.name for spec in STAGE_PLAN) == PLAN_NAMES


def test_stage_plan_required_fine_grained_stages():
    expected = {
        "prep": ("ingest", "probe", "proxy", "audio"),
        "transcribe": ("transcribe", "sentences"),
        "analyze": ("shots", "silence", "energy"),
        "candidates": ("candidates",),
        "scoring": ("scoring",),
        "selection": ("selection",),
        "clip_plans": ("clip_plans",),
        "layout": ("layout",),
        "render": ("render",),
    }
    for spec in STAGE_PLAN:
        assert spec.required == expected[spec.name]


def test_required_artifacts_returns_canonical_names():
    spec = {s.name: s for s in STAGE_PLAN}
    assert required_artifacts("prep") == spec["prep"].artifacts
    assert (
        set(required_artifacts("prep"))
        == {"ingest.json", "probe.json", "proxy.mp4", "audio.wav"}
    )
    assert required_artifacts("scoring") == ("scores.json",)
    assert required_artifacts("render") == ("clips",)  # M8a-J3


def test_required_artifacts_unknown_stage_raises():
    with pytest.raises(ValueError, match="unknown coarse stage"):
        required_artifacts("shots")  # a FINE-grained stage name
    with pytest.raises(ValueError):
        required_artifacts("nope")


def test_done_states_accept_both_done_and_completed():
    # M8a-J1: M1/M2 write "completed", M3+ write "done"
    assert DONE_STATES == frozenset({"done", "completed"})


# ---------------------------------------------------------------------------
# is_coarse_done
# ---------------------------------------------------------------------------


def _json(name: str, job_dir: Path) -> Path:
    return _write_json_artifact(job_dir, name, {"schema_version": 1})


def _complete_prep(conn, job_dir: Path, state: str = "done"):
    """Mark all four prep fine-grained stages done with existing artifacts."""
    _mark_done(conn, "ingest", _json("ingest.json", job_dir))
    _mark_done(conn, "probe", _json("probe.json", job_dir))
    proxy = job_dir / "proxy.mp4"
    proxy.write_bytes(b"fake")
    _mark_done(conn, "proxy", proxy, state=state)
    audio = job_dir / "audio.wav"
    audio.write_bytes(b"RIFFfakewav")
    _mark_done(conn, "audio", audio, state=state)


def test_is_coarse_done_complete(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir)
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is True


def test_is_coarse_done_accepts_completed_state(fixture):
    # M8a-J1: M1/M2 terminal state is "completed"
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir, state="completed")
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is True


def test_is_coarse_done_false_when_stage_state_not_done(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir)
    _mark_stage(conn, "probe", "running", job_dir / "probe.json")
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is False


def test_is_coarse_done_false_when_stage_row_missing(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir)
    conn.execute("DELETE FROM stages WHERE job_id = ? AND name = 'audio'", (JOB_ID,))
    conn.commit()
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is False


def test_is_coarse_done_false_when_artifact_missing(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    probe = _write_json_artifact(job_dir, "probe.json", {"schema_version": 1})
    _complete_prep(conn, job_dir)
    probe.unlink()
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is False


def test_is_coarse_done_false_when_schema_version_invalid(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir)
    (job_dir / "probe.json").write_text(json.dumps({"schema_version": 2}))
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is False
    # and non-JSON garbage
    (job_dir / "probe.json").write_text("not json at all")
    assert is_coarse_done(conn, JOB_ID, STAGE_PLAN[0]) is False


def test_is_coarse_done_false_when_no_rows(fixture):
    assert is_coarse_done(fixture["conn"], JOB_ID, STAGE_PLAN[0]) is False


# ---------------------------------------------------------------------------
# next_coarse_stage
# ---------------------------------------------------------------------------


def test_next_coarse_stage_first_when_nothing_done(fixture):
    assert next_coarse_stage(fixture["conn"], JOB_ID) is STAGE_PLAN[0]


def test_next_coarse_stage_skips_completed_coarse_stages(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir)
    for name, fname in (
        ("transcribe", "words.json"),
        ("sentences", "sentences.json"),
    ):
        _mark_done(conn, name, _json(fname, job_dir))
    for name in ("shots", "silence", "energy"):
        _mark_done(conn, name, _json(f"{name}.json", job_dir))
    assert next_coarse_stage(conn, JOB_ID) is STAGE_PLAN[3]  # candidates


def test_next_coarse_stage_none_when_everything_done(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    for spec in STAGE_PLAN:
        for name in spec.required:
            if name in ("proxy", "audio"):
                art = job_dir / (f"{name}.bin")
                art.write_bytes(b"x")
            elif name in ("layout", "render"):
                clips_dir = job_dir / "clips"
                clips_dir.mkdir(parents=True, exist_ok=True)
                art = clips_dir
            else:
                art = _write_json_artifact(
                    job_dir, f"{name}.json", {"schema_version": 1, "x": name}
                )
            _mark_done(conn, name, art)
    assert next_coarse_stage(conn, JOB_ID) is None


def test_next_coarse_stage_skips_only_when_artifacts_exist(fixture):
    # A done stage whose artifact file vanished must NOT be treated as done.
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _complete_prep(conn, job_dir)
    (job_dir / "proxy.mp4").unlink()
    assert next_coarse_stage(conn, JOB_ID) is STAGE_PLAN[0]


# ---------------------------------------------------------------------------
# failed_fine_stages
# ---------------------------------------------------------------------------


def test_failed_fine_stages_returns_only_failed(fixture):
    conn, job_dir = fixture["conn"], fixture["job_dir"]
    _write_json_artifact(job_dir, "probe.json", {"schema_version": 1})
    _mark_stage(conn, "probe", "failed", None, error="boom")
    _mark_done(conn, "ingest", _json("ingest.json", job_dir))
    _mark_stage(conn, "scoring", "failed", None, error="llm down")
    _mark_stage(conn, "shots", "running", None)
    assert failed_fine_stages(conn, JOB_ID) == ["probe", "scoring"]


def test_failed_fine_stages_empty_when_none_failed(fixture):
    assert failed_fine_stages(fixture["conn"], JOB_ID) == []


def test_failed_fine_stages_uses_pipeline_order_not_db_order(fixture):
    conn = fixture["conn"]
    _mark_stage(conn, "energy", "failed", None, error="e2")
    _mark_stage(conn, "silence", "failed", None, error="e1")
    _mark_stage(conn, "probe", "failed", None, error="e0")
    # pipeline order: probe (prep) before silence/energy (analyze)
    assert failed_fine_stages(conn, JOB_ID) == ["probe", "silence", "energy"]


def test_failed_fine_stages_ignores_other_jobs(fixture):
    conn = fixture["conn"]
    conn.execute(
        "INSERT INTO jobs (job_id, source_path, client, niche, state, created_at, "
        "updated_at) VALUES ('other', '/tmp/x', 'c', 'n', 'failed', 't', 't')"
    )
    conn.execute(
        "INSERT INTO stages (job_id, name, state, error) "
        "VALUES ('other', 'probe', 'failed', 'x')"
    )
    conn.commit()
    assert failed_fine_stages(conn, JOB_ID) == []
