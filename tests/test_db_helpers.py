"""M8a DB helper tests — CPU only, temp DB.

Covers the additive `clipper.db` worker helpers: get_job, list_jobs,
next_queued_job, set_job_state, set_stage_state, reset_failed_stages,
write_job_report (idempotency + "resets failed only, not done").
"""

import json

import pytest

from clipper.config import Config, PathsConfig
from clipper.db import (
    Database,
    get_job,
    list_jobs,
    next_queued_job,
    reset_failed_stages,
    set_job_state,
    set_stage_state,
    write_job_report,
)
from clipper.models import JobReport, StageReport

JOB_A = "m8a-job-a"
JOB_B = "m8a-job-b"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


@pytest.fixture
def env(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(data_dir)))
    db = Database(cfg)
    assert db.create_job(JOB_A, "/tmp/a.mp4", "client", "niche")
    assert db.create_job(JOB_B, "/tmp/b.mp4", "client", "niche")
    conn = db.get_connection()
    yield {"db": db, "conn": conn, "tmp": tmp_path}
    conn.close()


def _report(job_id: str) -> JobReport:
    return JobReport(
        job_id=job_id,
        state="queued",
        clip_count=0,
        stages=[
            StageReport(name="prep", required=["ingest"], state="done")
        ],
    )


# ---------------------------------------------------------------------------
# get_job / list_jobs / next_queued_job
# ---------------------------------------------------------------------------


def test_get_job_returns_row_and_none_for_unknown(env):
    conn = env["conn"]
    row = get_job(conn, JOB_A)
    assert row is not None and row["job_id"] == JOB_A and row["state"] == "queued"
    assert get_job(conn, "missing") is None


def test_list_jobs_all_and_filtered(env):
    conn = env["conn"]
    assert {j["job_id"] for j in list_jobs(conn)} == {JOB_A, JOB_B}
    set_job_state(conn, JOB_A, "running")
    assert [j["job_id"] for j in list_jobs(conn, state="queued")] == [JOB_B]
    assert [j["job_id"] for j in list_jobs(conn, state="running")] == [JOB_A]
    assert list_jobs(conn, state="failed") == []


def test_next_queued_job_oldest_first_then_none(env):
    conn = env["conn"]
    assert next_queued_job(conn)["job_id"] == JOB_A  # created first
    assert len(list_jobs(conn, state="queued")) == 2
    set_job_state(conn, JOB_A, "running")
    set_job_state(conn, JOB_A, "queued")  # back to pool, still next
    assert next_queued_job(conn)["job_id"] == JOB_A


# ---------------------------------------------------------------------------
# set_job_state / set_stage_state
# ---------------------------------------------------------------------------


def test_set_job_state_idempotent(env):
    conn = env["conn"]
    set_job_state(conn, JOB_A, "running")
    assert get_job(conn, JOB_A)["state"] == "running"
    set_job_state(conn, JOB_A, "running")  # idempotent
    row1 = get_job(conn, JOB_A)
    set_job_state(conn, JOB_A, "running")
    row2 = get_job(conn, JOB_A)
    assert row1["state"] == row2["state"] == "running"
    # unknown job: no crash, nothing matched
    set_job_state(conn, "missing", "running")


def test_set_stage_state_lifecycle(env):
    conn = env["conn"]
    set_stage_state(conn, JOB_A, "ingest", "running")
    row = _stage_row(conn, JOB_A, "ingest")
    assert row["state"] == "running"
    assert row["started_at"] and row["finished_at"] is None

    first_started = row["started_at"]
    set_stage_state(conn, JOB_A, "ingest", "running")  # re-entry keeps start
    assert _stage_row(conn, JOB_A, "ingest")["started_at"] == first_started

    set_stage_state(conn, JOB_A, "ingest", "done", artifact_path="/j/ingest.json")
    row = _stage_row(conn, JOB_A, "ingest")
    assert row["state"] == "done"
    assert row["artifact_path"] == "/j/ingest.json"
    assert row["finished_at"] and row["started_at"] == first_started
    assert row["error"] is None


def test_set_stage_state_failed_then_resettable(env):
    conn = env["conn"]
    set_stage_state(conn, JOB_A, "scoring", "failed", error="boom")
    row = _stage_row(conn, JOB_A, "scoring")
    assert row["state"] == "failed" and row["error"] == "boom"


def test_set_stage_state_pending_resets_row(env):
    conn = env["conn"]
    set_stage_state(conn, JOB_A, "shots", "done", artifact_path="/j/shots.json")
    set_stage_state(conn, JOB_A, "shots", "pending")
    row = _stage_row(conn, JOB_A, "shots")
    assert (
        row["state"]
        == "pending"
        and row["artifact_path"] is None
        and row["error"] is None
        and row["started_at"] is None
        and row["finished_at"] is None
    )


def _stage_row(conn, job_id: str, name: str) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM stages WHERE job_id = ? AND name = ?", (job_id, name)
    )
    return dict(cur.fetchone())


# ---------------------------------------------------------------------------
# reset_failed_stages
# ---------------------------------------------------------------------------


def test_reset_failed_stages_resets_failed_only_not_done(env):
    conn = env["conn"]
    set_stage_state(conn, JOB_A, "ingest", "done", artifact_path="/j/i.json")
    set_stage_state(conn, JOB_A, "probe", "done", artifact_path="/j/p.json")
    set_stage_state(conn, JOB_A, "proxy", "failed", error="boom")
    set_stage_state(conn, JOB_A, "audio", "failed", error="boom again")
    set_stage_state(conn, JOB_A, "shots", "pending")

    assert reset_failed_stages(conn, JOB_A) == 2

    assert _stage_row(conn, JOB_A, "ingest")["state"] == "done"
    assert _stage_row(conn, JOB_A, "probe")["state"] == "done"
    proxy = _stage_row(conn, JOB_A, "proxy")
    audio = _stage_row(conn, JOB_A, "audio")
    for row in (proxy, audio):
        assert (
            row["state"]
            == "pending"
            and row["error"] is None
            and row["artifact_path"] is None
        )
    assert _stage_row(conn, JOB_A, "shots")["state"] == "pending"
    assert reset_failed_stages(conn, JOB_A) == 0  # idempotent second call


def test_reset_failed_stages_scoped_to_job(env):
    conn = env["conn"]
    set_stage_state(conn, JOB_A, "proxy", "failed", error="a")
    set_stage_state(conn, JOB_B, "proxy", "failed", error="b")
    assert reset_failed_stages(conn, JOB_A) == 1
    assert _stage_row(conn, JOB_A, "proxy")["state"] == "pending"
    assert _stage_row(conn, JOB_B, "proxy")["state"] == "failed"


# ---------------------------------------------------------------------------
# write_job_report
# ---------------------------------------------------------------------------


def test_write_job_report_round_trip_and_idempotent(env):
    conn = env["conn"]
    report = _report(JOB_A)
    write_job_report(conn, JOB_A, report)
    stored = _stored_report(conn, JOB_A)
    assert JobReport.model_validate_json(stored) == report

    report2 = report.model_copy(
        update={"state": "awaiting_qc", "clip_count": 3, "finished_at": "t1"}
    )
    write_job_report(conn, JOB_A, report2)
    # upsert: exactly ONE row for the job, holding the LATEST report
    assert JobReport.model_validate_json(_stored_report(conn, JOB_A)) == report2
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM job_reports WHERE job_id = ?", (JOB_A,)
    ).fetchone()["n"]
    assert count == 1


def test_write_job_report_missing_job_table_is_migrated(env):
    # Migration v3 created the table on Database init for a temp DB.
    conn = env["conn"]
    write_job_report(conn, JOB_B, _report(JOB_B))
    assert JobReport.model_validate_json(_stored_report(conn, JOB_B)).job_id == JOB_B


def _stored_report(conn, job_id: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT report_json FROM job_reports WHERE job_id = ?", (job_id,))
    row = cur.fetchone()
    assert row is not None
    assert json.loads(row["report_json"])["schema_version"] == 1
    return row["report_json"]
