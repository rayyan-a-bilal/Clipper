"""
M8b worker-loop + CLI tests — CPU only.

FakeStageRunner runners, a temp data dir and a temp DB. No GPU, no
network, no LLM, no real M1–M7 stage wiring (that lands in M8c). The
worker loop semantics under test are the fixed M8 decisions in
`docs/spec/m8.md` (one job at a time, coarse stage order/skip, failure
+ report, retry, job states, report.json + job_reports).
"""

import fcntl
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import clipper.cli as cli
from clipper import db as db_helpers
from clipper import worker
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.worker import FakeStageRunner, WorkerError
from clipper.worker_plan import STAGE_PLAN

JOB_ID = "job-m8b"
OTHER_JOB_ID = "job-m8b-older"

# (fine stage, artifact name) pairs that make a coarse stage "done".
_COARSE_FIXTURES: dict[str, list[tuple[str, str]]] = {
    "prep": [
        ("ingest", "ingest.json"),
        ("probe", "probe.json"),
        ("proxy", "proxy.mp4"),
        ("audio", "audio.wav"),
    ],
    "transcribe": [
        ("transcribe", "words.json"),
        ("sentences", "sentences.json"),
    ],
    "analyze": [
        ("shots", "shots.json"),
        ("silence", "silence.json"),
        ("energy", "energy.json"),
    ],
    "candidates": [("candidates", "candidates.json")],
    "scoring": [("scoring", "scores.json")],
    "selection": [("selection", "selection.json")],
    "clip_plans": [("clip_plans", "clip_plans.json")],
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


def _cfg(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(data_dir=str(data_dir)))


def _setup(tmp_path: Path) -> tuple[object, object, Config, Path]:
    """cfg + Database + one queued job + its job dir."""
    cfg = _cfg(tmp_path)
    dbobj = Database(cfg)
    assert dbobj.create_job(JOB_ID, "/tmp/m8b-source.mp4", "client", "niche")
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    conn = dbobj.get_connection()
    return dbobj, conn, cfg, job_dir


def _all_fakes(**overrides) -> dict[str, FakeStageRunner]:
    """A FakeStageRunner per coarse stage; `overrides` map name->error."""
    return {
        spec.name: FakeStageRunner(error=overrides.get(spec.name))
        for spec in STAGE_PLAN
    }


def _mark_coarse_done(
    conn: object, job_id: str, coarse: str, job_dir: Path
) -> None:
    """Mark a coarse stage's fine rows done with valid artifact files."""
    assert coarse in _COARSE_FIXTURES
    for name, artifact in _COARSE_FIXTURES[coarse]:
        path = job_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps({"schema_version": 1, artifact: {}}))
        else:
            path.write_bytes(b"\x00fake")
        state = "completed" if name in ("transcribe", "sentences") else "done"
        db_helpers.set_stage_state(
            conn, job_id, name, state, artifact_path=str(path),
        )


def _stage_states(conn: object, job_id: str) -> dict[str, str]:
    cursor = conn.cursor()
    cursor.execute("SELECT name, state FROM stages WHERE job_id = ?", (job_id,))
    return {row[0]: row[1] for row in cursor.fetchall()}


def _report_file(cfg: Config, job_id: str) -> Path:
    return cfg.resolved_data_dir / "jobs" / job_id / "report.json"


# ---------------------------------------------------------------------------
# run_job
# ---------------------------------------------------------------------------


def test_run_job_processes_fake_stages_in_order(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    order: list[str] = []
    runners: dict[str, FakeStageRunner] = {}
    for spec in STAGE_PLAN:

        class Recording(FakeStageRunner):
            coarse_name = spec.name

            def run(self, job_id, cfg_, conn_):
                order.append(type(self).coarse_name)
                super().run(job_id, cfg_, conn_)

        runners[spec.name] = Recording()
    report = worker.run_job(JOB_ID, cfg, conn, runners)

    assert order == [s.name for s in STAGE_PLAN]  # all 9, exact plan order
    assert [r.name for r in report.stages] == [s.name for s in STAGE_PLAN]
    assert all(r.state == "done" for r in report.stages)
    row = db_helpers.get_job(conn, JOB_ID)
    assert row["state"] == "done"  # zero clips -> done
    assert report.clip_count == 0
    # report.json file + DB row both written
    data = json.loads(_report_file(cfg, JOB_ID).read_text())
    assert data["schema_version"] == 1
    assert data["state"] == "done"
    assert data["stages"] == [r.model_dump() for r in report.stages]
    cursor = conn.cursor()
    cursor.execute("SELECT report_json FROM job_reports WHERE job_id = ?", (JOB_ID,))
    assert cursor.fetchone() is not None


def test_completed_coarse_stages_are_skipped(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    _mark_coarse_done(conn, JOB_ID, "prep", job_dir)
    _mark_coarse_done(conn, JOB_ID, "transcribe", job_dir)
    _mark_coarse_done(conn, JOB_ID, "analyze", job_dir)
    runners = _all_fakes()
    report = worker.run_job(JOB_ID, cfg, conn, runners)

    assert runners["prep"].calls == []
    assert runners["transcribe"].calls == []
    assert runners["analyze"].calls == []
    assert runners["candidates"].calls == [JOB_ID]
    states = {r.name: r.state for r in report.stages}
    assert states["prep"] == "skipped"
    assert states["transcribe"] == "skipped"
    assert states["analyze"] == "skipped"
    assert states["candidates"] == "done"


def test_failure_marks_job_failed_and_writes_report(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    runners = _all_fakes(scoring=RuntimeError("llm batch exploded"))
    with pytest.raises(WorkerError, match="scoring"):
        worker.run_job(JOB_ID, cfg, conn, runners)

    assert db_helpers.get_job(conn, JOB_ID)["state"] == "failed"
    # the failed fine stage is marked failed with the error text
    stages = _stage_states(conn, JOB_ID)
    assert stages["scoring"] == "failed"
    # later stages never started
    assert runners["selection"].calls == []
    assert runners["render"].calls == []
    # report.json carries the error (JobReport + the coarse stage)
    data = json.loads(_report_file(cfg, JOB_ID).read_text())
    assert data["state"] == "failed"
    assert "llm batch exploded" in data["error"]
    scoring_report = next(s for s in data["stages"] if s["name"] == "scoring")
    assert scoring_report["state"] == "failed"
    assert "llm batch exploded" in scoring_report["error"]
    # earlier stages completed before the failure are reported done
    prep_report = next(s for s in data["stages"] if s["name"] == "prep")
    assert prep_report["state"] == "done"
    # missing job fails cleanly
    with pytest.raises(WorkerError, match="unknown job"):
        worker.run_job("nope", cfg, conn, _all_fakes())


def test_retry_resets_failed_stages_only(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    # candidates is already done; first run fails at scoring
    _mark_coarse_done(conn, JOB_ID, "candidates", job_dir)
    bad = _all_fakes(scoring=ValueError("first attempt fails"))
    with pytest.raises(WorkerError):
        worker.run_job(JOB_ID, cfg, conn, bad)

    stages = _stage_states(conn, JOB_ID)
    assert stages["candidates"] == "done"  # untouched by failure
    assert stages["scoring"] == "failed"

    # retry: only the failed stage re-runs; done stages stay skipped
    good = _all_fakes()
    report = worker.run_job(JOB_ID, cfg, conn, good, retry=True)

    assert good["candidates"].calls == []  # done stage: never re-run
    assert good["scoring"].calls == [JOB_ID]
    assert [r.name for r in report.stages if r.state == "failed"] == []
    assert report.state == "done"


def test_zero_clips_done_and_clips_present_awaiting_qc(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    report = worker.run_job(JOB_ID, cfg, conn, _all_fakes())
    assert report.state == "done"
    assert db_helpers.get_job(conn, JOB_ID)["state"] == "done"

    # one clip present -> awaiting_qc (zero clips is a valid success too)
    conn.execute(
        "INSERT INTO clips "
        "(clip_id, job_id, plan_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'pending_qc', 't0', 't0')",
        (f"{JOB_ID}-clip01", JOB_ID, "{}"),
    )
    conn.commit()
    report2 = worker.run_job(JOB_ID, cfg, conn, _all_fakes())
    assert report2.clip_count == 1
    assert report2.state == "awaiting_qc"
    assert db_helpers.get_job(conn, JOB_ID)["state"] == "awaiting_qc"


def test_worker_lock_prevents_concurrent_run_job(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    lock_file = worker.lock_path(cfg)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    held = lock_file.open("a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(WorkerError, match="worker lock"):
            worker.run_job(JOB_ID, cfg, conn, _all_fakes())
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    # the job never started: it is still queued, no report written
    assert db_helpers.get_job(conn, JOB_ID)["state"] == "queued"
    assert not _report_file(cfg, JOB_ID).exists()
    # and the lock is usable afterwards
    assert worker.run_job(JOB_ID, cfg, conn, _all_fakes()).state == "done"


# ---------------------------------------------------------------------------
# run_next
# ---------------------------------------------------------------------------


def test_run_next_processes_one_queued_job(tmp_path):
    dbobj, conn, cfg, job_dir = _setup(tmp_path)
    # TWO queued jobs: run_next takes the OLDEST first (jobs are ordered
    # by created_at — JOB_ID was created first in _setup) and the second
    # call takes the remaining one.
    assert dbobj.create_job(
        OTHER_JOB_ID, "/tmp/oldest.mp4", "c", "n")
    runners = _all_fakes()
    report = worker.run_next(cfg, conn, runners)
    assert report is not None
    assert report.job_id == JOB_ID
    assert runners["prep"].calls == [JOB_ID]
    assert db_helpers.get_job(conn, OTHER_JOB_ID)["state"] == "queued"
    # the second call takes the next oldest job
    runners2 = _all_fakes()
    report2 = worker.run_next(cfg, conn, runners2)
    assert report2 is not None
    assert report2.job_id == OTHER_JOB_ID
    assert runners2["prep"].calls == [OTHER_JOB_ID]
    # and the queue is now empty
    assert worker.run_next(cfg, conn, _all_fakes()) is None


# ---------------------------------------------------------------------------
# CLI (typer CliRunner, FakeStageRunner wiring)
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Data dir under tmp + fake runners injected into the CLI."""
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    runners = _all_fakes()
    monkeypatch.setattr(cli, "_stage_runners", lambda cfg: runners)
    return tmp_path, runners


def test_cli_run_status_ls_report(cli_env):
    tmp_path, runners = cli_env
    cfg = Config.load_config()
    dbobj = Database(cfg)
    assert dbobj.create_job(JOB_ID, "/tmp/cli-source.mp4", "c", "n")
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    _mark_coarse_done(dbobj.get_connection(), JOB_ID, "prep", job_dir)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["run", JOB_ID])
    assert result.exit_code == 0, result.output
    assert "state: done" in result.output
    assert "report.json" in result.output
    data = json.loads(_report_file(cfg, JOB_ID).read_text())
    assert data["state"] == "done"
    skipped = [s for s in data["stages"] if s["name"] == "prep"]
    assert skipped and skipped[0]["state"] == "skipped"

    result = runner.invoke(cli.app, ["status", JOB_ID])
    assert result.exit_code == 0, result.output
    assert "State: done" in result.output
    assert "prep" in result.output
    assert "render" in result.output

    result = runner.invoke(cli.app, ["ls"])
    assert result.exit_code == 0, result.output
    assert JOB_ID in result.output

    result = runner.invoke(cli.app, ["ls", "--state", "done"])
    assert result.exit_code == 0, result.output
    assert JOB_ID in result.output

    result = runner.invoke(cli.app, ["report", JOB_ID])
    assert result.exit_code == 0, result.output
    assert "report.json" in result.output
    assert "state: done" in result.output
    assert "clips: 0" in result.output


def test_cli_run_next_and_failure(cli_env, monkeypatch: pytest.MonkeyPatch):
    tmp_path, runners = cli_env
    cfg = Config.load_config()
    dbobj = Database(cfg)
    assert dbobj.create_job(JOB_ID, "/tmp/cli-source.mp4", "c", "n")
    runner = CliRunner()

    result = runner.invoke(cli.app, ["run", "--next"])
    assert result.exit_code == 0, result.output
    assert "state: done" in result.output
    assert dbobj.get_job(JOB_ID)["state"] == "done"

    # empty queue: clean "no queued jobs"
    result = runner.invoke(cli.app, ["run", "--next"])
    assert result.exit_code == 0, result.output
    assert "No queued jobs" in result.output

    # a failing runner: exit 1, clean error, job failed
    new_fakes = _all_fakes(scoring=RuntimeError("boom from cli fake"))
    monkeypatch.setattr(cli, "_stage_runners", lambda cfg: new_fakes)
    assert dbobj.create_job("job-again", "/tmp/x.mp4", "c", "n")
    dbobj.update_job_state("job-again", "queued")
    result = runner.invoke(cli.app, ["run", "job-again"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert dbobj.get_job("job-again")["state"] == "failed"

    # mutually exclusive: JOB_ID and --next together
    result = runner.invoke(cli.app, ["run", JOB_ID, "--next"])
    assert result.exit_code == 1
    assert "exactly one" in result.output
