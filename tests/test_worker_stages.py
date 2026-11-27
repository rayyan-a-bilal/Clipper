"""M8c — the REAL M1–M7 stage runners (clipper.worker_stages) tests.

CPU-only (M8c task card): every adapter factory, the managed-ollama
runtime and the M1–M7 stage functions are injected/monkeypatched — no
real Ollama, GPU, network or subprocess anywhere. The M8b worker-loop,
CLI and stage-contract suites cover their own layers; here we verify
the thin runner wiring: one real function per coarse stage, exact
arguments, managed-ollama usage exactly as m8.md specifies, and the
M8b worker loop driving it end-to-end.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from clipper import worker
from clipper import worker_stages as ws
from clipper.adapters.llm import LLMError
from clipper.config import Config, PathsConfig
from clipper.db import Database, set_stage_state

JOB_ID = "test-m8c-job"
SOURCE = "/tmp/clipper-m8c-source.mp4"
CLIENT_NAME = "acme"
NICHE = "tech"
MANAGED_BASE_URL = "http://127.0.0.1:11435-managed-test"

# Fine-stage/artifact fixtures per coarse stage (m8.md table + layout/
# render), used by the worker-loop tests to mark coarse stages skipped.
COARSE_FIXTURES = {
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
    "scoring": [("scores", "scores.json")],
    "selection": [("selection", "selection.json")],
    "clip_plans": [("clip_plans", "clip_plans.json")],
    "layout": [("layout", "layout.json")],
    "render": [("render", "clips")],
}

def _setup(tmp_path):
    """cfg + Database + connection + one job row (same shape as
    tests/test_worker.py)."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(tmp_path / "data")))
    database = Database(cfg)
    database.create_job(JOB_ID, SOURCE, CLIENT_NAME, NICHE)
    return cfg, database, database.get_connection()

def _patch_factories(monkeypatch) -> None:
    """Patch every adapter factory the runners use (CPU test fakes).

    The patches target the EXACT module objects `clipper.worker_stages`
    bound at import time (the runners call e.g. `media_adapters.get_adapter`
    through those very objects). String-path monkeypatching
    ("clipper.adapters.media.get_adapter") is deliberately AVOIDED:
    `tests/test_faces.py` (pre-existing M5 test)
    pops `sys.modules["clipper.adapters"]` and re-imports the package,
    leaving a fresh package object WITHOUT the `media` attribute when the
    full suite runs — a string path resolved through `sys.modules` then
    fails. Bound-object patching is immune to that isolation quirk."""
    monkeypatch.setattr(
        ws.media_adapters, "get_adapter", lambda: "MEDIA_ADAPTER"
    )
    monkeypatch.setattr(
        ws.asr_adapters, "get_adapter", lambda cfg: "ASR_ADAPTER"
    )
    monkeypatch.setattr(
        ws.scene_adapters, "get_scene_adapter", lambda: "SCENES_ADAPTER"
    )
    monkeypatch.setattr(
        ws.vision_adapters, "get_vision_adapter", lambda cfg: "VISION_ADAPTER"
    )
    monkeypatch.setattr(
        ws.face_adapters, "get_face_detector",
        lambda cfg, cpu=False: "FACE_DETECTOR",
    )
    monkeypatch.setattr(
        ws.render_adapters, "get_render_adapter",
        lambda cfg, cpu_render=None: "RENDER_ADAPTER",
    )
    monkeypatch.setattr(
        ws.llm_adapters, "get_llm_client",
        lambda cfg, base_url: f"LLM_CLIENT<{base_url}>",
    )

def _fake_managed(monkeypatch, events, fail=None) -> None:
    """Record-style stand-in for llm_runtime.managed_ollama:
    `managed_ollama(cfg)` is a context manager whose base_url is
    MANAGED_BASE_URL; `enter`/`exit` are logged even when the managed
    call site raises (exit is `finally`). `fail` raises at entry (the
    managed runtime cannot start)."""

    @contextmanager
    def fake(cfg):
        events.append("enter")
        if fail is not None:
            raise fail
        try:
            yield MANAGED_BASE_URL
        finally:
            events.append("exit")

    monkeypatch.setattr(ws, "managed_ollama", fake)

def _stage_spy(monkeypatch, target, calls, result=None, exc=None) -> None:
    """Replace the mapped stage function with a recording spy."""

    def spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr(target, spy)

def _write_selection(cfg, selected_flags) -> None:
    """Write a minimal valid selection.json for the test job."""
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decisions": [
                    {"cand_id": f"candidate-{n}", "selected": flag}
                    for n, flag in enumerate(selected_flags)
                ],
            }
        ),
        encoding="utf-8",
    )

def _mark_coarse_done(cfg, conn, job_id, name) -> None:
    """Write the coarse stage's fine-stage done rows + artifacts so the
    M8b worker loop's skip check treats it as already complete
    (mirrors tests/test_worker.py::_mark_coarse_done)."""
    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    for fine, artifact in COARSE_FIXTURES[name]:
        path = job_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if artifact.endswith(".json"):
                path.write_text(
                    json.dumps({"schema_version": 1}), encoding="utf-8"
                )
            else:
                path.write_bytes(b"")
        set_stage_state(conn, job_id, fine, "done", artifact_path=str(path))
    set_stage_state(conn, job_id, name, "done")

# ---------------------------------------------------------------------------
# Runner map identity (m8.md: exactly the coarse stage names, no extras)
# ---------------------------------------------------------------------------

def test_runner_map_covers_stage_plan_exactly():
    expected = [spec.name for spec in ws.STAGE_PLAN]
    assert list(ws.STAGE_RUNNERS) == expected
    for name, runner in ws.STAGE_RUNNERS.items():
        assert isinstance(runner, ws.CoarseRunner)
        assert runner.name == name
        assert callable(runner.run)

def test_cli_stage_runners_hook_returns_real_mapping(monkeypatch, tmp_path):
    import clipper.cli as cli

    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    cfg = cli.get_config()
    runners = cli._stage_runners(cfg)
    assert list(runners) == [spec.name for spec in ws.STAGE_PLAN]
    for runner in runners.values():
        assert isinstance(runner, ws.CoarseRunner)

# ---------------------------------------------------------------------------
# Per-stage wiring: the mapped stage function is called exactly once
# with the job row's values / injected adapters (no conn where the
# M1-M7 signature has none).
# ---------------------------------------------------------------------------

def test_prep_calls_run_ingest_once_with_job_values(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(
            monkeypatch,
            "clipper.stages.ingest.run_ingest",
            calls,
            result=(JOB_ID, True, False),
        )
        ws.STAGE_RUNNERS["prep"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        args, kwargs = calls[0]["args"], calls[0]["kwargs"]
        assert args == ()
        assert kwargs == {
            "source": SOURCE,
            "client": CLIENT_NAME,
            "niche": NICHE,
            "cfg": cfg,
            "conn": conn,
            "media": "MEDIA_ADAPTER",
        }
    finally:
        conn.close()

def test_prep_rejects_a_different_returned_job_id(tmp_path, monkeypatch):
    import pytest

    from clipper.stages import StageError

    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(
            monkeypatch,
            "clipper.stages.ingest.run_ingest",
            calls,
            result=("other-job-id", True, False),
        )
        with pytest.raises(StageError) as excinfo:
            ws.STAGE_RUNNERS["prep"].run(JOB_ID, cfg, conn)
        assert "other-job-id" in str(excinfo.value)
        assert JOB_ID in str(excinfo.value)

    finally:
        conn.close()

def test_prep_unknown_job_raises_clean_error(tmp_path, monkeypatch):
    import pytest

    from clipper.stages import StageError

    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.ingest.run_ingest", calls)
        with pytest.raises(StageError) as excinfo:
            ws.STAGE_RUNNERS["prep"].run("no-such-job", cfg, conn)
        assert "no-such-job" in str(excinfo.value)
        assert calls == []
    finally:
        conn.close()

def test_transcribe_calls_run_transcribe_with_asr_only(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(
            monkeypatch, "clipper.stages.transcribe.run_transcribe", calls, True
        )
        ws.STAGE_RUNNERS["transcribe"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        # M2b signature: run_transcribe(job_id, cfg, asr=...) — NO conn.
        assert calls[0]["args"] == (JOB_ID, cfg)
        assert calls[0]["kwargs"] == {"asr": "ASR_ADAPTER"}
    finally:
        conn.close()

def test_analyze_calls_run_analysis_with_media_and_scenes(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.analyze.run_analysis", calls)
        ws.STAGE_RUNNERS["analyze"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        assert calls[0]["args"] == (
            JOB_ID, cfg, conn, "MEDIA_ADAPTER", "SCENES_ADAPTER"
        )
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

def test_candidates_calls_run_candidates_with_no_adapters(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    try:
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.candidates.run_candidates", calls)
        ws.STAGE_RUNNERS["candidates"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        assert calls[0]["args"] == (JOB_ID, cfg, conn)
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Managed-ollama usage (m8.md / M8 decision 9): scoring always under ONE
# managed runtime; clip_plans only when >=1 selected; selection/layout/
# render NEVER; managed child ALWAYS torn down (exit on success + failure).
# ---------------------------------------------------------------------------

def test_scoring_enters_managed_ollama_once_and_builds_client_from_base_url(
    tmp_path, monkeypatch
):
    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        events = []
        _fake_managed(monkeypatch, events)
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.score.run_scoring", calls)
        ws.STAGE_RUNNERS["scoring"].run(JOB_ID, cfg, conn)

        assert events == ["enter", "exit"]  # exactly ONE managed runtime
        assert len(calls) == 1
        assert calls[0]["args"] == (
            JOB_ID,
            cfg,
            conn,
            f"LLM_CLIENT<{MANAGED_BASE_URL}>",
        )
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

def test_scoring_managed_ollama_torn_down_when_stage_raises(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        events = []
        _fake_managed(monkeypatch, events)
        _stage_spy(
            monkeypatch,
            "clipper.stages.score.run_scoring",
            [],
            exc=RuntimeError("scoring exploded"),
        )
        with pytest.raises(RuntimeError):
            ws.STAGE_RUNNERS["scoring"].run(JOB_ID, cfg, conn)
        # managed child must still be torn down on failure.
        assert events == ["enter", "exit"]
    finally:
        conn.close()

def test_selection_never_enters_managed_ollama(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    try:
        def boom(cfg):
            raise AssertionError("selection must NEVER enter managed_ollama")

        monkeypatch.setattr(ws, "managed_ollama", boom)
        calls = []
        _stage_spy(
            monkeypatch, "clipper.stages.select.run_selection_stage", calls
        )
        ws.STAGE_RUNNERS["selection"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        assert calls[0]["args"] == (JOB_ID, cfg, conn)
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

def test_clip_plans_zero_selected_runs_client_none_without_managed(
    tmp_path, monkeypatch
):
    cfg, database, conn = _setup(tmp_path)
    try:
        _write_selection(cfg, [False, False])

        def boom(cfg):
            raise AssertionError(
                "zero selected clips must NOT enter managed_ollama"
            )

        monkeypatch.setattr(ws, "managed_ollama", boom)
        calls = []
        _stage_spy(
            monkeypatch, "clipper.stages.select.run_clip_plans_stage", calls
        )
        ws.STAGE_RUNNERS["clip_plans"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        # 4th positional arg is the LLM client: must be None (fallback).
        assert calls[0]["args"][:3] == (JOB_ID, cfg, conn)
        assert calls[0]["args"][3] is None
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

def test_clip_plans_selected_clips_enters_managed_ollama_once(
    tmp_path, monkeypatch
):
    cfg, database, conn = _setup(tmp_path)
    try:
        _write_selection(cfg, [True, False])
        _patch_factories(monkeypatch)
        events = []
        _fake_managed(monkeypatch, events)
        calls = []
        _stage_spy(
            monkeypatch, "clipper.stages.select.run_clip_plans_stage", calls
        )
        ws.STAGE_RUNNERS["clip_plans"].run(JOB_ID, cfg, conn)

        assert events == ["enter", "exit"]
        assert len(calls) == 1
        assert calls[0]["args"] == (
            JOB_ID,
            cfg,
            conn,
            f"LLM_CLIENT<{MANAGED_BASE_URL}>",
        )
    finally:
        conn.close()

def test_clip_plans_managed_failure_falls_back_and_keeps_job(
    tmp_path, monkeypatch
):
    cfg, database, conn = _setup(tmp_path)
    try:
        _write_selection(cfg, [True])
        _patch_factories(monkeypatch)
        events = []
        _fake_managed(monkeypatch, events, fail=LLMError("ollama down"))
        calls = []
        spy_stage_path = "clipper.stages.select.run_clip_plans_stage"
        _stage_spy(monkeypatch, spy_stage_path, calls)

        # Must NOT raise: the job keeps going with fallback metadata.
        ws.STAGE_RUNNERS["clip_plans"].run(JOB_ID, cfg, conn)

        assert len(calls) == 1
        assert calls[0]["args"][:3] == (JOB_ID, cfg, conn)
        assert calls[0]["args"][3] is None
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

def test_layout_never_enters_managed_ollama_and_injects_adapters(
    tmp_path, monkeypatch
):
    cfg, database, conn = _setup(tmp_path)
    try:
        def boom(cfg):
            raise AssertionError("layout must NEVER enter managed_ollama")

        monkeypatch.setattr(ws, "managed_ollama", boom)
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.layout.run_layout", calls)
        ws.STAGE_RUNNERS["layout"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        assert calls[0]["args"] == (
            JOB_ID,
            cfg,
            conn,
            "VISION_ADAPTER",
            "FACE_DETECTOR",
        )
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

def test_render_is_the_final_stage_without_managed_ollama(
    tmp_path, monkeypatch
):
    cfg, database, conn = _setup(tmp_path)
    try:
        def boom(cfg):
            raise AssertionError("render must NEVER enter managed_ollama")

        monkeypatch.setattr(ws, "managed_ollama", boom)
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.render.run_render", calls)
        ws.STAGE_RUNNERS["render"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
        assert calls[0]["args"] == (JOB_ID, cfg, conn, "RENDER_ADAPTER")
        assert calls[0]["kwargs"] == {}
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# M8b worker loop driving the REAL runners: failed stage marks the job
# failed + report.json; zero-clip finishes `done`; retry resets only the
# failed fine stage.
# ---------------------------------------------------------------------------

def _patch_all_stages(monkeypatch, spies):
    """Patch every mapped stage function with a recording spy (spies[name])."""
    paths = {
        "prep": "clipper.stages.ingest.run_ingest",
        "transcribe": "clipper.stages.transcribe.run_transcribe",
        "analyze": "clipper.stages.analyze.run_analysis",
        "candidates": "clipper.stages.candidates.run_candidates",
        "scoring": "clipper.stages.score.run_scoring",
        "selection": "clipper.stages.select.run_selection_stage",
        "clip_plans": "clipper.stages.select.run_clip_plans_stage",
        "layout": "clipper.stages.layout.run_layout",
        "render": "clipper.stages.render.run_render",
    }
    for name, target in paths.items():
        calls = spies.setdefault(name, [])
        kwargs = {}
        if name == "prep":
            kwargs["result"] = (JOB_ID, True, False)
        elif name == "transcribe":
            kwargs["result"] = True
        _stage_spy(monkeypatch, target, calls, **kwargs)
    return paths

def test_real_runners_drive_worker_loop_failure_to_failed_and_report(
    tmp_path, monkeypatch
):
    import json

    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        _fake_managed(monkeypatch, [])
        spies = {}
        paths = _patch_all_stages(monkeypatch, spies)
        _stage_spy(  # analyze fails this run
            monkeypatch,
            paths["analyze"],
            spies["analyze"],
            exc=RuntimeError("analyze boom"),
        )
        with pytest.raises(worker.WorkerError):
            worker.run_job(JOB_ID, cfg, conn, ws.STAGE_RUNNERS)

        row = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
        assert row["state"] == "failed"
        report_path = (
            cfg.resolved_data_dir / "jobs" / JOB_ID / "report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["state"] == "failed"
        # the failed coarse stage is recorded; the fine row is failed too
        analyze_report = next(
            s for s in report["stages"] if s["name"] == "analyze"
        )
        assert analyze_report["state"] == "failed"
        shots_row = conn.execute(
            "SELECT state FROM stages WHERE job_id = ? AND name = 'shots'",
            (JOB_ID,),
        ).fetchone()
        assert shots_row["state"] == "failed"
        # stages up to the failure ran exactly once; later ones never ran.
        assert len(spies["prep"]) == 1
        assert len(spies["transcribe"]) == 1
        assert len(spies["analyze"]) == 1
        assert spies["candidates"] == []
        assert spies["render"] == []
    finally:
        conn.close()

def test_real_runners_zero_clip_end_to_end_finishes_done(tmp_path, monkeypatch):
    import json

    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        events = []
        _fake_managed(monkeypatch, events)
        _write_selection(cfg, [False, False])  # zero selected is valid
        spies = {}
        _patch_all_stages(monkeypatch, spies)

        report = worker.run_job(JOB_ID, cfg, conn, ws.STAGE_RUNNERS)

        # m8.md: terminal state = awaiting_qc or done (done with 0 clips).
        assert report.state == "done"
        assert report.clip_count == 0
        assert [s.name for s in report.stages] == [spec.name for spec in ws.STAGE_PLAN]
        for name in spies:
            assert len(spies[name]) == 1, name
        # managed Ollama entered exactly once in the whole job (scoring);
        # clip_plans ran with client=None (zero selected).
        assert events == ["enter", "exit"]
        assert spies["scoring"][0]["args"][3] == f"LLM_CLIENT<{MANAGED_BASE_URL}>"
        assert spies["clip_plans"][0]["args"][3] is None
        # job row + report.json on disk
        row = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
        assert row["state"] == "done"
        report_path = cfg.resolved_data_dir / "jobs" / JOB_ID / "report.json"
        assert json.loads(report_path.read_text(encoding="utf-8"))["state"] == "done"
    finally:
        conn.close()

def test_real_runners_retry_resets_only_failed_stages(tmp_path, monkeypatch):
    cfg, database, conn = _setup(tmp_path)
    _mark_coarse_done(cfg, conn, JOB_ID, "prep")
    _mark_coarse_done(cfg, conn, JOB_ID, "transcribe")
    try:
        _patch_factories(monkeypatch)
        _fake_managed(monkeypatch, [])
        _write_selection(cfg, [False])
        spies = {}
        paths = _patch_all_stages(monkeypatch, spies)

        # Pass 1: prep/transcribe skipped (done), analyze fails.
        _stage_spy(
            monkeypatch,
            paths["analyze"],
            spies["analyze"],
            exc=RuntimeError("analyze boom"),
        )
        with pytest.raises(worker.WorkerError):
            worker.run_job(JOB_ID, cfg, conn, ws.STAGE_RUNNERS)
        assert spies["prep"] == []
        assert spies["transcribe"] == []
        assert len(spies["analyze"]) == 1
        failed = conn.execute(
            "SELECT state FROM stages WHERE job_id = ? AND name = 'shots'",
            (JOB_ID,),
        ).fetchone()
        assert failed["state"] == "failed"

        # Pass 2 (retry): only the failed fine stage is reset; done stages
        # (prep/transcribe) are skipped, not re-run.
        analyze_ok = []

        def analyze_spy(*args, **kwargs):
            analyze_ok.append((args, kwargs))
            _mark_coarse_done(cfg, conn, JOB_ID, "analyze")

        monkeypatch.setattr("clipper.stages.analyze.run_analysis", analyze_spy)
        report = worker.run_job(JOB_ID, cfg, conn, ws.STAGE_RUNNERS, retry=True)

        assert report.state == "done"
        assert spies["prep"] == []
        assert spies["transcribe"] == []
        assert len(analyze_ok) == 1
        assert spies["render"] and len(spies["render"]) == 1
        # done fine stages were never re-opened by the reset
        ingest_row = conn.execute(
            "SELECT state FROM stages WHERE job_id = ? AND name = 'ingest'",
            (JOB_ID,),
        ).fetchone()
        assert ingest_row["state"] == "done"
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Runner hygiene (task card): no self-lock, no second worker, no
# subprocess/scripts — and the CLI default is the real mapping while the
# M8b fake-injection seam keeps working.
# ---------------------------------------------------------------------------

def test_runners_never_acquire_worker_lock_or_start_a_second_job(
    tmp_path, monkeypatch
):
    def explode(*args, **kwargs):
        raise AssertionError("runners must not touch the worker lock")

    monkeypatch.setattr(worker, "worker_lock", explode)
    cfg, database, conn = _setup(tmp_path)
    try:
        _patch_factories(monkeypatch)
        calls = []
        _stage_spy(monkeypatch, "clipper.stages.candidates.run_candidates", calls)
        ws.STAGE_RUNNERS["candidates"].run(JOB_ID, cfg, conn)
        assert len(calls) == 1
    finally:
        conn.close()

def test_runner_source_is_lock_subprocess_and_spawn_free():
    source = Path(ws.__file__).resolve().read_text(encoding="utf-8")
    for token in (
        "fcntl",
        "worker_lock",
        "subprocess",
        "multiprocessing",
        "run_job(",
    ):
        assert token not in source, token

def test_cli_run_uses_real_runners_by_default(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import clipper.cli as cli

    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(tmp_path / "data")))
    database = Database(cfg)
    database.create_job(JOB_ID, SOURCE, CLIENT_NAME, NICHE)
    conn = database.get_connection()
    try:
        _patch_factories(monkeypatch)
        _fake_managed(monkeypatch, [])
        _write_selection(cfg, [False])
        spies = {}
        _patch_all_stages(monkeypatch, spies)
        conn.close()

        result = CliRunner().invoke(cli.app, ["run", JOB_ID])
        assert result.exit_code == 0, result.output
        assert "state: done" in result.output
        for name in spies:
            assert len(spies[name]) == 1, name
    finally:
        conn.close()

def test_cli_run_fake_runner_injection_still_works(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import clipper.cli as cli

    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(tmp_path / "data")))
    database = Database(cfg)
    database.create_job(JOB_ID, SOURCE, CLIENT_NAME, NICHE)
    conn = database.get_connection()
    conn.close()

    order = []

    class FakeRunner:
        def run(self, job_id, cfg_, conn_):
            order.append(job_id)

    fake_map = {name: FakeRunner() for name in (
        "prep", "transcribe", "analyze", "candidates", "scoring",
        "selection", "clip_plans", "layout", "render",
    )}
    monkeypatch.setattr(cli, "_stage_runners", lambda cfg_: fake_map)

    result = CliRunner().invoke(cli.app, ["run", JOB_ID])
    assert result.exit_code == 0, result.output
    assert order == [JOB_ID] * 9
