"""
M5b scoring stage tests — CPU only, temp DB + synthetic artifacts + MockLLM.

Covers: artifact + stage row + candidates.score_json updates, skip on second
run, per-batch retry -> rejection (job still succeeds), all-batches-failed
(policy: artifact written, job succeeds, zero usable scores), clean failures
(missing candidates.json / sentences.json / probe.json, schema_version
mismatch, missing job), zero-candidate success, and byte-identical
determinism across two fresh runs with the same mock.
"""

import json

import pytest

from clipper.adapters.llm import LLMError, MockLLM
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.models import (
    Candidate,
    CandidatesArtifact,
    ProbeResult,
    ScoredClip,
    ScoringBatch,
    ScoringResponse,
    Sentence,
    SentencesArtifact,
)
from clipper.scoring import build_context, pack_batches
from clipper.stages import StageError
from clipper.stages.score import run_scoring

JOB_ID = "job-m5b"

_SENTENCES = [
    Sentence(start=5.0 * i, end=5.0 * i + 4.0, text=f"S{i}", word_idx=(i, i))
    for i in range(12)
]

_PROBE = ProbeResult(
    duration_s=60.0,
    width=1920,
    height=1080,
    fps=30.0,
    vcodec="av1",
    acodec="opus",
    has_audio=True,
    size_bytes=1000,
    schema_version=1,
)


def _candidates(n: int = 7) -> list[Candidate]:
    out = []
    for i in range(n):
        out.append(
            Candidate(
                cand_id=f"{JOB_ID}-c{i:04d}-{i + 1:04d}",
                start=5.0 * i,
                end=5.0 * (i + 1) + 4.0,
                sentence_range=(i, i + 1),
                cheap_score=0.5,
            )
        )
    return out


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


def _cfg(tmp_path) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(data_dir=str(data_dir)))


def _valid_response(batch: ScoringBatch) -> ScoringResponse:
    """Deterministic in-bounds scores (mock only — not a judgement)."""
    scores = []
    for j, cand in enumerate(batch.candidates):
        v = 10 - (2 * j) % 11
        scores.append(
            ScoredClip(
                cand_id=cand.cand_id,
                hook=v,
                payoff=v,
                self_contained=v,
                emotion=v,
                clarity=v,
                suggested_in=cand.start + 0.5,
                suggested_out=cand.end - 0.2,
                reject_reason=None,
            )
        )
    return ScoringResponse(scores=scores)


def _expected_batches(cfg: Config, n: int = 7) -> list[ScoringBatch]:
    candidates = _candidates(n)
    contexts = [build_context(c, _SENTENCES, cfg) for c in candidates]
    return pack_batches(contexts, cfg.llm.batch_size, cfg.llm.max_batch_tokens)


def _setup(
    tmp_path, n_candidates: int = 7
) -> Config:
    cfg = _cfg(tmp_path)
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")

    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "probe.json").write_text(
        _PROBE.model_dump_json(indent=2), encoding="utf-8"
    )
    (job_dir / "sentences.json").write_text(
        SentencesArtifact(
            schema_version=1, sentences=_SENTENCES
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    candidates = _candidates(n_candidates)
    (job_dir / "candidates.json").write_text(
        CandidatesArtifact(
            schema_version=1, candidates=candidates
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    # M4b already upserted these rows with score_json = NULL.
    conn = db.get_connection()
    try:
        for cand in candidates:
            conn.execute(
                "INSERT INTO candidates (cand_id, job_id, candidate_json, score_json) "
                "VALUES (?, ?, ?, NULL)",
                (cand.cand_id, JOB_ID, cand.model_dump_json()),
            )
        conn.commit()
    finally:
        conn.close()
    return cfg


def _run(cfg: Config, client: MockLLM) -> dict:
    db = Database(cfg)
    conn = db.get_connection()
    try:
        return run_scoring(JOB_ID, cfg, conn, client)
    finally:
        conn.close()


def _db(cfg: Config) -> Database:
    return Database(cfg)


def _score_rows(cfg: Config) -> list[dict]:
    conn = _db(cfg).get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM candidates WHERE job_id = ? ORDER BY cand_id",
            (JOB_ID,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _stages(cfg: Config) -> dict:
    db = _db(cfg)
    return {s["name"]: s for s in db.get_all_stages_for_job(JOB_ID)}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_scoring_writes_scores_json_and_updates_score_json(tmp_path):
    cfg = _setup(tmp_path)
    batches = _expected_batches(cfg)
    assert len(batches) == 2  # 7 candidates, batch size 5 -> 5+2
    mock = MockLLM()
    mock.enqueue(*(_valid_response(b) for b in batches))

    summary = _run(cfg, mock)

    assert summary["skipped"] is False
    assert summary["batches"] == 2
    assert summary["scored"] == 7
    assert summary["rejected"] == 0
    assert summary["model"] == cfg.llm.model

    # scores.json: schema_version=1, model, ALL candidates in artifact order
    artifact_path = cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json"
    data = json.loads(artifact_path.read_text())
    assert data["schema_version"] == 1
    assert data["model"] == cfg.llm.model
    assert [s["cand_id"] for s in data["scores"]] == [
        c.cand_id for c in _candidates()
    ]

    # stage row
    stages = _stages(cfg)
    assert stages["scoring"]["state"] == "done"
    assert stages["scoring"]["artifact_path"] == str(artifact_path)
    assert stages["scoring"]["error"] is None
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"

    # candidates.score_json updated for every cand_id
    rows = _score_rows(cfg)
    assert len(rows) == 7
    by_id = {r["cand_id"]: r for r in rows}
    for s in data["scores"]:
        assert by_id[s["cand_id"]]["score_json"] == ScoredClip(
            **s
        ).model_dump_json()


def test_second_run_skips_completed_stage(tmp_path):
    cfg = _setup(tmp_path)
    batches = _expected_batches(cfg)
    mock = MockLLM()
    mock.enqueue(*(_valid_response(b) for b in batches))
    _run(cfg, mock)

    artifact_path = cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json"
    before = artifact_path.read_bytes()
    stages_before = _stages(cfg)
    calls_after_first = len(mock.calls)

    summary = _run(cfg, mock)  # empty queue — must not be consulted

    assert summary["skipped"] is True
    assert summary["scored"] == 7
    assert summary["rejected"] == 0
    assert len(mock.calls) == calls_after_first
    assert artifact_path.read_bytes() == before
    assert _stages(cfg) == stages_before
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"


def test_out_of_bounds_suggestion_kept_with_reject_reason(tmp_path):
    cfg = _setup(tmp_path, n_candidates=1)
    batches = _expected_batches(cfg, n=1)
    mock = MockLLM()
    good = _valid_response(batches[0])
    s0 = good.scores[0]
    s0.suggested_in = s0.suggested_in - 100.0  # far outside start-8
    s0.suggested_out = s0.suggested_out + 100.0
    mock.enqueue(good)

    summary = _run(cfg, mock)
    assert summary["rejected"] == 1

    data = json.loads(
        (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").read_text()
    )
    s = data["scores"][0]
    assert s["reject_reason"] == "suggestion_out_of_bounds"
    # scores themselves are KEPT (not zeroed)
    assert s["hook"] != 0


def test_zero_candidates_is_valid_success(tmp_path):
    cfg = _setup(tmp_path, n_candidates=0)
    mock = MockLLM()
    assert mock.pending == 0

    summary = _run(cfg, mock)

    assert summary["skipped"] is False
    assert summary["batches"] == 0
    assert summary["scored"] == 0
    assert mock.calls == []  # no LLM calls at all

    data = json.loads(
        (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").read_text()
    )
    assert data["scores"] == []
    assert _stages(cfg)["scoring"]["state"] == "done"
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"


# ---------------------------------------------------------------------------
# Batch-failure policy: reject the batch, NEVER fail the job
# ---------------------------------------------------------------------------


def test_batch_failure_rejects_only_that_batch(tmp_path):
    cfg = _setup(tmp_path)
    batches = _expected_batches(cfg)
    assert len(batches) == 2
    mock = MockLLM()
    # batch 1: all max_retries+1 attempts fail; batch 2: valid
    for _ in range(cfg.llm.max_retries + 1):
        mock.enqueue(LLMError("boom"))
    mock.enqueue(_valid_response(batches[1]))

    summary = _run(cfg, mock)

    first_ids = [c.cand_id for c in batches[0].candidates]
    second_ids = [c.cand_id for c in batches[1].candidates]

    # Job success
    assert summary["skipped"] is False
    assert summary["scored"] == 7
    assert summary["rejected"] == 5  # batch 1's 5 candidates
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"
    assert _stages(cfg)["scoring"]["state"] == "done"
    # Exactly 3 failing calls for batch 1 (+1 valid for batch 2)
    assert len(mock.calls) == cfg.llm.max_retries + 1 + 1

    data = json.loads(
        (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").read_text()
    )
    ordered = {s["cand_id"]: s for s in data["scores"]}
    assert [s["cand_id"] for s in data["scores"]] == first_ids + second_ids

    for cid in first_ids:
        s = ordered[cid]
        assert s["hook"] == s["payoff"] == s["self_contained"] == 0
        assert s["emotion"] == s["clarity"] == 0
        assert s["reject_reason"].startswith("llm_batch_failed:")
        assert "boom" in s["reject_reason"]
    for cid in second_ids:
        assert ordered[cid]["reject_reason"] is None
        assert ordered[cid]["hook"] > 0

    # score_json rows reflect the same policy
    rows = {r["cand_id"]: r for r in _score_rows(cfg)}
    assert json.loads(rows[first_ids[0]]["score_json"])["reject_reason"].startswith(
        "llm_batch_failed:"
    )


def test_retry_then_success(tmp_path):
    cfg = _setup(tmp_path)
    batches = _expected_batches(cfg)
    mock = MockLLM()
    mock.enqueue(LLMError("transient"), LLMError("transient"))
    mock.enqueue(_valid_response(batches[0]))
    for b in batches[1:]:
        mock.enqueue(_valid_response(b))

    summary = _run(cfg, mock)

    assert summary["rejected"] == 0
    assert summary["scored"] == 7
    assert len(mock.calls) == 2 + len(batches)
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"


def test_all_batches_fail_still_succeeds_with_zero_usable_scores(tmp_path):
    cfg = _setup(tmp_path)
    batches = _expected_batches(cfg)
    mock = MockLLM()
    for _ in batches:
        for _ in range(cfg.llm.max_retries + 1):
            mock.enqueue(LLMError("model down"))

    summary = _run(cfg, mock)

    assert summary["skipped"] is False
    assert summary["scored"] == 7
    assert summary["rejected"] == 7
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"
    assert _stages(cfg)["scoring"]["state"] == "done"

    data = json.loads(
        (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").read_text()
    )
    assert len(data["scores"]) == 7
    for s in data["scores"]:
        for axis in ("hook", "payoff", "self_contained", "emotion", "clarity"):
            assert s[axis] == 0
        assert s["reject_reason"].startswith("llm_batch_failed:")


def test_semantic_validation_failure_is_retryable(tmp_path):
    """A response that fails schema validation is a ValueError -> retried,
    then the batch is rejected (job still succeeds)."""
    cfg = _setup(tmp_path, n_candidates=1)
    mock = MockLLM()
    bad = {"scores": ["this is not a ScoredClip"]}
    for _ in range(cfg.llm.max_retries + 1):
        mock.enqueue(bad)

    summary = _run(cfg, mock)

    assert summary["rejected"] == 1
    assert len(mock.calls) == cfg.llm.max_retries + 1
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"
    data = json.loads(
        (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").read_text()
    )
    assert data["scores"][0]["reject_reason"].startswith("llm_batch_failed:")


# ---------------------------------------------------------------------------
# Clean failures (job must be marked failed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing,hint",
    [
        (
            "candidates.json",
            "M4b candidates stage (scripts/candidates_job.py)",
        ),
        (
            "sentences.json",
            "M2b transcribe stage (scripts/transcribe_job.py)",
        ),
        ("probe.json", "M1 ingest stage (python -m clipper ingest)"),
    ],
    ids=["candidates", "sentences", "probe"],
)
def test_missing_input_fails_cleanly_with_milestone_hint(tmp_path, missing, hint):
    cfg = _setup(tmp_path)
    (cfg.resolved_data_dir / "jobs" / JOB_ID / missing).unlink()
    mock = MockLLM()

    with pytest.raises(StageError) as excinfo:
        _run(cfg, mock)

    message = str(excinfo.value)
    assert missing in message
    assert hint in message
    assert mock.calls == []
    assert _db(cfg).get_job(JOB_ID)["state"] == "failed"
    assert _stages(cfg)["scoring"]["state"] == "failed"
    assert missing in _stages(cfg)["scoring"]["error"]
    assert not (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").exists()


def test_schema_version_mismatch_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path)
    path = cfg.resolved_data_dir / "jobs" / JOB_ID / "candidates.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 2
    path.write_text(json.dumps(data))

    with pytest.raises(StageError, match="schema_version"):
        _run(cfg, MockLLM())

    assert _db(cfg).get_job(JOB_ID)["state"] == "failed"
    assert _stages(cfg)["scoring"]["state"] == "failed"
    assert not (cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json").exists()


def test_missing_job_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path)
    db = _db(cfg)
    with pytest.raises(StageError, match="not found"):
        run_scoring("no-such-job", cfg, db.get_connection(), MockLLM())
    assert db.get_all_stages_for_job("no-such-job") == []


def test_rerun_after_failure_retries(tmp_path):
    cfg = _setup(tmp_path)
    (cfg.resolved_data_dir / "jobs" / JOB_ID / "probe.json").unlink()
    with pytest.raises(StageError):
        _run(cfg, MockLLM())
    assert _db(cfg).get_job(JOB_ID)["state"] == "failed"

    # Restore input -> the failed stage is retried and now succeeds.
    (cfg.resolved_data_dir / "jobs" / JOB_ID / "probe.json").write_text(
        _PROBE.model_dump_json(indent=2), encoding="utf-8"
    )
    batches = _expected_batches(cfg)
    mock = MockLLM()
    mock.enqueue(*(_valid_response(b) for b in batches))
    summary = _run(cfg, mock)
    assert summary["skipped"] is False
    assert summary["scored"] == 7
    assert _db(cfg).get_job(JOB_ID)["state"] == "queued"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_fresh_runs_are_byte_identical_with_same_mock(tmp_path, tmp_path_factory):
    def fresh(root) -> bytes:
        cfg = _setup(root)
        batches = _expected_batches(cfg)
        mock = MockLLM()
        mock.enqueue(*(_valid_response(b) for b in batches))
        summary = _run(cfg, mock)
        assert summary["skipped"] is False
        job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
        return job_dir / "scores.json"

    a_path = fresh(tmp_path)
    root_b = tmp_path_factory.mktemp("fresh_b")
    b_path = fresh(root_b)
    assert a_path.read_bytes() == b_path.read_bytes()


# ---------------------------------------------------------------------------
# Script (scripts/score_job.py)
# ---------------------------------------------------------------------------


def _load_score_job_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "score_job.py"
    spec = importlib.util.spec_from_file_location("clipper_score_job_script", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_job_script_mock_success(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)
    script = _load_score_job_script()

    rc = script.main(JOB_ID, use_mock=True)

    assert rc == 0
    out = capsys.readouterr().out
    assert "job:" in out and JOB_ID in out
    assert "model:" in out
    assert "batches:  2" in out
    assert "scored:   7" in out
    assert "rejected: 0" in out
    assert "top 5:" in out
    assert "wiring test only" in out
    assert str(cfg.resolved_data_dir / "jobs" / JOB_ID / "scores.json") in out


def test_score_job_script_missing_job_exits_1(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    Database(cfg)  # initialize the DB file
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_score_job_script()

    rc = script.main("no-such-job", use_mock=True)
    assert rc == 1
    assert "not found" in capsys.readouterr().out


def test_score_job_script_failure_exits_1(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    (cfg.resolved_data_dir / "jobs" / JOB_ID / "candidates.json").unlink()
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_score_job_script()

    rc = script.main(JOB_ID, use_mock=True)
    assert rc == 1
    out = capsys.readouterr().out
    assert "candidates.json" in out
    assert "scripts/candidates_job.py" in out


def test_score_job_script_mock_scores_derive_from_batch_position(
    tmp_path
):
    """The --mock scorer: position j in a batch -> 10 - (2j)%11 on all axes."""
    cfg = _setup(tmp_path, n_candidates=2)
    script = _load_score_job_script()
    mock = script._MockScorer()
    batch = _expected_batches(cfg, n=2)[0]
    resp = mock.generate_json("sys", batch.model_dump_json(), ScoringResponse)
    values = [10 - (2 * j) % 11 for j in range(len(batch.candidates))]
    for s, v, cand in zip(resp.scores, values, batch.candidates, strict=True):
        assert (s.hook, s.payoff, s.self_contained, s.emotion, s.clarity) == (
            v,) * 5
        assert s.suggested_in == cand.start
        assert s.suggested_out == cand.end
