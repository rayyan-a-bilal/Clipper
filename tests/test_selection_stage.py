"""
M5C-4 selection-stage tests — CPU only, temp DB + synthetic artifacts.

Covers: ``selection.json`` + stage row + job state; a decision record for
EVERY candidate (selected / gate-failed / cut-rejected); refined
``(start, end)`` persisted for selected candidates and matching
``refine_cut`` exactly; CutError backfill reaching target; zero-selected
valid success; skip on second run (byte-identical artifact); clean
failures (missing input per file with the correct milestone hint,
schema_version mismatch, missing job) marking stage+job failed; and that
the stage never touches the LLM/Ollama runtime.
"""

import inspect
import json

import pytest

from clipper.config import Config, PathsConfig
from clipper.cuts import refine_cut
from clipper.db import Database
from clipper.models import (
    Candidate,
    CandidatesArtifact,
    ProbeResult,
    ScoredClip,
    ScoresArtifact,
    Sentence,
    SentencesArtifact,
    ShotsArtifact,
    SilenceArtifact,
    Word,
    WordsArtifact,
)
from clipper.stages import StageError
from clipper.stages.select import run_selection_stage

JOB_ID = "job-m5c4"

_N_SENTENCES = 30  # 30 x 5s sentences over [0, 150]
_SENTENCES = [
    Sentence(
        start=5.0 * i,
        end=5.0 * i + 4.0,
        text=f"Sentence {i}",
        word_idx=(2 * i, 2 * i + 1),
    )
    for i in range(_N_SENTENCES)
]

_WORDS = [
    word
    for i in range(_N_SENTENCES)
    for word in (
        Word(start=5.0 * i, end=5.0 * i + 2.0, text="w", prob=1.0),
        Word(start=5.0 * i + 2.0, end=5.0 * i + 4.0, text="w", prob=1.0),
    )
]


def _probe(duration_s: float) -> ProbeResult:
    return ProbeResult(
        duration_s=duration_s,
        width=1920,
        height=1080,
        fps=30.0,
        vcodec="av1",
        acodec="opus",
        has_audio=True,
        size_bytes=1000,
        schema_version=1,
    )


def _candidates(n: int) -> list[Candidate]:
    """n non-overlapping 6-sentence (29s) windows, spaced 35s apart."""
    out = []
    for k in range(n):
        i0 = 7 * k
        out.append(
            Candidate(
                cand_id=f"{JOB_ID}-c{i0:04d}-{i0 + 5:04d}",
                start=5.0 * i0,
                end=5.0 * (i0 + 5) + 4.0,
                sentence_range=(i0, i0 + 5),
                cheap_score=0.6,
            )
        )
    return out


def _good_score(cand: Candidate) -> ScoredClip:
    return ScoredClip(
        cand_id=cand.cand_id,
        hook=9,
        payoff=9,
        self_contained=9,
        emotion=9,
        clarity=9,
        suggested_in=cand.start + 1.0,
        suggested_out=cand.end - 1.0,
        reject_reason=None,
    )


def _setup(tmp_path, n: int, score_for=None) -> Config:
    """Temp data dir + DB job + ALL required artifacts written to disk.

    ``score_for`` optionally overrides the score per candidate (default:
    all pass the gate and refine cleanly).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(data_dir)))
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")

    candidates = _candidates(n)
    scores = [
        (score_for or _good_score)(c) for c in candidates
    ]
    duration = candidates[-1].end + 10.0 if candidates else 160.0

    artifacts = {
        "probe.json": _probe(duration),
        "sentences.json": SentencesArtifact(sentences=_SENTENCES),
        "words.json": WordsArtifact(words=_WORDS),
        "silence.json": SilenceArtifact(regions=[]),
        "shots.json": ShotsArtifact(shots=[]),
        "candidates.json": CandidatesArtifact(candidates=candidates),
        "scores.json": ScoresArtifact(model="testmodel", scores=scores),
    }
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        (job_dir / name).write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return cfg


def _run(cfg: Config) -> dict:
    conn = Database(cfg).get_connection()
    try:
        return run_selection_stage(JOB_ID, cfg, conn)
    finally:
        conn.close()


def _db_state(cfg: Config) -> tuple[str, dict | None]:
    """(job state, selection stage row as dict or None if absent)."""
    conn = Database(cfg).get_connection()
    try:
        job = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,)
        ).fetchone()
        stage = conn.execute(
            """
            SELECT state, artifact_path, error
            FROM stages WHERE job_id = ? AND name = 'selection'
            """,
            (JOB_ID,),
        ).fetchone()
        return job[0], dict(stage) if stage else None
    finally:
        conn.close()


def _decisions(cfg: Config) -> list[dict]:
    path = (
        cfg.resolved_data_dir / "jobs" / JOB_ID / "selection.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["decisions"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


# ---------------------------------------------------------------------------
# Happy path: artifact + stage row + decisions for ALL candidates
# ---------------------------------------------------------------------------
def test_selection_artifact_stage_row_and_all_decisions(tmp_path):
    n = 3
    cfg = _setup(tmp_path, n)
    summary = _run(cfg)

    candidates = _candidates(n)
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    artifact_path = job_dir / "selection.json"
    assert artifact_path.exists()

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    decisions = data["decisions"]
    # A decision record for EVERY candidate, in artifact order.
    assert [d["cand_id"] for d in decisions] == [c.cand_id for c in candidates]

    # All pass the gate and refine cleanly -> all selected.
    assert all(d["selected"] for d in decisions)
    assert all(d["eligible"] for d in decisions)
    assert all(d["gate_reasons"] == [] for d in decisions)
    assert all(d["cut_reject_reason"] is None for d in decisions)
    assert all(d["suppressed_by"] is None for d in decisions)
    assert all(
        d["refined_start"] is not None and d["refined_end"] is not None
        for d in decisions
    )

    # Refined cut persisted on the decision EXACTLY matches refine_cut.
    sentences = _SENTENCES
    words = _WORDS
    for cand, dec in zip(candidates, decisions, strict=True):
        expected = refine_cut(
            cand,
            _good_score(cand),
            sentences,
            words,
            [],
            [],
            _probe(candidates[-1].end + 10.0).duration_s,
        )
        assert (dec["refined_start"], dec["refined_end"]) == expected

    # Stage + job bookkeeping: stage done with artifact_path, job back to queued.
    assert summary["selected"] == n
    assert summary["eligible"] == n
    assert summary["gate_failures"] == 0
    assert summary["cut_failures"] == 0
    assert summary["selected_ids"] == [c.cand_id for c in candidates]
    assert summary["skipped"] is False
    assert summary["artifact"] == str(artifact_path)

    job_state, stage_state = _db_state(cfg)
    assert job_state == "queued"
    assert stage_state["state"] == "done"
    assert stage_state["artifact_path"] == str(artifact_path)
    assert stage_state["error"] is None


# ---------------------------------------------------------------------------
# Backfill: CutError on the top-ranked candidate, target still reached
# ---------------------------------------------------------------------------
def test_backfill_after_cut_error_reaches_target(tmp_path):
    n = 4  # target = 3 -> 1 CutError + backfill must still yield 3
    candidates = _candidates(n)
    top = candidates[0]

    def score_for(cand: Candidate) -> ScoredClip:
        if cand.cand_id == top.cand_id:
            # Inverted suggestion -> refine_cut raises CutError.
            return ScoredClip(
                cand_id=cand.cand_id,
                hook=9,
                payoff=9,
                self_contained=9,
                emotion=9,
                clarity=9,
                suggested_in=cand.end,
                suggested_out=cand.start,
                reject_reason=None,
            )
        return _good_score(cand)

    cfg = _setup(tmp_path, n, score_for=score_for)
    summary = _run(cfg)

    assert summary["selected"] == 3
    assert summary["cut_failures"] == 1
    assert summary["selected_ids"] == [c.cand_id for c in candidates[1:]]

    decisions = {d["cand_id"]: d for d in _decisions(cfg)}
    assert decisions[top.cand_id]["eligible"] is True
    assert decisions[top.cand_id]["selected"] is False
    assert decisions[top.cand_id]["cut_reject_reason"]
    assert decisions[top.cand_id]["refined_start"] is None
    assert decisions[top.cand_id]["refined_end"] is None
    for c in candidates[1:]:
        d = decisions[c.cand_id]
        assert d["selected"] is True
        assert d["cut_reject_reason"] is None
        assert d["refined_start"] is not None

    job_state, stage_state = _db_state(cfg)
    assert job_state == "queued"
    assert stage_state["state"] == "done"


# ---------------------------------------------------------------------------
# Zero selected is a VALID success
# ---------------------------------------------------------------------------
def test_zero_selected_is_valid_success(tmp_path):
    n = 3

    def failing_score(cand: Candidate) -> ScoredClip:
        # All axes below any gate threshold -> gate failures, zero eligible.
        return ScoredClip(
            cand_id=cand.cand_id,
            hook=1,
            payoff=1,
            self_contained=1,
            emotion=1,
            clarity=1,
            suggested_in=cand.start + 1.0,
            suggested_out=cand.end - 1.0,
            reject_reason=None,
        )

    cfg = _setup(tmp_path, n, score_for=failing_score)
    summary = _run(cfg)

    assert summary["selected"] == 0
    assert summary["eligible"] == 0
    assert summary["gate_failures"] == n
    assert summary["selected_ids"] == []

    decisions = _decisions(cfg)
    assert len(decisions) == n
    assert all(d["selected"] is False for d in decisions)
    assert all(d["eligible"] is False for d in decisions)
    assert all(d["gate_reasons"] for d in decisions)
    assert all(
        d["refined_start"] is None and d["refined_end"] is None
        for d in decisions
    )

    job_state, stage_state = _db_state(cfg)
    assert job_state == "queued"  # valid success, NOT failed
    assert stage_state["state"] == "done"


# ---------------------------------------------------------------------------
# Skip on second run
# ---------------------------------------------------------------------------
def test_second_run_skips(tmp_path):
    cfg = _setup(tmp_path, 3)
    first = _run(cfg)
    first_bytes = (
        cfg.resolved_data_dir / "jobs" / JOB_ID / "selection.json"
    ).read_bytes()

    second = _run(cfg)
    assert second["skipped"] is True
    first.pop("skipped")
    second.pop("skipped")
    assert second == first  # identical summary (except the flag itself)

    job_state, stage_state = _db_state(cfg)
    assert job_state == "queued"
    assert stage_state["state"] == "done"
    # Artifact untouched by the skipped run.
    assert (
        cfg.resolved_data_dir / "jobs" / JOB_ID / "selection.json"
    ).read_bytes() == first_bytes


# ---------------------------------------------------------------------------
# Clean failures: missing inputs (with milestone hints), schema, job
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("missing", "hint"),
    [
        ("candidates.json", "M4b candidates stage"),
        ("scores.json", "M5b scoring stage"),
        ("sentences.json", "M2b transcribe stage"),
        ("words.json", "M2b transcribe stage"),
        ("silence.json", "M3b analysis stage"),
        ("shots.json", "M3b analysis stage"),
        ("probe.json", "M1 ingest stage"),
    ],
)
def test_missing_input_fails_cleanly_with_hint(tmp_path, missing, hint):
    cfg = _setup(tmp_path, 3)
    (cfg.resolved_data_dir / "jobs" / JOB_ID / missing).unlink()

    conn = Database(cfg).get_connection()
    with pytest.raises(StageError) as exc:
        run_selection_stage(JOB_ID, cfg, conn)
    conn.close()

    message = str(exc.value)
    assert missing in message
    assert hint in message

    job_state, stage_state = _db_state(cfg)
    assert job_state == "failed"
    assert stage_state["state"] == "failed"
    assert stage_state["artifact_path"] is None
    assert missing in stage_state["error"]


def test_schema_version_mismatch_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, 3)
    cands_path = cfg.resolved_data_dir / "jobs" / JOB_ID / "candidates.json"
    data = json.loads(cands_path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    cands_path.write_text(json.dumps(data), encoding="utf-8")

    conn = Database(cfg).get_connection()
    with pytest.raises(StageError) as exc:
        run_selection_stage(JOB_ID, cfg, conn)
    conn.close()

    message = str(exc.value)
    assert "candidates.json" in message
    assert "schema_version" in message
    assert "M4b candidates stage" in message

    job_state, stage_state = _db_state(cfg)
    assert job_state == "failed"
    assert stage_state["state"] == "failed"


def test_missing_job_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, 3)
    conn = Database(cfg).get_connection()
    with pytest.raises(StageError, match="not found"):
        run_selection_stage("no-such-job", cfg, conn)
    conn.close()
    # Unknown job -> no selection stage row created anywhere; setup job untouched.
    conn = Database(cfg).get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM stages WHERE name = 'selection'"
    ).fetchone()
    conn.close()
    assert count[0] == 0
    assert _db_state(cfg) == ("queued", None)


# ---------------------------------------------------------------------------
# NO LLM / NO Ollama anywhere in this stage
# ---------------------------------------------------------------------------
def test_stage_never_touches_llm_or_ollama(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError(
            "the selection stage must never touch the LLM/Ollama runtime"
        )

    import clipper.adapters.llm as llm_mod
    import clipper.llm_runtime as llm_rt
    import clipper.stages.select as sel

    monkeypatch.setattr(llm_mod, "OllamaLLMClient", boom)
    monkeypatch.setattr(llm_rt, "managed_ollama", boom)

    # The stage module has no LLM surface at all, and the public call
    # signature takes no client parameter.
    assert not hasattr(sel, "LLMClient")
    assert "client" not in inspect.signature(run_selection_stage).parameters

    cfg = _setup(tmp_path, 3)
    summary = _run(cfg)  # would raise AssertionError if LLM were touched
    assert summary["selected"] == 3
