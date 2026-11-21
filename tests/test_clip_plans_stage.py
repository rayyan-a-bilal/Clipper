"""
M5C-5 clip_plans-stage tests — CPU only, MockLLM + temp DB.

Covers (stage level): zero selected -> empty artifact, NO LLM call, job
succeeds; valid metadata -> ``clip_plans.json`` + ``clips`` rows;
metadata LLMError / timeout / invalid-schema -> deterministic fallback and
the job STILL succeeds; DRAFT plans are ``center_crop`` /
``crop_path=[]`` / ``pending_qc``; ``clip_id`` format + uniqueness + cut
reuse; second run skips with a byte-identical artifact; missing
``selection.json`` fails cleanly; the metadata request contains ONLY the
selected clips' transcripts (never the full video); the client's
``num_predict`` is capped at 512 for the metadata call;
``OllamaLLMClient`` passes ``httpx`` timeout on every request.
"""

import json
import re
from pathlib import Path

import httpx
import pytest

from clipper.adapters.llm import LLMError, MockLLM, OllamaLLMClient
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.metadata import fallback_metadata
from clipper.models import (
    Candidate,
    CandidatesArtifact,
    ClipMetadata,
    Layout,
    MetadataResponse,
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
from clipper.stages.select import (
    METADATA_NUM_PREDICT_CAP,
    run_clip_plans_stage,
    run_selection_stage,
)

JOB_ID = "job-m5c5"

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
        width=1920, height=1080, fps=30.0,
        vcodec="av1", acodec="opus",
        has_audio=True, size_bytes=1000,
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
        hook=9, payoff=9, self_contained=9, emotion=9, clarity=9,
        suggested_in=cand.start + 1.0, suggested_out=cand.end - 1.0,
        reject_reason=None,
    )


def _failing_score(cand: Candidate) -> ScoredClip:
    """All axes below any gate threshold -> zero eligible candidates."""
    return ScoredClip(
        cand_id=cand.cand_id,
        hook=1, payoff=1, self_contained=1, emotion=1, clarity=1,
        suggested_in=cand.start + 1.0, suggested_out=cand.end - 1.0,
        reject_reason=None,
    )


def _setup(tmp_path, n: int, score_for=_good_score) -> Config:
    """Temp data dir + DB job + all required artifacts on disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(data_dir)))
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/fake-source.mp4", "client", "saas")

    candidates = _candidates(n)
    scores = [score_for(c) for c in candidates]
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
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8",
        )
    return cfg


def _job_dir(cfg: Config) -> Path:
    return cfg.resolved_data_dir / "jobs" / JOB_ID


def _run_selection(cfg: Config) -> dict:
    conn = Database(cfg).get_connection()
    try:
        return run_selection_stage(JOB_ID, cfg, conn)
    finally:
        conn.close()


def _run_plans(cfg: Config, client) -> dict:
    conn = Database(cfg).get_connection()
    try:
        return run_clip_plans_stage(JOB_ID, cfg, conn, client)
    finally:
        conn.close()


def _db_state(cfg: Config):
    """(job state, clip_plans stage row dict or None)."""
    conn = Database(cfg).get_connection()
    try:
        job = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (JOB_ID,),
        ).fetchone()
        stage = conn.execute(
            """
            SELECT state, artifact_path, error
            FROM stages WHERE job_id = ? AND name = 'clip_plans'
            """,
            (JOB_ID,),
        ).fetchone()
        return job[0], dict(stage) if stage else None
    finally:
        conn.close()


def _selected_decisions(cfg: Config) -> list[dict]:
    data = json.loads((_job_dir(cfg) / "selection.json").read_text("utf-8"))
    return [d for d in data["decisions"] if d["selected"]]


def _plans(cfg: Config) -> list[dict]:
    data = json.loads((_job_dir(cfg) / "clip_plans.json").read_text("utf-8"))
    return data["plans"]


def _sentences_json(cfg: Config) -> list[dict]:
    data = json.loads((_job_dir(cfg) / "sentences.json").read_text("utf-8"))
    return data["sentences"]


def _valid_items(decisions: list[dict]) -> list[ClipMetadata]:
    """Validate-passing canned LLM metadata, one item per selected clip."""
    return [
        ClipMetadata(
            cand_id=d["cand_id"],
            title=f"LLM title for {d['cand_id']}",
            hook_overlay="a strong hook now",
            description="What the clip actually says, in one sentence.",
            hashtags=["#saas", "#demo", "#clipper"],
        )
        for d in decisions
    ]


def _clip_transcript(sentences: list[dict], start: float, end: float) -> str:
    return " ".join(
        s["text"] for s in sentences if s["start"] < end and s["end"] > start
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


# ---------------------------------------------------------------------------
# Zero selected: empty artifact, NO LLM call, job succeeds
# ---------------------------------------------------------------------------
def test_zero_selected_empty_artifact_no_llm_call(tmp_path):
    cfg = _setup(tmp_path, 3, score_for=_failing_score)
    _run_selection(cfg)

    client = MockLLM()  # empty queue: ANY call would fail the attempt
    summary = _run_plans(cfg, client)

    assert client.calls == []  # LLM never invoked
    assert summary["plans"] == 0
    assert summary["metadata_source"] is None
    assert summary["skipped"] is False

    data = json.loads((_job_dir(cfg) / "clip_plans.json").read_text("utf-8"))
    assert data == {"schema_version": 1, "plans": []}

    job_state, stage = _db_state(cfg)
    assert job_state == "queued"  # VALID success, not failed
    assert stage["state"] == "done"
    assert stage["artifact_path"].endswith("clip_plans.json")
    assert stage["error"] is None


# ---------------------------------------------------------------------------
# Valid metadata: clip_plans.json + clips rows
# ---------------------------------------------------------------------------
def test_valid_metadata_writes_plans_and_clips_rows(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)
    assert len(decisions) == 3

    client = MockLLM()
    items = _valid_items(decisions)
    client.enqueue(MetadataResponse(items=items))
    summary = _run_plans(cfg, client)

    assert summary["plans"] == 3
    assert summary["metadata_source"] == "llm"
    assert summary["fallback_reason"] is None
    assert len(client.calls) == 1  # ONE metadata call for the whole batch
    assert client.calls[0]["schema"] is MetadataResponse

    plans = _plans(cfg)
    assert len(plans) == 3
    for d in decisions:
        expected = next(m for m in items if m.cand_id == d["cand_id"])
        p = next(p for p in plans if p["title"] == expected.title)
        assert p["start"] == d["refined_start"]
        assert p["end"] == d["refined_end"]
        assert p["weighted_score"] == d["weighted"]
        assert p["hashtags"] == expected.hashtags

    # One clips-table row per plan, upserted with plan_json + status.
    conn = Database(cfg).get_connection()
    rows = conn.execute(
        "SELECT clip_id, job_id, plan_json, status FROM clips"
    ).fetchall()
    conn.close()
    assert len(rows) == 3
    for clip_id, job_id, plan_json, status in rows:
        assert job_id == JOB_ID
        assert status == "pending_qc"
        assert json.loads(plan_json)["clip_id"] == clip_id

    job_state, stage = _db_state(cfg)
    assert job_state == "queued"
    assert stage["state"] == "done"


# ---------------------------------------------------------------------------
# Metadata failure (LLMError / timeout / invalid schema) => fallback,
# job STILL succeeds — metadata must NEVER fail the job
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exception", [
    LLMError("model returned invalid JSON"),
    LLMError("transport error: ConnectTimeout('timed out')"),
])
def test_metadata_llmerror_falls_back_job_succeeds(tmp_path, exception):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    client = MockLLM()
    for _ in range(3):  # max_retries=2 -> 3 attempts, all fail
        client.enqueue(exception)
    summary = _run_plans(cfg, client)

    assert len(client.calls) == 3  # retries exhausted
    assert summary["plans"] == 3
    assert summary["metadata_source"] == "fallback"
    assert summary["fallback_reason"]

    # Fallback metadata is exactly the deterministic M5C-3 fallback.
    sentences = _sentences_json(cfg)
    plans = _plans(cfg)
    for p in plans:
        d = next(
            d for d in decisions
            if p["start"] == d["refined_start"] and p["end"] == d["refined_end"]
        )
        expected = fallback_metadata(
            d["cand_id"],
            _clip_transcript(
                sentences,
                d["refined_start"],
                d["refined_end"],
            ),
            "saas",
        )
        assert p["title"] == expected.title
        assert p["hook_overlay"] == expected.hook_overlay
        assert p["description"] == expected.description
        assert p["hashtags"] == expected.hashtags

    job_state, stage = _db_state(cfg)
    assert job_state == "queued"  # metadata failure NEVER fails the job
    assert stage["state"] == "done"
    assert stage["error"] is None


def test_invalid_schema_response_falls_back_job_succeeds(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    client = MockLLM()
    for _ in range(3):
        # 1st attempt: response fails the MetadataResponse schema itself.
        client.enqueue({"bogus": "not-a-MetadataResponse"})
        # 2nd/3rd: valid schema but violates metadata limits (ValueError
        # from validate_metadata) — still a retryable failure.
        client.enqueue(MetadataResponse(items=[
            ClipMetadata(
                cand_id=d["cand_id"],
                title="x" * 120,  # over the 80-char prompt limit
                hook_overlay="a strong hook now",
                description="fine",
                hashtags=["#only-one"],  # needs 3-6
            )
            for d in decisions
        ]))
    summary = _run_plans(cfg, client)

    assert len(client.calls) == 3
    assert summary["metadata_source"] == "fallback"
    assert summary["plans"] == 3

    job_state, _stage = _db_state(cfg)
    assert job_state == "queued"


def test_client_none_uses_fallback_without_error(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)

    summary = _run_plans(cfg, None)

    assert summary["plans"] == 3
    assert summary["metadata_source"] == "fallback"
    assert summary["fallback_reason"] == "no LLM client"
    job_state, _ = _db_state(cfg)
    assert job_state == "queued"


# ---------------------------------------------------------------------------
# Draft plans: center_crop / [] / pending_qc; clip_id format + uniqueness
# ---------------------------------------------------------------------------
def test_draft_plans_shape_and_clip_ids(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    client = MockLLM()
    client.enqueue(MetadataResponse(items=_valid_items(decisions)))
    _run_plans(cfg, client)

    plans = _plans(cfg)
    assert len(plans) == 3

    clip_ids = [p["clip_id"] for p in plans]
    assert len(set(clip_ids)) == len(clip_ids)  # unique
    for p in plans:
        assert re.fullmatch(rf"{re.escape(JOB_ID)}-clip\d{{2}}", p["clip_id"])
        assert p["job_id"] == JOB_ID
        assert p["layout"] == Layout.center_crop.value
        assert p["crop_path"] == []
        assert p["status"] == "pending_qc"
    # 1-based, contiguous, rank order (here: equal weighted/cheap -> start ASC)
    assert clip_ids == [f"{JOB_ID}-clip{i:02d}" for i in (1, 2, 3)]

    # start/end come VERBATIM from the selection stage's refined cut
    for p in plans:
        d = next(
            d for d in decisions
            if p["start"] == d["refined_start"] and p["end"] == d["refined_end"]
        )
        assert p["weighted_score"] == d["weighted"]


# ---------------------------------------------------------------------------
# Second run skips — artifact byte-identical, LLM not re-queried
# ---------------------------------------------------------------------------
def test_second_run_skips_byte_identical(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    client = MockLLM()
    client.enqueue(MetadataResponse(items=_valid_items(decisions)))
    first = _run_plans(cfg, client)
    first_bytes = (_job_dir(cfg) / "clip_plans.json").read_bytes()

    # Second run: even a BROKEN client must not be used (stage is skipped).
    broken = MockLLM()  # empty queue
    second = _run_plans(cfg, broken)

    assert broken.calls == []
    assert second["skipped"] is True
    assert second["plans"] == first["plans"]
    for key in ("skipped", "metadata_source", "fallback_reason"):
        first.pop(key)
        second.pop(key)
    assert second == first
    assert (_job_dir(cfg) / "clip_plans.json").read_bytes() == first_bytes

    job_state, stage = _db_state(cfg)
    assert job_state == "queued"
    assert stage["state"] == "done"


# ---------------------------------------------------------------------------
# Clean failures (NON-metadata): missing selection.json, missing job
# ---------------------------------------------------------------------------
def test_missing_selection_json_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, 3)  # does NOT run the selection stage

    conn = Database(cfg).get_connection()
    with pytest.raises(StageError) as exc:
        run_clip_plans_stage(JOB_ID, cfg, conn, None)
    conn.close()

    message = str(exc.value)
    assert "selection.json" in message
    assert "M5C-4 selection stage" in message
    assert "scripts/select_job.py" in message

    job_state, stage = _db_state(cfg)
    assert job_state == "failed"
    assert stage["state"] == "failed"
    assert "selection.json" in stage["error"]


def test_selection_schema_mismatch_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    sel_path = _job_dir(cfg) / "selection.json"
    data = json.loads(sel_path.read_text("utf-8"))
    data["schema_version"] = 2
    sel_path.write_text(json.dumps(data), encoding="utf-8")

    conn = Database(cfg).get_connection()
    with pytest.raises(StageError, match="schema_version"):
        run_clip_plans_stage(JOB_ID, cfg, conn, None)
    conn.close()

    job_state, _stage = _db_state(cfg)
    assert job_state == "failed"


def test_missing_job_fails_cleanly(tmp_path):
    cfg = _setup(tmp_path, 3)
    conn = Database(cfg).get_connection()
    with pytest.raises(StageError, match="not found"):
        run_clip_plans_stage("no-such-job", cfg, conn, None)
    conn.close()
    # Unknown job -> no clip_plans stage row created anywhere
    conn = Database(cfg).get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM stages WHERE name = 'clip_plans'"
    ).fetchone()
    conn.close()
    assert count[0] == 0


# ---------------------------------------------------------------------------
# Prompt scope: ONLY selected clip transcripts — never the full video
# ---------------------------------------------------------------------------
def test_metadata_request_contains_only_selected_transcripts(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    captured: list[tuple[str, str]] = []

    class RecordingClient:
        def generate_json(self, system, user, schema):
            captured.append((system, user))
            return MetadataResponse(items=_valid_items(decisions))

    _run_plans(cfg, RecordingClient())

    assert len(captured) == 1
    system, user = captured[0]
    from clipper.prompts import METADATA_SYSTEM

    assert system == METADATA_SYSTEM

    payload = json.loads(user)
    assert set(payload) == {"clips"}
    assert [c["cand_id"] for c in payload["clips"]] == [
        d["cand_id"] for d in decisions
    ]
    for c in payload["clips"]:
        assert set(c) == {"cand_id", "transcript", "niche", "duration_s"}
        assert c["niche"] == "saas"
        assert c["duration_s"] > 15.0 and c["duration_s"] <= 75.0

    # In-window sentences are present...
    for marker in ("Sentence 0", "Sentence 7", "Sentence 14"):
        assert marker in user
    # ...and sentences OUTSIDE every selected clip window are NOT.
    for marker in (
        "Sentence 6",
        "Sentence 13",
        "Sentence 20",
        "Sentence 25",
        "Sentence 29",
    ):
        assert marker not in user


# ---------------------------------------------------------------------------
# OllamaLLMClient: httpx timeout on every request; num_predict capped 512
# ---------------------------------------------------------------------------
def _capture_ollama_client(cfg, decisions, sink):
    """Real OllamaLLMClient with a fake httpx transport (zero network)."""

    class FakeHttpClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def post(self, url, json=None):
            import json as json_module

            sink.append({"url": url, "json": json, "timeout": self.timeout})
            body = MetadataResponse(items=_valid_items(decisions)).model_dump()
            resp = httpx.Response(
                200,
                json={"message": {"role": "assistant",
                                 "content": json_module.dumps(body)}},
            )
            resp.request = httpx.Request("POST", url)
            return resp

        def close(self):
            pass

    client = OllamaLLMClient(
        cfg,
        base_url="http://127.0.0.1:11435",
        client_factory=lambda timeout_s: FakeHttpClient(timeout_s),
    )
    return client


def test_ollama_client_passes_config_timeout_on_every_request(tmp_path):
    cfg = _setup(tmp_path, 3)
    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    sink: list = []
    client = _capture_ollama_client(cfg, decisions, sink)

    # The stage creates a fresh httpx client PER request (retry-safe);
    # each must carry cfg.llm.timeout_s.
    _run_plans(cfg, client)

    assert len(sink) == 1
    for request in sink:
        assert request["timeout"] == cfg.llm.timeout_s
        assert request["url"] == "http://127.0.0.1:11435/api/chat"


def test_metadata_num_predict_capped_at_512(tmp_path):
    assert METADATA_NUM_PREDICT_CAP == 512
    cfg = _setup(tmp_path, 3)
    assert cfg.llm.num_predict == 1024  # config default unchanged

    _run_selection(cfg)
    decisions = _selected_decisions(cfg)

    sink: list = []
    client = _capture_ollama_client(cfg, decisions, sink)
    _run_plans(cfg, client)

    body = sink[0]["json"]
    options = body["options"]
    assert options["num_predict"] == METADATA_NUM_PREDICT_CAP
    assert options["num_ctx"] == cfg.llm.num_ctx
    assert options["num_batch"] == cfg.llm.num_batch
    assert options["temperature"] == cfg.llm.temperature
