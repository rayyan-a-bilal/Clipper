"""
M9d — QC selection actions + export tests (CPU only).

Covers the service layer `clipper.qc_actions` and its I/O adapter
`clipper.adapters.qc` per `docs/spec/m9.md` m9-4 / m9-5 (approved halves):

- `force_select`  — m9-4: NO LLM, candidate bounds + existing padding,
  `overrides.effective_cut` validation (15–75s, concrete errors, no
  clamping), deterministic `metadata.fallback_metadata`,
  `clip_plans.json` append + `clips` row upsert, immutable artifacts
- `approve` / `reject` — m9-5: status only (plan + row), nothing else
- `export`      — m9-5 (approved clips, copy-only, NO rerender)
- every mutation under the EXISTING `worker.worker_lock` (contention →
  clean `WorkerError` refusal)
- the CLI commands (`approve` / `reject` / `export JOB` /
  `force-select CAND [--reason]`)

CPU-only: temp data dir, temp DB, fake artifact files. No GPU, no
network, no Ollama, no ffmpeg, no sys.modules pops of parent packages.
"""

import fcntl
import inspect
from ast import parse as ast_parse
from pathlib import Path

import pytest
from typer.testing import CliRunner

import clipper.adapters.qc as qc
import clipper.cli as cli
from clipper import qc_actions
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.metadata import (
    FALLBACK_HOOK_WORDS,
    FALLBACK_TITLE_WORDS,
)
from clipper.models import (
    Candidate,
    ClipPlan,
    ClipPlansArtifact,
    Layout,
    Sentence,
    SentencesArtifact,
)
from clipper.qc_actions import QCActionsError
from clipper.worker import WorkerError, lock_path

JOB_ID = "job-m9d"
CAND_ID = f"{JOB_ID}-cand01"
CLIP01 = f"{JOB_ID}-clip01"
CLIP02 = f"{JOB_ID}-clip02"
CLIP03 = f"{JOB_ID}-clip03"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


def _setup(tmp_path: Path) -> tuple[Config, Database, "object", Path]:
    """cfg + Database + one job (niche 'science') + its job dir + conn."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(paths=PathsConfig(data_dir=str(data_dir)))
    dbobj = Database(cfg)
    assert dbobj.create_job(JOB_ID, "/tmp/m9d-source.mp4", "client", "science")
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    conn = dbobj.get_connection()
    return cfg, dbobj, conn, job_dir


def _candidate(start: float, end: float, cand_id: str = CAND_ID) -> Candidate:
    return Candidate(
        cand_id=cand_id,
        start=start,
        end=end,
        sentence_range=(1, 2),
        cheap_score=5.5,
    )


def _insert_candidate(conn, cand: Candidate, job_id: str = JOB_ID) -> None:
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO candidates (cand_id, job_id, candidate_json, score_json) "
        "VALUES (?, ?, ?, NULL)",
        (cand.cand_id, job_id, cand.model_dump_json()),
    )
    conn.commit()


def _sentence(start: float, end: float, text: str) -> Sentence:
    return Sentence(start=start, end=end, text=text, word_idx=(0, len(text.split())))


# One sentence BEFORE the happy cut window, two INSIDE, one AFTER.
_SENT_BEFORE = _sentence(90.0, 99.0, "zed yank xray")
_SENT_B = _sentence(
    100.0, 110.0, "alpha beta gamma delta epsilon zeta eta theta iota kappa"
)
_SENT_C = _sentence(115.0, 125.0, "lambda mu nu xi omicron pi rho sigma tau")
_SENT_AFTER = _sentence(130.26, 140.0, "quebec oscar")


def _write_sentences(job_dir: Path, sentences: list[Sentence]) -> None:
    artifact = SentencesArtifact(schema_version=1, sentences=sentences)
    (job_dir / "sentences.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _write_plans(job_dir: Path, plans: list[ClipPlan]) -> None:
    qc.write_clip_plans(job_dir, ClipPlansArtifact(schema_version=1, plans=plans))


def _plan(
    clip_id: str,
    status: str = "pending_qc",
    start: float = 200.0,
    end: float = 230.0,
    job_id: str = JOB_ID,
) -> ClipPlan:
    return ClipPlan(
        clip_id=clip_id,
        job_id=job_id,
        start=start,
        end=end,
        layout=Layout.center_crop,
        crop_path=[],
        weighted_score=7.5,
        title="T",
        hook_overlay="H",
        description="D",
        hashtags=["#x"],
        status=status,
    )


def _make_clip_env(conn, job_dir: Path, plans: list[ClipPlan]) -> None:
    """A clip as it would exist after the clip_plans stage: artifact + rows."""
    _write_plans(job_dir, plans)
    cursor = conn.cursor()
    for plan in plans:
        cursor.execute(
            "INSERT INTO clips (clip_id, job_id, plan_json, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 't0', 't0')",
            (plan.clip_id, plan.job_id, plan.model_dump_json(), plan.status),
        )
    conn.commit()


def _clip_rows(conn, job_id: str = JOB_ID) -> dict[str, str]:
    rows = conn.execute(
        "SELECT clip_id, status FROM clips WHERE job_id = ?", (job_id,)
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _write_render_outputs(job_dir: Path, clip_id: str) -> None:
    clip_dir = job_dir / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "final.mp4").write_bytes(b"FAKE-MP4-BYTES")
    (clip_dir / "thumbnail.jpg").write_bytes(b"FAKE-JPG-BYTES")


# ---------------------------------------------------------------------------
# adapters/qc.py — file I/O surface
# ---------------------------------------------------------------------------


def test_read_clip_plans_missing_is_empty_artifact(tmp_path):
    artifact = qc.read_clip_plans(tmp_path / "no-such-job")
    assert artifact.schema_version == 1
    assert artifact.plans == []


def test_clip_plans_write_read_roundtrip_canonical(tmp_path):
    plans = [_plan(CLIP01, "approved")]
    path = qc.write_clip_plans(tmp_path, ClipPlansArtifact(plans=plans))
    assert path == tmp_path / qc.CLIP_PLANS_FILENAME
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.startswith('{\n  "schema_version": 1,')  # indent=2 canonical
    loaded = qc.read_clip_plans(tmp_path)
    assert loaded.plans == plans


def test_clip_plans_invalid_content_raises(tmp_path):
    (tmp_path / qc.CLIP_PLANS_FILENAME).write_text(
        '{"schema_version": 99, "plans": "nope"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid .*clip_plans.json"):
        qc.read_clip_plans(tmp_path)


def test_read_sentences_missing_names_m2b_producer(tmp_path):
    with pytest.raises(ValueError, match="M2b transcribe stage") as exc:
        qc.read_sentences(tmp_path)
    assert "transcribe_job.py" in str(exc.value)


def test_read_sentences_roundtrip_and_invalid(tmp_path):
    sentences = [_SENT_B]
    _write_sentences(tmp_path, sentences)
    assert qc.read_sentences(tmp_path).sentences == sentences
    (tmp_path / "sentences.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid .*sentences.json"):
        qc.read_sentences(tmp_path)


def test_copy_file_creates_parents_and_matches_bytes(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00\x01binary\xff")
    dst = tmp_path / "a" / "b" / "deep" / "dst.bin"
    assert qc.copy_file(src, dst) == dst
    assert dst.read_bytes() == src.read_bytes()


def test_copy_file_missing_source_raises(tmp_path):
    with pytest.raises(ValueError, match="missing file to copy"):
        qc.copy_file(tmp_path / "gone.bin", tmp_path / "out.bin")


def test_has_file_only_files(tmp_path):
    assert not qc.has_file(tmp_path / "nothing")
    tmp_path.mkdir(parents=True, exist_ok=True)
    assert not qc.has_file(tmp_path)  # a directory is not a file
    (tmp_path / "real.bin").write_bytes(b"x")
    assert qc.has_file(tmp_path / "real.bin")


def test_write_text_file_roundtrip_creates_parents(tmp_path):
    target = tmp_path / "x" / "y" / "metadata.json"
    assert qc.write_text_file(target, '{"ok": 1}\n') == target
    assert target.read_text(encoding="utf-8") == '{"ok": 1}\n'


# ---------------------------------------------------------------------------
# force_select — m9-4 (NO LLM, padding + 15–75s validation, fallback metadata)
# ---------------------------------------------------------------------------


def _sentences_env(job_dir: Path) -> None:
    _write_sentences(
        job_dir,
        [_SENT_BEFORE, _SENT_B, _SENT_C, _SENT_AFTER],
    )


def test_force_select_happy_plan_fields(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 130.0))  # 30s span
    _sentences_env(job_dir)

    plan = qc_actions.force_select(CAND_ID, "editor pick", cfg, conn)

    # cut = candidate bounds + existing padding constants (outward)
    assert round(plan.start, 3) == 99.88   # 100.0 - cuts.PAD_START
    assert round(plan.end, 3) == 130.25    # 130.0 + cuts.PAD_END
    # m9-4 plan shape
    assert plan.forced is True
    assert plan.force_reason == "editor pick"
    assert plan.status == "pending_qc"
    assert plan.layout is Layout.center_crop
    assert plan.crop_path == []
    assert plan.weighted_score == 5.5      # candidate.cheap_score
    # m9-4 metadata = the deterministic fallback (clip-window transcript only)
    window = f"{_SENT_B.text} {_SENT_C.text}"
    words = window.split()
    assert plan.title == " ".join(words[:FALLBACK_TITLE_WORDS])
    assert plan.hook_overlay == " ".join(words[:FALLBACK_HOOK_WORDS])
    assert plan.description == window[:200]
    assert plan.hashtags == ["#science"]  # job niche
    # identity + persistence
    assert plan.clip_id == CLIP01
    stored = qc.read_clip_plans(job_dir).plans
    assert stored == [plan]
    assert _clip_rows(conn)[CLIP01] == "pending_qc"


def test_force_select_clip_id_lowest_free_slot(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 130.0))
    _sentences_env(job_dir)
    # clip01 + clip03 exist → clip02 is the lowest free slot
    _make_clip_env(conn, job_dir, [_plan(CLIP01), _plan(CLIP03)])

    plan = qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert plan.clip_id == CLIP02

    # next free slot after that one (clip01/02/03 all taken)
    plan2 = qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert plan2.clip_id == f"{JOB_ID}-clip04"


def test_force_select_start_clamped_at_zero(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(0.05, 50.0))  # span 49.95s
    _sentences_env(job_dir)

    plan = qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert plan.start == 0.0
    assert plan.end == 50.25


def test_force_select_too_short_rejects_and_writes_nothing(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 105.0))  # span + 0.37s pad
    _sentences_env(job_dir)

    with pytest.raises(
        QCActionsError, match="too short: 5\\.370s < 15\\.0s"
    ) as exc:
        qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert CAND_ID in str(exc.value)
    # rejected cut writes NOTHING
    assert not (job_dir / "clip_plans.json").exists()
    assert _clip_rows(conn) == {}


def test_force_select_too_long_rejects_and_writes_nothing(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 180.0))  # 80s + 0.37s pad
    _sentences_env(job_dir)

    with pytest.raises(
        QCActionsError, match="too long: 80\\.370s > 75\\.0s"
    ) as exc:
        qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert CAND_ID in str(exc.value)
    assert not (job_dir / "clip_plans.json").exists()
    assert _clip_rows(conn) == {}


def test_force_select_unknown_candidate_error(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _sentences_env(job_dir)
    with pytest.raises(
        QCActionsError, match="unknown candidate .*candidates table"
    ) as exc:
        qc_actions.force_select(f"{JOB_ID}-cand99", None, cfg, conn)
    assert "candidates_job.py" in str(exc.value)
    assert not (job_dir / "clip_plans.json").exists()


def test_force_select_candidate_of_ghost_job(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    ghost = _candidate(100.0, 130.0, cand_id=f"{JOB_ID}-cand01")
    _insert_candidate(conn, ghost, job_id="ghost-job")
    _sentences_env(job_dir)
    with pytest.raises(QCActionsError, match="unknown job 'ghost-job'"):
        qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert _clip_rows(conn) == {}


def test_force_select_missing_sentences_points_at_m2b(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 130.0))
    # NO sentences.json written
    with pytest.raises(
        QCActionsError, match="M2b transcribe stage"
    ) as exc:
        qc_actions.force_select(CAND_ID, None, cfg, conn)
    assert "transcribe_job.py" in str(exc.value)
    assert _clip_rows(conn) == {}


def test_force_select_window_sentences_only(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 130.0))
    _sentences_env(job_dir)

    plan = qc_actions.force_select(CAND_ID, None, cfg, conn)

    title_words = plan.title.split()
    assert title_words[0] == "alpha"  # inside window
    assert "zed" not in title_words   # before window: excluded
    assert "quebec" not in plan.description  # after window: excluded
    assert plan.description.startswith("alpha beta")
    assert plan.description.count(" ") == len(
        f"{_SENT_B.text} {_SENT_C.text}".split()
    ) - 1


def test_force_select_never_touches_immutable_artifacts(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 130.0))
    _sentences_env(job_dir)
    pins = {}
    for name in ("scores.json", "selection.json", "candidates.json", "words.json"):
        payload = f"IMMUTABLE-{name}\n".encode()
        (job_dir / name).write_bytes(payload)
        pins[name] = payload
    qc_actions.force_select(CAND_ID, "immutability", cfg, conn)

    for name, payload in pins.items():
        assert (job_dir / name).read_bytes() == payload  # byte-identical
    # and the candidates TABLE row is untouched
    rows = conn.execute("SELECT cand_id FROM candidates").fetchall()
    assert [r[0] for r in rows] == [CAND_ID]


def test_force_select_lock_contended_refuses(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _insert_candidate(conn, _candidate(100.0, 130.0))
    _sentences_env(job_dir)
    lock_file = lock_path(cfg)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    held = lock_file.open("a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(WorkerError, match="worker lock"):
            qc_actions.force_select(CAND_ID, None, cfg, conn)
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    # the contended action wrote nothing
    assert not (job_dir / "clip_plans.json").exists()
    assert _clip_rows(conn) == {}
    # and the lock is usable afterwards
    plan = qc_actions.force_select(CAND_ID, "after lock freed", cfg, conn)
    assert plan.clip_id == CLIP01


# ---------------------------------------------------------------------------
# approve / reject — m9-5 (status only, plan + row, nothing else)
# ---------------------------------------------------------------------------


def test_approve_updates_plan_and_row_only(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    first = _plan(CLIP01)
    other = _plan(CLIP02, start=240.0, end=250.0)
    _make_clip_env(conn, job_dir, [first, other])

    updated = qc_actions.approve(CLIP01, cfg, conn)

    assert updated.status == "approved"
    assert updated == first.model_copy(update={"status": "approved"})
    # the persisted artifact reflects the status, order + sibling untouched
    plans = qc.read_clip_plans(job_dir).plans
    assert [p.clip_id for p in plans] == [CLIP01, CLIP02]
    assert plans[0].status == "approved"
    assert plans[1] == other
    # the clips row agrees; the sibling row is untouched
    assert _clip_rows(conn) == {CLIP01: "approved", CLIP02: "pending_qc"}


def test_reject_updates_plan_and_row_only(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _make_clip_env(conn, job_dir, [_plan(CLIP01)])

    updated = qc_actions.reject(CLIP01, cfg, conn)

    assert updated.status == "rejected"
    assert qc.read_clip_plans(job_dir).plans[0].status == "rejected"
    assert _clip_rows(conn)[CLIP01] == "rejected"


def test_approve_unknown_clip_names_clips_table(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    with pytest.raises(
        QCActionsError, match="unknown clip .*clips table"
    ) as exc:
        qc_actions.approve(f"{JOB_ID}-clip99", cfg, conn)
    assert "force-select" in str(exc.value)  # the recovery path hint
    assert not (job_dir / "clip_plans.json").exists()


def test_reject_clip_missing_from_artifact(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    # row exists but clip_plans.json does NOT — hand-inserted row only
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO clips (clip_id, job_id, plan_json, status, created_at, "
        "updated_at) VALUES (?, ?, ?, 'pending_qc', 't0', 't0')",
        (CLIP01, JOB_ID, _plan(CLIP01).model_dump_json()),
    )
    conn.commit()
    assert not (job_dir / "clip_plans.json").exists()
    with pytest.raises(QCActionsError, match="not in .*clip_plans.json"):
        qc_actions.reject(CLIP01, cfg, conn)
    with pytest.raises(QCActionsError, match="not in .*clip_plans.json"):
        qc_actions.approve(CLIP01, cfg, conn)


def test_approve_lock_contended_refuses(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _make_clip_env(conn, job_dir, [_plan(CLIP01)])
    lock_file = lock_path(cfg)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    held = lock_file.open("a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(WorkerError, match="worker lock"):
            qc_actions.approve(CLIP01, cfg, conn)
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    # untouched by the contended call
    assert _clip_rows(conn)[CLIP01] == "pending_qc"
    assert qc.read_clip_plans(job_dir).plans[0].status == "pending_qc"


# ---------------------------------------------------------------------------
# export — m9-5 (approved only, copy-only, NO rerender)
# ---------------------------------------------------------------------------


def _approved_export_env(
    tmp_path, plan: ClipPlan | None = None
) -> tuple[Config, object, Path, ClipPlan]:
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    plan = plan or _plan(CLIP01, status="approved")
    _make_clip_env(conn, job_dir, [plan])
    _write_render_outputs(job_dir, plan.clip_id)
    return cfg, conn, job_dir, plan


def test_export_approved_copies_outputs_and_metadata(tmp_path):
    cfg, conn, job_dir, plan = _approved_export_env(tmp_path)

    written = qc_actions.export_job(JOB_ID, cfg, conn)

    out_dir = cfg.resolved_data_dir / "exports" / JOB_ID / plan.clip_id
    assert [p.name for p in written] == ["final.mp4", "thumbnail.jpg", "metadata.json"]
    assert all(p.parent == out_dir for p in written)
    # byte-identical copies
    expected = (job_dir / "clips" / plan.clip_id / "final.mp4").read_bytes()
    assert (out_dir / "final.mp4").read_bytes() == expected
    assert (out_dir / "thumbnail.jpg").read_bytes() == b"FAKE-JPG-BYTES"
    # metadata.json freshly written from the approved plan
    import json as _json

    meta = _json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["clip_id"] == plan.clip_id
    assert meta["job_id"] == JOB_ID
    assert meta["title"] == plan.title
    assert meta["hook_overlay"] == plan.hook_overlay
    assert meta["description"] == plan.description
    assert meta["hashtags"] == plan.hashtags
    assert meta["status"] == "approved"


def _job_tree(job_dir: Path) -> list[tuple[str, bytes]]:
    items: list[tuple[str, bytes]] = []
    for path in sorted(job_dir.rglob("*")):
        if path.is_file():
            items.append((str(path.relative_to(job_dir)), path.read_bytes()))
    return items


def test_export_is_read_only_for_the_job_dir(tmp_path):
    cfg, conn, job_dir, _plan_obj = _approved_export_env(tmp_path)
    before = _job_tree(job_dir)

    qc_actions.export_job(JOB_ID, cfg, conn)

    assert _job_tree(job_dir) == before  # nothing added/changed/removed
    assert (job_dir / "clips" / CLIP01 / "final.mp4").read_bytes() == b"FAKE-MP4-BYTES"


def test_export_zero_approved_is_empty_and_writes_nothing(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _make_clip_env(conn, job_dir, [_plan(CLIP01, status="pending_qc")])
    _write_render_outputs(job_dir, CLIP01)  # even rendered — not approved

    assert qc_actions.export_job(JOB_ID, cfg, conn) == []
    assert not (cfg.resolved_data_dir / "exports").exists()


def test_export_missing_final_names_clip(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _make_clip_env(conn, job_dir, [_plan(CLIP01, status="approved")])
    (job_dir / "clips" / CLIP01).mkdir(parents=True, exist_ok=True)
    (job_dir / "clips" / CLIP01 / "thumbnail.jpg").write_bytes(b"FAKE-JPG")

    with pytest.raises(QCActionsError, match=CLIP01) as exc:
        qc_actions.export_job(JOB_ID, cfg, conn)
    assert "final.mp4" in str(exc.value)
    assert "re-render" in str(exc.value)  # states it does NOT re-render
    assert not (cfg.resolved_data_dir / "exports" / JOB_ID / CLIP01).exists()


def test_export_missing_thumbnail_names_clip(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    _make_clip_env(conn, job_dir, [_plan(CLIP01, status="approved")])
    (job_dir / "clips" / CLIP01).mkdir(parents=True, exist_ok=True)
    (job_dir / "clips" / CLIP01 / "final.mp4").write_bytes(b"FAKE-MP4")

    with pytest.raises(QCActionsError, match=CLIP01) as exc:
        qc_actions.export_job(JOB_ID, cfg, conn)
    assert "thumbnail.jpg" in str(exc.value)


def test_export_approved_only_ignores_other_clips(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    approved = _plan(CLIP01, status="approved")
    pending = _plan(CLIP02, status="pending_qc", start=240.0, end=260.0)
    _make_clip_env(conn, job_dir, [approved, pending])
    _write_render_outputs(job_dir, CLIP01)
    _write_render_outputs(job_dir, CLIP02)  # rendered but NOT approved

    written = qc_actions.export_job(JOB_ID, cfg, conn)

    out_root = cfg.resolved_data_dir / "exports" / JOB_ID
    assert {p.name for p in written} == {"final.mp4", "thumbnail.jpg", "metadata.json"}
    assert all(p.parent == out_root / CLIP01 for p in written)
    assert not (out_root / CLIP02).exists()


def test_export_multiple_clips_ordered_by_clip_id(tmp_path):
    cfg, dbobj, conn, job_dir = _setup(tmp_path)
    a = _plan(CLIP01, status="approved")
    b = _plan(CLIP02, status="approved", start=240.0, end=260.0)
    _make_clip_env(conn, job_dir, [a, b])
    _write_render_outputs(job_dir, CLIP01)
    _write_render_outputs(job_dir, CLIP02)

    written = qc_actions.export_job(JOB_ID, cfg, conn)

    clip_dirs = [p.parent.name for p in written]
    assert clip_dirs == [CLIP01, CLIP01, CLIP01, CLIP02, CLIP02, CLIP02]


def test_export_unknown_job(tmp_path):
    cfg, dbobj, conn, _ = _setup(tmp_path)
    with pytest.raises(QCActionsError, match="unknown job 'ghost-job'"):
        qc_actions.export_job("ghost-job", cfg, conn)


# ---------------------------------------------------------------------------
# CLI (typer CliRunner)
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    cfg = Config.load_config()
    dbobj = Database(cfg)
    assert dbobj.create_job(JOB_ID, "/tmp/m9d-source.mp4", "client", "science")
    return cfg, dbobj


def _cli_job_dir(cfg: Config) -> Path:
    return cfg.resolved_data_dir / "jobs" / JOB_ID


def test_cli_approve_happy(cli_env, monkeypatch, tmp_path):
    cfg, dbobj = cli_env
    job_dir = _cli_job_dir(cfg)
    job_dir.mkdir(parents=True, exist_ok=True)
    _make_clip_env(dbobj.get_connection(), job_dir, [_plan(CLIP01)])
    runner = CliRunner()

    result = runner.invoke(cli.app, ["approve", CLIP01])
    assert result.exit_code == 0, result.output
    assert f"{CLIP01}: approved" in result.output


def test_cli_approve_unknown_exit_1(cli_env, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["approve", f"{JOB_ID}-clip99"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "unknown clip" in result.output


def test_cli_reject_happy(cli_env, tmp_path):
    cfg, dbobj = cli_env
    job_dir = _cli_job_dir(cfg)
    job_dir.mkdir(parents=True, exist_ok=True)
    _make_clip_env(dbobj.get_connection(), job_dir, [_plan(CLIP01)])
    runner = CliRunner()

    result = runner.invoke(cli.app, ["reject", CLIP01])
    assert result.exit_code == 0, result.output
    assert f"{CLIP01}: rejected" in result.output


def test_cli_force_select_happy(cli_env, tmp_path):
    cfg, dbobj = cli_env
    conn = dbobj.get_connection()
    job_dir = _cli_job_dir(cfg)
    job_dir.mkdir(parents=True, exist_ok=True)
    _insert_candidate(conn, _candidate(100.0, 130.0))
    _sentences_env(job_dir)
    runner = CliRunner()

    result = runner.invoke(
        cli.app, ["force-select", CAND_ID, "--reason", "editor pick"]
    )
    assert result.exit_code == 0, result.output
    assert CLIP01 in result.output
    assert "forced selection from candidate" in result.output
    assert "pending_qc" in result.output
    assert "editor pick" in result.output
    conn.close()


def test_cli_force_select_invalid_cut_exit_1(cli_env, tmp_path):
    cfg, dbobj = cli_env
    conn = dbobj.get_connection()
    job_dir = _cli_job_dir(cfg)
    job_dir.mkdir(parents=True, exist_ok=True)
    _insert_candidate(conn, _candidate(100.0, 105.0))
    _sentences_env(job_dir)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["force-select", CAND_ID])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "too short" in result.output
    assert "15.0s" in result.output
    conn.close()


def test_cli_export_zero_approved(cli_env):
    cfg, dbobj = cli_env
    _make_clip_env(dbobj.get_connection(), _cli_job_dir(cfg), [_plan(CLIP01)])
    _cli_job_dir(cfg).mkdir(parents=True, exist_ok=True)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["export", JOB_ID])
    assert result.exit_code == 0, result.output
    assert "No approved clips to export" in result.output


def test_cli_export_happy_lists_files(cli_env, tmp_path):
    cfg, dbobj = cli_env
    job_dir = _cli_job_dir(cfg)
    job_dir.mkdir(parents=True, exist_ok=True)
    _make_clip_env(dbobj.get_connection(), job_dir, [_plan(CLIP01, status="approved")])
    _write_render_outputs(job_dir, CLIP01)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["export", JOB_ID])
    assert result.exit_code == 0, result.output
    assert "Exported 3 file(s)" in result.output
    assert f"exports/{JOB_ID}/{CLIP01}/final.mp4" in result.output.replace("\\", "/")


# ---------------------------------------------------------------------------
# Module hygiene — no LLM / render / network surface in the QC actions layer
# ---------------------------------------------------------------------------


def test_qc_actions_has_no_llm_render_network_surface():
    from ast import Import, ImportFrom

    import clipper.qc_actions as actions_mod

    names: set[str] = set()
    for module in (actions_mod, qc):
        tree = ast_parse(inspect.getsource(module))
        for node in tree.body:
            if isinstance(node, Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ImportFrom) and node.module:
                names.add(node.module)
    forbidden = {
        "clipper.llm_runtime",
        "clipper.adapters.llm",
        "requests",
        "httpx",
        "urllib",
        "subprocess",
        "socket",
    }
    assert not (names & forbidden), names & forbidden
    # the LLM runtime must not even be reachable attribute-wise from the
    # QC actions namespace (the module docstrings may *mention* ollama/
    # ffmpeg — only live references are forbidden)
    assert not hasattr(actions_mod, "llm_runtime")
    assert "managed_ollama" not in dir(actions_mod)
