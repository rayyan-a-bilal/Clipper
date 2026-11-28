"""
M9e — single-clip rerender + authoritative preview tests (CPU only).

Covers the service layer `clipper.rerender` per `docs/spec/m9.md`
m9-5 / m9-10 / m9-12 (+ M9e judgment notes M9e-J1 … M9e-J5):

- pure helpers — `_next_version`, `_prune_old_versions` (retain last 3,
  prune older files), `_before_plan` (m9-12 snapshot; "first render"
  fallback with every M9 knob cleared), `_ass_text` (M9e-J1 shadow-cfg +
  m9-11 burn_in switches)
- `rerender_clip` — no-op, dirty-token matrix (caption-only reuses the
  mezzanine/audio; trim re-renders both), versioning (`final.v<N>.mp4`,
  `final.mp4` copy, `render.json` latest, `render_history.json` +
  `plan_snapshot`), pruning, concrete errors
- `preview_frame` — writes `preview.png`, clamps T into the effective
  cut, refuses stale/missing inputs with a concrete reason
- worker-lock contention → clean `WorkerError` refusal

CPU-only: temp data dir, temp DB, `FixtureRenderAdapter` / fake adapter.
No GPU, no ffmpeg, no network, no Ollama.
"""

from pathlib import Path

import pytest

import clipper.cli as cli
import clipper.rerender as rer
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.models import (
    CaptionOverrides,
    ClipLayoutArtifact,
    ClipPlan,
    Layout,
    ProbeResult,
    RenderArtifact,
    RenderHistory,
    RenderVersion,
    Word,
    WordsArtifact,
)
from clipper.rerender import RerenderError

JOB_ID = "job-m9e"
CLIP01 = f"{JOB_ID}-clip01"
CLIP02 = f"{JOB_ID}-clip02"


def _base_plan(**overrides) -> ClipPlan:
    defaults = {
        "clip_id": CLIP01,
        "job_id": JOB_ID,
        "start": 100.0,
        "end": 115.0,
        "layout": Layout.center_crop,
        "crop_path": [],
        "weighted_score": 8.0,
        "title": "t",
        "hook_overlay": "hook line",
        "description": "",
        "hashtags": [],
        "status": "pending_qc",
    }
    defaults.update(overrides)
    return ClipPlan(**defaults)



@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


def _mk_cfg(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(data_dir=str(data_dir)))


def _setup(tmp_path: Path, plans: dict[str, ClipPlan] | None = None):
    """cfg + db + conn + one job + clip rows (+ job dir). Returns
    (cfg, dbobj, conn, job_dir, clip_dir)."""
    plans = plans if plans is not None else {CLIP01: _base_plan()}
    cfg = _mk_cfg(tmp_path)
    dbobj = Database(cfg)
    assert dbobj.create_job(
        JOB_ID, "/tmp/m9e-source.mp4", "client", "science"
    )
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    conn = dbobj.get_connection()
    for clip_id, plan in plans.items():
        conn.execute(
            "INSERT INTO clips (clip_id, job_id, plan_json, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, "
            "datetime('now'), datetime('now'))",
            (clip_id, plan.job_id, plan.model_dump_json(), plan.status),
        )
    conn.commit()
    clip_dir = job_dir / "clips" / CLIP01
    clip_dir.mkdir(parents=True, exist_ok=True)
    return cfg, dbobj, conn, job_dir, clip_dir


def _write_words(job_dir: Path) -> None:
    words = [Word(start=100.0, end=100.4, text=f"w{i}", prob=0.9) for i in range(6)]
    artifact = WordsArtifact(schema_version=1, words=words)
    (job_dir / "words.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _write_probe(job_dir: Path) -> None:
    probe = ProbeResult(
        duration_s=600.0, width=1920, height=1080, fps=30.0,
        vcodec="h264", acodec="aac", has_audio=True, size_bytes=1,
    )
    (job_dir / "probe.json").write_text(
        probe.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _write_layout(clip_dir: Path, clip_id: str = CLIP01) -> None:
    artifact = ClipLayoutArtifact(
        clip_id=clip_id,
        layout=Layout.center_crop,
        source_width=1920,
        source_height=1080,
        native_fps=30.0,
        tracks=[],
        face_track_count=0,
        detector="",
        layout_source="auto",
    )
    (clip_dir / "layout.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _version(n: int, plan: ClipPlan | None = None) -> RenderVersion:
    return RenderVersion(
        n=n,
        final_path=f"/tmp/x/clips/{CLIP01}/final.v{n}.mp4",
        created_at="2026-12-02T00:00:{n:02d}",
        dirty=["caption"],
        plan_snapshot=plan,
    )


def _write_history(clip_dir: Path, versions: list[RenderVersion]) -> None:
    history = RenderHistory(
        schema_version=1, clip_id=CLIP01, versions=versions,
    )
    (clip_dir / "render_history.json").write_text(
        history.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _set_clip_row(conn, plan: ClipPlan) -> None:
    conn.execute(
        "UPDATE clips SET plan_json=?, status=? WHERE clip_id=?",
        (plan.model_dump_json(), plan.status, plan.clip_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Chunk 2 — pure helpers
# ---------------------------------------------------------------------------


class TestNextVersion:
    def test_empty_dir_starts_at_one(self, tmp_path: Path):
        d = tmp_path / "c"
        d.mkdir()
        assert rer._next_version(d, None) == 1

    def test_max_of_files_and_history(self, tmp_path: Path):
        d = tmp_path / "c"
        d.mkdir()
        (d / "final.v3.mp4").write_text("x")
        (d / "final.v9.mp4").write_text("x")
        (d / "final.mp4").write_text("x")  # not a version file
        (d / "final.v2.mp4.backup").write_text("x")  # must not match
        assert rer._next_version(d, None) == 10

    def test_history_can_be_ahead_of_files(self, tmp_path: Path):
        d = tmp_path / "c"
        d.mkdir()
        history = RenderHistory(
            schema_version=1, clip_id=CLIP01,
            versions=[self._v(7), self._v(6)],
        )
        assert rer._next_version(d, history) == 8

    @staticmethod
    def _v(n: int) -> RenderVersion:
        return RenderVersion(
            n=n, final_path=f"/tmp/z/final.v{n}.mp4",
            created_at="2026-12-02T00:00:00",
        )


class TestPruneOldVersions:
    def test_retains_last_three_and_prunes_files(self, tmp_path: Path):
        d = tmp_path / "c"
        d.mkdir()
        for n in (1, 2, 3, 4, 5):
            (d / f"final.v{n}.mp4").write_text(f"v{n}")
        kept = [self._v(5), self._v(4), self._v(3)]
        removed = rer._prune_old_versions(d, kept)
        assert {p.name for p in removed} == {"final.v1.mp4", "final.v2.mp4"}
        assert sorted(p.name for p in d.iterdir()) == [
            "final.v3.mp4", "final.v4.mp4", "final.v5.mp4",
        ]

    def test_history_kept_cap(self, tmp_path: Path):
        d = tmp_path / "c"
        d.mkdir()
        kept = [self._v(n) for n in (5, 4, 3, 2)]  # longer than retain
        rer._prune_old_versions(d, kept)  # nothing on disk, must not raise
        assert len(kept[: rer.RETAIN_VERSIONS]) == 3

    @staticmethod
    def _v(n: int) -> RenderVersion:
        return RenderVersion(
            n=n, final_path=f"/tmp/z/final.v{n}.mp4",
            created_at="2026-12-02T00:00:00",
        )


class TestBeforePlan:
    def test_snapshot_wins_latest_first(self):
        snap_a = _base_plan(caption_overrides=CaptionOverrides(font="A"))
        snap_b = _base_plan()
        history = RenderHistory(
            schema_version=1, clip_id=CLIP01, versions=[
                _version(2, None),            # no snapshot (old entry)
                _version(1, snap_a),
            ]
        )
        # latest-first: version 2 has no snapshot → version 1's wins
        assert rer._before_plan(snap_b, history) is snap_a

    def test_newer_snapshot_wins(self):
        old = _base_plan()
        new = _base_plan(caption_overrides=CaptionOverrides(font="B"))
        history = RenderHistory(
            schema_version=1, clip_id=CLIP01, versions=[
                _version(2, new), _version(1, old),
            ]
        )
        assert rer._before_plan(_base_plan(), history) is new

    def test_first_render_fallback_clears_m9_knobs(self):
        plan = _base_plan(
            caption_overrides=CaptionOverrides(font="F"),
            style_id="compact",
            manual_start=101.0,
            manual_end=114.0,
        )
        before = rer._before_plan(plan, None)
        assert before.caption_overrides is None
        assert before.style_id is None
        assert before.manual_start is None
        assert before.manual_end is None
        # non-M9 fields untouched
        assert before.start == plan.start and before.end == plan.end


class TestAssTextBurnIn:
    def _cfg(self, tmp_path: Path) -> Config:
        return _mk_cfg(tmp_path)

    def _words(self):
        # INSIDE the base plan window [100, 115] (the engine selects
        # words by the plan window — words outside it are no-ops).
        words = []
        for i in range(4):
            words.append(
                Word(start=100.0 + 0.3 * i, end=100.2 + 0.3 * i, text=f"w{i}", prob=0.9)
            )
        return words

    def _resolve(self, cfg: Config, clip_ovr=None, **extra):
        from clipper.adapters import assets as m9_assets
        from clipper.overrides import resolve_caption
        zones = m9_assets.load_safe_zones(rer._REPO_ROOT / "assets" / "safe_zones.json")
        return resolve_caption(cfg.captions, None, None, clip_ovr, zones)

    def test_full_keeps_hook_and_captions(self, tmp_path: Path):
        cfg = self._cfg(tmp_path)
        plan = _base_plan()
        resolved = self._resolve(cfg, CaptionOverrides(burn_in="full"))
        text = rer._ass_text(cfg, plan, self._words(), resolved)
        assert "Dialogue:" in text
        assert len([ln for ln in text.split("\n") if ln.startswith("Dialogue:")]) >= 2

    def test_none_drops_all_dialogues(self, tmp_path: Path):
        cfg = self._cfg(tmp_path)
        resolved = self._resolve(cfg, CaptionOverrides(burn_in="none"))
        text = rer._ass_text(cfg, _base_plan(), self._words(), resolved)
        assert not any(ln.startswith("Dialogue:") for ln in text.split("\n"))

    def test_captions_only_drops_hook_keeps_words(self, tmp_path: Path):
        cfg = self._cfg(tmp_path)
        resolved = self._resolve(cfg, CaptionOverrides(burn_in="captions_only"))
        text = rer._ass_text(cfg, _base_plan(), self._words(), resolved)
        lines = [ln for ln in text.split("\n") if ln.startswith("Dialogue:")]
        assert lines, "word caption dialogues must remain"
        assert not any(ln.split(",")[4].strip() == "Hook" for ln in lines)

    def test_hook_enabled_false_drops_hook(self, tmp_path: Path):
        cfg = self._cfg(tmp_path)
        resolved = self._resolve(cfg, CaptionOverrides(hook_enabled=False))
        text = rer._ass_text(cfg, _base_plan(), self._words(), resolved)
        lines = [ln for ln in text.split("\n") if ln.startswith("Dialogue:")]
        assert lines
        assert not any(ln.split(",")[4].strip() == "Hook" for ln in lines)


# ---------------------------------------------------------------------------
# Chunk 3 — rerender_clip with a fake render adapter
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Records every render call; writes tiny placeholder files."""

    def __init__(self):
        self.calls: list[tuple] = []

    def _write(self, dst: str, tag: str) -> str:
        path = Path(dst)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tag.encode())
        return dst

    def render_mezzanine(self, source, plan, layout, dst, cfg, **kw):
        self.calls.append(("mezzanine", layout))
        return self._write(dst, "MEZZ")

    def extract_audio(self, source, start, end, dst):
        self.calls.append(("audio", start, end))
        return self._write(dst, "AUD")

    def compose_final(self, mezz, audio, ass, dst, cfg):
        self.calls.append(("compose", ass))
        from clipper.models import RenderArtifact
        artifact = RenderArtifact(
            clip_id="placeholder",
            mezzanine_path=mezz,
            audio_path=audio,
            captions_path=ass,
            final_path=dst,
            thumbnail_path=dst + ".thumb",
            width=1080, height=1920, fps=30.0, duration_s=15.0,
            encoder="h264_nvenc", hwaccel="nvdec", render_seconds=1.0,
        )
        self._write(dst, "FINAL")
        return artifact

    def make_thumbnail(self, final, dst):
        self.calls.append(("thumbnail", final))
        return self._write(dst, "THUMB")


@pytest.fixture
def fake_adapter(monkeypatch):
    adapter = FakeAdapter()
    monkeypatch.setattr(rer, "get_render_adapter", lambda cfg: adapter)
    return adapter


class TestRerenderClip:
    def test_caption_only_reuses_mezzanine_and_audio(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        _write_words(job_dir)
        # The M7d originals (caption-only rerender must REUSE these —
        # never re-render mezzanine/audio).
        (clip_dir / "mezzanine.mp4").write_bytes(b"old-meze")
        (clip_dir / "audio.m4a").write_bytes(b"old-aud")
        # A plan WITH a caption override → dirty {"caption"} vs the
        # first-render fallback (knobs cleared).
        plan = _base_plan(
            caption_overrides=CaptionOverrides(font="Montserrat ExtraBold")
        )
        _set_clip_row(conn, plan)

        result = rer.rerender_clip(CLIP01, cfg, conn)

        assert result["no_op"] is False
        assert result["dirty"] == ["caption"]
        assert result["version"] == 1
        assert result["relayout"] is False
        # PASS MATRIX: caption-only → compose + thumbnail ONLY
        assert [c[0] for c in fake_adapter.calls] == ["compose", "thumbnail"]
        # mezzanine/audio untouched (still the M7d bytes)
        assert (clip_dir / "mezzanine.mp4").read_bytes() == b"old-meze"
        assert (clip_dir / "audio.m4a").read_bytes() == b"old-aud"
        # version history contract
        assert (clip_dir / "final.v1.mp4").read_bytes() == b"FINAL"
        assert (clip_dir / "final.mp4").read_bytes() == b"FINAL"
        assert (clip_dir / "thumbnail.jpg").read_bytes() == b"THUMB"
        # render.json: stable final.mp4 name, latest content
        artifact = RenderArtifact.model_validate_json(
            (clip_dir / "render.json").read_text()
        )
        assert artifact.clip_id == CLIP01
        assert artifact.final_path.endswith("final.mp4")
        # history: latest first, plan_snapshot == the rendered plan (m9-12)
        history = RenderHistory.model_validate_json(
            (clip_dir / "render_history.json").read_text()
        )
        assert [v.n for v in history.versions] == [1]
        assert history.versions[0].dirty == ["caption"]
        assert history.versions[0].plan_snapshot is not None
        assert history.versions[0].plan_snapshot.style_id is None
        assert history.versions[0].resolved_caption is not None

    def test_trim_rerender_re_renders_media(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-video")
        # point the job at a REAL file (the code checks Path.is_file)
        conn.execute("UPDATE jobs SET source_path=? WHERE job_id=?", (str(src), JOB_ID))
        conn.commit()
        _write_words(job_dir)
        _write_probe(job_dir)
        _write_layout(clip_dir)
        plan = _base_plan(
            manual_start=101.0, manual_end=116.0,
            caption_overrides=CaptionOverrides(font="X"),
        )
        _set_clip_row(conn, plan)

        result = rer.rerender_clip(CLIP01, cfg, conn)

        assert [c[0] for c in fake_adapter.calls] == [
            "mezzanine", "audio", "compose", "thumbnail",
        ]
        audio_call = fake_adapter.calls[1]
        assert (audio_call[1], audio_call[2]) == (101.0, 116.0)
        assert result["dirty"] == ["caption", "trim"]
        # the clip row is untouched (status/plan) — QC state is orthogonal
        row = conn.execute(
            "SELECT status, plan_json FROM clips WHERE clip_id=?", (CLIP01,)
        ).fetchone()
        again = ClipPlan.model_validate_json(row[1])
        assert again.manual_start == 101.0
        assert again.status == "pending_qc"

    def test_second_rerender_dirty_computed_from_snapshot(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        _write_words(job_dir)
        (clip_dir / "mezzanine.mp4").write_bytes(b"m")
        (clip_dir / "audio.m4a").write_bytes(b"a")
        _write_probe(job_dir)
        _write_layout(clip_dir)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"v")
        conn.execute("UPDATE jobs SET source_path=? WHERE job_id=?", (str(src), JOB_ID))
        conn.commit()
        # v1: caption override present vs the first-render fallback (cleared)
        plan1 = _base_plan(caption_overrides=CaptionOverrides(font="F1"))
        _set_clip_row(conn, plan1)
        r1 = rer.rerender_clip(CLIP01, cfg, conn)
        assert r1["version"] == 1 and r1["dirty"] == ["caption"]
        # v2: only the TRIM changes vs the v1 snapshot → dirty == {trim}
        plan2 = plan1.model_copy(update={"manual_start": 101.0, "manual_end": 116.0})
        _set_clip_row(conn, plan2)
        r2 = rer.rerender_clip(CLIP01, cfg, conn)
        assert r2["version"] == 2
        assert r2["dirty"] == ["trim"]
        # call order: v1 caption-only (reuse), v2 trim pass (re-render)
        assert [c[0] for c in fake_adapter.calls] == [
            "compose", "thumbnail", "mezzanine", "audio",
            "compose", "thumbnail",
        ]
        history = RenderHistory.model_validate_json(
            (clip_dir / "render_history.json").read_text()
        )
        assert [v.n for v in history.versions] == [2, 1]
        assert history.versions[0].dirty == ["trim"]
        assert history.versions[1].dirty == ["caption"]

    def test_no_op_success_when_unchanged(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        _write_words(job_dir)
        plan = _base_plan()
        _set_clip_row(conn, plan)
        # Simulate a prior M9e render that recorded an IDENTICAL snapshot.
        _write_history(clip_dir, [_version(1, plan.model_copy(deep=True))])
        (clip_dir / "final.mp4").write_bytes(b"FINAL")
        (clip_dir / "final.v1.mp4").write_bytes(b"FINAL")

        result = rer.rerender_clip(CLIP01, cfg, conn)

        assert result["no_op"] is True
        assert result["dirty"] == []
        assert fake_adapter.calls == []
        assert not (clip_dir / "final.v2.mp4").exists()

    def test_no_dirty_no_final_is_concrete_error(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        _write_words(job_dir)
        plan = _base_plan()
        _set_clip_row(conn, plan)
        _write_history(clip_dir, [_version(1, plan.model_copy(deep=True))])
        with pytest.raises(RerenderError, match="M7 render stage"):
            rer.rerender_clip(CLIP01, cfg, conn)
        assert fake_adapter.calls == []

    def test_version_numbers_prune_to_last_three(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        _write_words(job_dir)
        (clip_dir / "mezzanine.mp4").write_bytes(b"m")
        (clip_dir / "audio.m4a").write_bytes(b"a")
        # 5 renders, distinct caption override each time
        for i, font in enumerate(["F1", "F2", "F3", "F4", "F5"]):
            _set_clip_row(
                conn, _base_plan(caption_overrides=CaptionOverrides(font=font))
            )
            result = rer.rerender_clip(CLIP01, cfg, conn)
            assert result["version"] == i + 1
        assert sorted(p.name for p in clip_dir.glob("final.v*.mp4")) == [
            "final.v3.mp4", "final.v4.mp4", "final.v5.mp4",
        ]
        history = RenderHistory.model_validate_json(
            (clip_dir / "render_history.json").read_text()
        )
        assert [v.n for v in history.versions] == [5, 4, 3]
        assert [v.plan_snapshot.caption_overrides.font for v in history.versions] == [
            "F5", "F4", "F3",
        ]

    def test_relayout_reruns_layout_for_clip(self, tmp_path, fake_adapter, monkeypatch):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(
            tmp_path, plans={CLIP01: _base_plan(), CLIP02: _base_plan(clip_id=CLIP02)}
        )
        src = tmp_path / "src.mp4"
        src.write_bytes(b"v")
        conn.execute("UPDATE jobs SET source_path=? WHERE job_id=?", (str(src), JOB_ID))
        conn.commit()
        _write_words(job_dir)
        _write_probe(job_dir)
        _write_layout(clip_dir)
        other_clip_dir = job_dir / "clips" / CLIP02
        other_clip_dir.mkdir(parents=True, exist_ok=True)

        layout_calls: list[tuple] = []

        def fake_run_layout(job_id, cfg2, conn2, vision, detector, clip_id=None):
            layout_calls.append((job_id, clip_id))
            # the stage records the decision on the clip row + artifact
            _set_clip_row(conn, _base_plan(layout=Layout.face_follow))

        monkeypatch.setattr(rer, "run_layout", fake_run_layout)
        monkeypatch.setattr(rer, "get_face_detector", lambda cfg: "fake-detector")
        monkeypatch.setattr(rer, "get_vision_adapter", lambda cfg: "fake-vision")
        result = rer.rerender_clip(CLIP01, cfg, conn, relayout=True)

        assert layout_calls == [(JOB_ID, CLIP01)]  # THIS clip only
        assert result["relayout"] is True
        assert "layout" in result["dirty"]
        assert [c[0] for c in fake_adapter.calls][0] == "mezzanine"
        # Other clip's dir untouched
        assert list(other_clip_dir.iterdir()) == []

    def test_unknown_clip_is_concrete_error(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        with pytest.raises(RerenderError, match="unknown clip"):
            rer.rerender_clip("no-such-clip", cfg, conn)

    def test_caption_only_missing_mezzanine_errors(self, tmp_path, fake_adapter):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        _write_words(job_dir)
        _set_clip_row(conn, _base_plan(caption_overrides=CaptionOverrides(font="X")))
        with pytest.raises(RerenderError, match="mezzanine"):
            rer.rerender_clip(CLIP01, cfg, conn)
        assert fake_adapter.calls == []  # never reached any render pass

    def test_lock_contended_refuses_cleanly(self, tmp_path, fake_adapter):
        import fcntl
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        from clipper.worker import WorkerError, lock_path
        lock_file = lock_path(cfg)
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        held = lock_file.open("a+")
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(WorkerError, match="worker lock"):
                rer.rerender_clip(CLIP01, cfg, conn)
        finally:
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)
            held.close()
        assert fake_adapter.calls == []


# ---------------------------------------------------------------------------
# Chunk 4 — preview_frame (m9-10 tier 2)
# ---------------------------------------------------------------------------


class TestPreviewFrame:
    def test_burns_one_frame_and_clamps_t(self, tmp_path, monkeypatch):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        (clip_dir / "mezzanine.mp4").write_bytes(b"m")
        (clip_dir / "captions.ass").write_text("ass")
        plan = _base_plan()
        _set_clip_row(conn, plan)
        # recorded snapshot == plan → no pending change (not stale)
        _write_history(clip_dir, [_version(1, plan.model_copy(deep=True))])

        preview_calls: list[tuple] = []

        class PrevAdapter:
            def preview(self, mezz, ass, at_s, dst):
                preview_calls.append((mezz, ass, at_s, dst))
                Path(dst).write_bytes(b"PNG")
                return dst

        monkeypatch.setattr(rer, "get_render_adapter", lambda c: PrevAdapter())
        out = rer.preview_frame(CLIP01, 103.0, cfg, conn)
        assert out.name == "preview.png"
        assert out.read_bytes() == b"PNG"
        mezz, ass, t, dst = preview_calls[0]
        assert mezz.endswith("mezzanine.mp4")
        assert ass.endswith("captions.ass")
        assert t == pytest.approx(3.0)  # 103 clamped into [100,115] → 100+3
        assert dst == str(clip_dir / "preview.png")

    def test_out_of_range_t_clamped(self, tmp_path, monkeypatch):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        (clip_dir / "mezzanine.mp4").write_bytes(b"m")
        (clip_dir / "captions.ass").write_text("ass")
        plan = _base_plan()
        _set_clip_row(conn, plan)
        _write_history(clip_dir, [_version(1, plan.model_copy(deep=True))])

        seen: list[float] = []

        class PrevAdapter:
            def preview(self, mezz, ass, at_s, dst):
                seen.append(at_s)
                Path(dst).write_bytes(b"PNG")
                return dst

        monkeypatch.setattr(rer, "get_render_adapter", lambda c: PrevAdapter())
        rer.preview_frame(CLIP01, 90.0, cfg, conn)   # before start  → 0
        assert seen[0] == pytest.approx(0.0)
        rer.preview_frame(CLIP01, 500.0, cfg, conn)  # past end      → 15
        assert seen[-1] == pytest.approx(15.0)

    def test_missing_mezzanine_is_concrete_error(self, tmp_path):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        with pytest.raises(RerenderError, match="stale|missing"):
            rer.preview_frame(CLIP01, 103.0, cfg, conn)

    def test_pending_trim_is_stale(self, tmp_path):
        cfg, dbobj, conn, job_dir, clip_dir = _setup(tmp_path)
        (clip_dir / "mezzanine.mp4").write_bytes(b"m")
        (clip_dir / "captions.ass").write_text("ass")
        old_plan = _base_plan()
        history_plan = old_plan.model_copy(deep=True)
        _set_clip_row(conn, old_plan.model_copy(update={"manual_start": 102.0}))
        # latest snapshot is the OLD (pre-trim) plan → trim pending → stale
        _write_history(clip_dir, [_version(1, history_plan)])
        with pytest.raises(RerenderError, match="stale"):
            rer.preview_frame(CLIP01, 103.0, cfg, conn)


# ---------------------------------------------------------------------------
# Chunk 6 — CLI (typer) surface
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    from clipper.db import Database
    cfg = Config.load_config()
    dbobj = Database(cfg)
    assert dbobj.create_job(JOB_ID, "/tmp/x.mp4", "client", "science")
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    conn = dbobj.get_connection()
    _set_clip_row(conn, _base_plan())
    return cfg, dbobj


class TestCli:
    def test_cli_rerender_happy(self, cli_env_cli, monkeypatch):
        import typer.testing as tt
        calls = []

        def fake_rerender(clip_id, config, conn, relayout=False, cpu_render=False):
            calls.append((clip_id, relayout, cpu_render))
            return {
                "no_op": False, "dirty": ["caption"], "relayout": False,
                "version": 2, "final_path": "/x/final.mp4",
                "versioned_path": "/x/final.v2.mp4",
                "thumbnail_path": "/x/thumbnail.jpg",
                "removed_versions": ["/x/final.v1.mp4"], "render_seconds": 0.1,
            }

        monkeypatch.setattr(rer, "rerender_clip", fake_rerender)
        result = tt.CliRunner().invoke(cli.app, ["rerender", CLIP01, "--relayout"])
        assert result.exit_code == 0, result.output
        assert calls == [(CLIP01, True, False)]
        assert "rendered v2" in result.output
        assert "dirty: caption" in result.output
        assert "pruned: /x/final.v1.mp4" in result.output

    def test_cli_rerender_noop_message(self, cli_env_cli, monkeypatch):
        import typer.testing as tt
        monkeypatch.setattr(rer, "rerender_clip", lambda *a, **k: {
            "no_op": True, "dirty": [], "relayout": False, "version": None,
            "final_path": "/x/final.mp4", "versioned_path": None,
            "thumbnail_path": None, "removed_versions": [], "render_seconds": 0.0,
        })
        result = tt.CliRunner().invoke(cli.app, ["rerender", CLIP01])
        assert result.exit_code == 0, result.output
        assert "nothing changed" in result.output and "no-op" in result.output

    def test_cli_rerender_error_exit_1(self, cli_env_cli, monkeypatch):
        import typer.testing as tt
        def boom(*a, **k):
            raise RerenderError("the concrete reason")
        monkeypatch.setattr(rer, "rerender_clip", boom)
        result = tt.CliRunner().invoke(cli.app, ["rerender", CLIP01])
        assert result.exit_code == 1
        assert "the concrete reason" in result.output

    def test_cli_preview_passes_at_and_prints_path(self, cli_env_cli, monkeypatch):
        import typer.testing as tt
        calls = []
        monkeypatch.setattr(rer, "preview_frame", lambda clip_id, at, config, conn: (
            calls.append((clip_id, at)) or Path("/x/preview.png")
        ))
        result = tt.CliRunner().invoke(cli.app, ["preview", CLIP01, "--at", "103.5"])
        assert result.exit_code == 0, result.output
        assert calls == [(CLIP01, 103.5)]
        assert "/x/preview.png" in result.output

    def test_cli_preview_at_required(self, cli_env_cli):
        import typer.testing as tt
        result = tt.CliRunner().invoke(cli.app, ["preview", CLIP01])
        assert result.exit_code != 0  # usage error: --at is required
