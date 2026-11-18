"""
M3b analysis stage tests — CPU only.

Uses FixtureMediaAdapter / FixtureSceneAdapter, a temp DB and generated
WAV files. The final end-to-end test (real ffmpeg + scenedetect against
tests/fixtures/sample.mp4) is skipped when ffmpeg or scenedetect are
unavailable.
"""

import importlib.util
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

import clipper.adapters.media as media_mod
import clipper.adapters.scenes as scenes_mod
from clipper.adapters.media import FixtureMediaAdapter
from clipper.adapters.scenes import FixtureSceneAdapter
from clipper.config import Config, PathsConfig
from clipper.db import Database
from clipper.models import Shot, SilenceRegion
from clipper.stages import StageError
from clipper.stages.analyze import run_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
DURATION_S = 1.5
JOB_ID = "job-test"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(data_dir=str(data_dir)))


def _write_wav(
    path: Path,
    seconds: float = DURATION_S,
    rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write a mono/stereo 16-bit PCM sine-tone WAV (440 Hz, ~-10 dBFS)."""
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    sig = (0.4 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
    if channels > 1:
        sig = np.tile(np.expand_dims(sig, 1), (1, channels))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.ascontiguousarray(sig, dtype="<i2").tobytes())


def _setup(tmp_path: Path) -> tuple[Database, object, Config, Path]:
    """Create cfg + DB + a job with M1 outputs (proxy.mp4, audio.wav, probe.json)."""
    cfg = _cfg(tmp_path)
    db = Database(cfg)
    assert db.create_job(JOB_ID, "/tmp/sample-source.mp4", "client", "niche")
    job_dir = cfg.resolved_data_dir / "jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "proxy.mp4").write_bytes(b"\x00fake-proxy-bytes")
    _write_wav(job_dir / "audio.wav")
    (job_dir / "probe.json").write_text(
        json.dumps(
            {
                "duration_s": DURATION_S,
                "width": 640,
                "height": 360,
                "fps": 25.0,
                "vcodec": "h264",
                "acodec": "aac",
                "has_audio": True,
                "size_bytes": 100,
                "schema_version": 1,
            }
        )
    )
    conn = db.get_connection()
    return db, conn, cfg, job_dir


def _stages(db, job_id) -> dict:
    return {s["name"]: s for s in db.get_all_stages_for_job(job_id)}


class CountingSceneAdapter(FixtureSceneAdapter):
    def __init__(self):
        super().__init__(shots=None)
        self.detect_calls = 0

    def detect_shots(self, video_path: str, duration_s: float) -> list[Shot]:
        self.detect_calls += 1
        return [
            Shot(start=0.0, end=duration_s * 0.4),
            Shot(start=duration_s * 0.4, end=duration_s),
        ]


class CountingMediaAdapter(FixtureMediaAdapter):
    def __init__(self):
        super().__init__(silence_regions=[SilenceRegion(start=0.5, end=1.0)])
        self.silence_calls = 0

    def detect_silence(self, wav_path, duration_s, noise_db=None, min_dur=None):
        self.silence_calls += 1
        return [SilenceRegion(start=0.5, end=1.0)]


class BoomSceneAdapter(FixtureSceneAdapter):
    def detect_shots(self, video_path: str, duration_s: float) -> list[Shot]:
        raise RuntimeError("detector exploded")


def _media() -> CountingMediaAdapter:
    return CountingMediaAdapter()


# ---------------------------------------------------------------------------
# run_analysis: happy path
# ---------------------------------------------------------------------------


def test_run_analysis_creates_artifacts_and_stage_rows(tmp_path):
    db, conn, cfg, job_dir = _setup(tmp_path)
    scenes = CountingSceneAdapter()
    media = _media()

    summary = run_analysis(JOB_ID, cfg, conn, media, scenes)

    # Summary
    assert summary["job_id"] == JOB_ID
    assert summary["duration_s"] == DURATION_S
    assert summary["shots"] == 2
    assert summary["silence_regions"] == 1
    assert summary["energy_windows"] == int(DURATION_S / 0.5)
    assert summary["skipped"] == []

    # shots.json
    shots = json.loads((job_dir / "shots.json").read_text())
    assert shots["schema_version"] == 1
    assert shots["shots"][0]["start"] == 0.0
    assert shots["shots"][-1]["end"] == DURATION_S
    assert shots["shots"][0]["end"] == shots["shots"][1]["start"]
    for s in shots["shots"]:
        assert 0.0 <= s["start"] < s["end"] <= DURATION_S

    # silence.json
    silence = json.loads((job_dir / "silence.json").read_text())
    assert silence["schema_version"] == 1
    assert silence["noise_db"] == -35.0
    assert silence["min_dur"] == 0.5
    assert silence["regions"] == [{"start": 0.5, "end": 1.0}]

    # energy.json (sine tone: every 0.5s window loud, far above -120 dB floor)
    energy = json.loads((job_dir / "energy.json").read_text())
    assert energy["schema_version"] == 1
    assert energy["window_s"] == 0.5
    wins = energy["windows"]
    assert [w["start"] for w in wins] == [0.0, 0.5, 1.0]
    assert [w["end"] for w in wins] == [0.5, 1.0, 1.5]
    assert all(w["rms_db"] > -30.0 for w in wins)

    # Stage rows
    stages = _stages(db, JOB_ID)
    for name, artifact in (
        ("shots", "shots.json"),
        ("silence", "silence.json"),
        ("energy", "energy.json"),
    ):
        assert stages[name]["state"] == "done"
        assert stages[name]["artifact_path"] == str(job_dir / artifact)
        assert Path(stages[name]["artifact_path"]).exists()
        assert stages[name]["error"] is None
        assert stages[name]["started_at"] is not None
        assert stages[name]["finished_at"] is not None

    # Job returns to queued
    assert db.get_job(JOB_ID)["state"] == "queued"


def test_second_run_skips_completed_stages(tmp_path):
    db, conn, cfg, job_dir = _setup(tmp_path)
    scenes = CountingSceneAdapter()
    media = _media()

    run_analysis(JOB_ID, cfg, conn, media, scenes)
    before = {
        name: (job_dir / (name + ".json")).read_bytes()
        for name in ("shots", "silence", "energy")
    }

    summary = run_analysis(JOB_ID, cfg, conn, media, scenes)

    assert scenes.detect_calls == 1
    assert media.silence_calls == 1
    assert summary["skipped"] == ["shots", "silence", "energy"]
    assert summary["shots"] == 2
    assert summary["silence_regions"] == 1
    assert summary["energy_windows"] == 3
    after = {
        name: (job_dir / (name + ".json")).read_bytes()
        for name in ("shots", "silence", "energy")
    }
    assert before == after
    assert db.get_job(JOB_ID)["state"] == "queued"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_missing_proxy_fails_cleanly(tmp_path):
    db, conn, cfg, job_dir = _setup(tmp_path)
    (job_dir / "proxy.mp4").unlink()

    with pytest.raises(StageError, match="proxy.mp4"):
        run_analysis(JOB_ID, cfg, conn, _media(), CountingSceneAdapter())

    assert db.get_job(JOB_ID)["state"] == "failed"
    assert _stages(db, JOB_ID) == {}


def test_missing_audio_fails_cleanly(tmp_path):
    db, conn, cfg, job_dir = _setup(tmp_path)
    (job_dir / "audio.wav").unlink()

    with pytest.raises(StageError, match="audio.wav"):
        run_analysis(JOB_ID, cfg, conn, _media(), CountingSceneAdapter())

    assert db.get_job(JOB_ID)["state"] == "failed"
    assert _stages(db, JOB_ID) == {}


def test_missing_job_fails_cleanly(tmp_path):
    db, conn, cfg, job_dir = _setup(tmp_path)

    with pytest.raises(StageError, match="not found"):
        run_analysis("no-such-job", cfg, conn, _media(), CountingSceneAdapter())

    assert db.get_all_stages_for_job("no-such-job") == []
    assert db.get_job(JOB_ID)["state"] == "queued"


def test_adapter_exception_marks_stage_and_job_failed(tmp_path):
    db, conn, cfg, job_dir = _setup(tmp_path)

    with pytest.raises(StageError, match="detector exploded"):
        run_analysis(JOB_ID, cfg, conn, _media(), BoomSceneAdapter())

    stages = _stages(db, JOB_ID)
    assert stages["shots"]["state"] == "failed"
    assert "detector exploded" in stages["shots"]["error"]
    assert "silence" not in stages
    assert "energy" not in stages
    assert not (job_dir / "shots.json").exists()
    assert db.get_job(JOB_ID)["state"] == "failed"


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2)])
def test_energy_stage_rejects_wrong_wav_format(tmp_path, rate, channels):
    db, conn, cfg, job_dir = _setup(tmp_path)
    _write_wav(job_dir / "audio.wav", seconds=DURATION_S, rate=rate, channels=channels)

    with pytest.raises(StageError, match="16kHz|mono PCM"):
        run_analysis(JOB_ID, cfg, conn, _media(), CountingSceneAdapter())

    stages = _stages(db, JOB_ID)
    assert stages["shots"]["state"] == "done"
    assert stages["silence"]["state"] == "done"
    assert stages["energy"]["state"] == "failed"
    assert db.get_job(JOB_ID)["state"] == "failed"


# ---------------------------------------------------------------------------
# Script (scripts/analyze_job.py) with monkeypatched adapter factories
# ---------------------------------------------------------------------------


def _load_analyze_job_script() -> object:
    path = REPO_ROOT / "scripts" / "analyze_job.py"
    spec = importlib.util.spec_from_file_location(
        "clipper_analyze_job_script", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analyze_job_script_success_with_monkeypatched_adapters(
    tmp_path, monkeypatch, capsys
):
    db, conn, cfg, job_dir = _setup(tmp_path)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_analyze_job_script()

    # The task's contract: get_scene_adapter / get_adapter are monkeypatchable.
    monkeypatch.setattr(scenes_mod, "get_scene_adapter", lambda: CountingSceneAdapter())
    monkeypatch.setattr(media_mod, "get_adapter", lambda: _media())

    rc = script.main(JOB_ID)

    assert rc == 0
    out = capsys.readouterr().out
    assert "job:" in out and JOB_ID in out
    assert "shots:" in out and "silence regions: 1" in out
    assert "energy windows:" in out and " 3" in out
    assert str(job_dir / "shots.json") in out
    assert str(job_dir / "silence.json") in out
    assert str(job_dir / "energy.json") in out


def test_analyze_job_script_missing_job_exits_1(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    Database(cfg)  # initialize the DB file
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))
    script = _load_analyze_job_script()

    rc = script.main("no-such-job")

    assert rc == 1
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Real integration (ffmpeg + scenedetect + opencv must be installed)
# ---------------------------------------------------------------------------

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
_HAVE_SCENEDETECT = (
    importlib.util.find_spec("scenedetect") is not None
    and importlib.util.find_spec("cv2") is not None
)


@pytest.mark.skipif(
    not (HAVE_FFMPEG and _HAVE_SCENEDETECT),
    reason="requires ffmpeg binary, scenedetect and opencv (headless)",
)
def test_integration_ingest_then_analysis(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    db = Database(cfg)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(cfg.resolved_data_dir))

    from clipper.adapters.media import FFmpegMediaAdapter
    from clipper.stages import run_ingest

    src = REPO_ROOT / "tests" / "fixtures" / "sample.mp4"
    media = FFmpegMediaAdapter(timeout=300)

    conn = db.get_connection()
    job_id, _is_new, _is_retry = run_ingest(
        str(src), "client", "niche", cfg, conn, media
    )
    assert db.get_job(job_id)["state"] != "failed"

    summary = run_analysis(job_id, cfg, conn, media, scenes_mod.get_scene_adapter())
    assert summary["shots"] >= 1
    assert summary["energy_windows"] >= 1
    assert db.get_job(job_id)["state"] == "queued"

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    duration = json.loads((job_dir / "probe.json").read_text())["duration_s"]
    assert duration > 0

    # Shots: >= 1, ordered, contiguous, covering [0, duration]
    shots = json.loads((job_dir / "shots.json").read_text())
    assert shots["schema_version"] == 1
    seq = shots["shots"]
    assert len(seq) >= 1
    assert seq[0]["start"] == 0.0
    assert abs(seq[-1]["end"] - duration) < 0.05
    for a, b in zip(seq, seq[1:], strict=False):
        assert a["end"] > 0 and abs(a["end"] - b["start"]) < 0.05

    # Silence: valid artifact (regions may be empty for the tone fixture)
    silence = json.loads((job_dir / "silence.json").read_text())
    assert silence["schema_version"] == 1
    for r in silence["regions"]:
        assert 0.0 <= r["start"] < r["end"] <= duration + 0.05

    # Energy: windows cover the audio; sine tone -> rms_db > -120 everywhere
    energy = json.loads((job_dir / "energy.json").read_text())
    assert energy["schema_version"] == 1
    wins = energy["windows"]
    assert wins and wins[0]["start"] == 0.0
    assert sum(w["end"] - w["start"] for w in wins) >= duration - 0.1
    assert all(w["rms_db"] > -120.0 for w in wins)
    assert max(w["rms_db"] for w in wins) > -60.0
