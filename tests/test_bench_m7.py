"""
M7f bench.json contract tests (docs/spec/m7.md M7f) — CPU-only, no GPU:

- `scripts/bench_m7.py` `_bench_payload` produces EXACTLY the required
  bench.json fields with sane types per clip,
- the payload is DETERMINISTIC for the same inputs,
- positional slicing works for MULTI-clip jobs,
- `main()` fails CLEANLY (exit 1, no traceback) when `nvidia-smi` is
  absent and when the stage yields nothing to measure.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clipper.config import Config
from clipper.models import RenderArtifact
from clipper.stages import StageError

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO_ROOT / "scripts" / "bench_m7.py"


def _load_bench() -> "importlib.util.ModuleType":
    spec = importlib.util.spec_from_file_location(
        "bench_m7_under_test", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Bench = _load_bench()

CFG = Config()

FIELDS = {
    "clip_id": str,
    "mezzanine_seconds": float,
    "compose_seconds": float,
    "total_seconds": float,
    "decode_fps": float,
    "encode_fps": float,
    "output_fps": float,
    "frames_processed": int,
    "peak_gpu0_util": int,
    "nvenc_session_seen": bool,
    "encoder": str,
    "hwaccel": str,
}


def _clip_dir(tmp_path: Path, job_id: str, clip_id: str) -> Path:
    clip_dir = tmp_path / "jobs" / job_id / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "final.mp4").write_bytes(b"placeholder-final")
    artifact = RenderArtifact(
        clip_id=clip_id,
        mezzanine_path=str(clip_dir / "mezzanine.mp4"),
        final_path=str(clip_dir / "final.mp4"),
        audio_path=str(clip_dir / "audio.m4a"),
        captions_path=str(clip_dir / "captions.ass"),
        thumbnail_path=str(clip_dir / "thumbnail.jpg"),
        encoder="h264_nvenc",
        hwaccel="cuda",
        width=1080,
        height=1920,
        fps=29.97,
        duration_s=15.0,
        render_seconds=9.0,
    )
    (clip_dir / "render.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return clip_dir


def _timed() -> "Bench.TimedRender":
    timed = Bench.TimedRender(impl=None)  # only its lists are read
    for _ in range(2):
        timed.mezzanine_seconds.append(4.2)
        timed.compose_seconds.append(1.1)
    return timed


def _sampler() -> "Bench.GpuSampler":
    sampler = Bench.GpuSampler()
    sampler.peak_gpu = 83
    sampler.peak_encoder = 61
    sampler.samples = 40
    return sampler


def test_bench_payload_has_exact_required_fields(tmp_path, monkeypatch):
    job = "b7"
    clip_dir = _clip_dir(tmp_path, job, "b7-clip01")
    monkeypatch.setattr(Bench, "_probe_frames", lambda path: 450)
    rec = {
        "clip_id": "b7-clip01",
        "elapsed_seconds": 6.3,
        "mezzanine_path": str(clip_dir / "mezzanine.mp4"),
        "frames": 450,
        "fps": 29.97,
        "encoder": "h264_nvenc",
        "hwaccel": "cuda",
    }
    payload = Bench._bench_payload(0, rec, _timed(), clip_dir, _sampler())
    assert set(payload) == set(FIELDS), \
        "bench.json must have EXACTLY the required M7f fields"
    for field, typ in FIELDS.items():
        assert isinstance(payload[field], typ), (
            f"bench.json field {field!r} must be {typ.__name__}, "
            f"got {type(payload[field]).__name__}")
    assert payload["frames_processed"] == 450
    assert payload["decode_fps"] == pytest.approx(450 / 4.2, abs=0.01)
    assert payload["encode_fps"] == pytest.approx(450 / 1.1, abs=0.01)
    assert payload["output_fps"] == pytest.approx(29.97)
    assert payload["mezzanine_seconds"] == pytest.approx(4.2)
    assert payload["compose_seconds"] == pytest.approx(1.1)
    assert payload["total_seconds"] == pytest.approx(6.3)
    assert payload["peak_gpu0_util"] == 83
    assert payload["nvenc_session_seen"] is True
    assert payload["encoder"] == "h264_nvenc"
    assert payload["hwaccel"] == "cuda"
    # serialisable
    json.loads(json.dumps(payload))


def test_bench_payload_is_deterministic(tmp_path, monkeypatch):
    job = "b7d"
    clip_dir = _clip_dir(tmp_path, job, "b7d-clip01")
    monkeypatch.setattr(Bench, "_probe_frames", lambda path: 300)
    rec = {"clip_id": "b7d-clip01", "elapsed_seconds": 5.0}

    p1 = Bench._bench_payload(0, rec, _timed(), clip_dir, _sampler())
    p2 = Bench._bench_payload(0, rec, _timed(), clip_dir, _sampler())
    assert p1 == p2


def test_bench_payload_multi_clip_positional_slices(tmp_path, monkeypatch):
    job = "b7m"
    dirs = [_clip_dir(tmp_path, job, f"{job}-clip0{i}") for i in range(1, 3)]
    monkeypatch.setattr(Bench, "_probe_frames", lambda path: 450)
    recs = [{"clip_id": f"{job}-clip0{i}", "elapsed_seconds": 5.0 + i}
            for i in range(1, 3)]
    timed = Bench.TimedRender(impl=None)
    for extra in (0.0, 0.5):  # one (mezzanine, compose) pair per clip
        timed.mezzanine_seconds.append(4.2 + extra)
        timed.compose_seconds.append(1.1 + extra)

    p0 = Bench._bench_payload(0, recs[0], timed, dirs[0], _sampler())
    p1 = Bench._bench_payload(1, recs[1], timed, dirs[1], _sampler())
    assert p0["mezzanine_seconds"] == pytest.approx(4.2)
    assert p1["mezzanine_seconds"] == pytest.approx(4.7)
    assert p0["clip_id"] == f"{job}-clip01"
    assert p1["clip_id"] == f"{job}-clip02"


def test_bench_main_clean_error_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert Bench.main(["b7"]) == 1


def test_bench_main_clean_error_when_stage_yields_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(Bench, "run_render", _raise_stage_error)

    class _FakeConn:
        def close(self):
            pass

    class _FakeDB:
        def __init__(self, *a, **k):
            pass

        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr(Bench, "Database", _FakeDB)
    monkeypatch.setattr(Bench, "get_render_adapter", lambda cfg: object())
    assert Bench.main(["nope-not-a-real-job"]) == 1


def _raise_stage_error(*a, **k):
    raise StageError("job does not exist")
