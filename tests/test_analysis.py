"""
M3a analysis foundations tests — CPU only.

No ffmpeg binary, no scenedetect, no opencv required.
"""

import json
import math

import numpy as np
import pytest

from clipper.adapters.media import FixtureMediaAdapter
from clipper.adapters.scenes import (
    SHOT_THRESHOLD,
    FixtureSceneAdapter,
)
from clipper.audio_analysis import (
    ENERGY_WINDOW_S,
    SILENCE_MIN_DUR,
    SILENCE_NOISE_DB,
    parse_silence_log,
    rms_windows,
)
from clipper.models import (
    EnergyArtifact,
    EnergyWindow,
    Shot,
    ShotsArtifact,
    SilenceArtifact,
    SilenceRegion,
)

# ---------------------------------------------------------------------------
# Contracts / JSON round-trip
# ---------------------------------------------------------------------------


def test_shots_artifact_json_round_trip():
    artifact = ShotsArtifact(
        shots=[Shot(start=0.0, end=5.25), Shot(start=5.25, end=10.0)]
    )
    assert json.loads(artifact.model_dump_json()) == {
        "schema_version": 1,
        "shots": [
            {"start": 0.0, "end": 5.25},
            {"start": 5.25, "end": 10.0},
        ],
    }
    restored = ShotsArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact


def test_silence_artifact_json_round_trip():
    artifact = SilenceArtifact(
        regions=[SilenceRegion(start=2.0, end=3.5)]
    )
    assert json.loads(artifact.model_dump_json()) == {
        "schema_version": 1,
        "noise_db": -35.0,
        "min_dur": 0.5,
        "regions": [{"start": 2.0, "end": 3.5}],
    }
    restored = SilenceArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact


def test_energy_artifact_json_round_trip():
    artifact = EnergyArtifact(
        windows=[EnergyWindow(start=0.0, end=0.5, rms_db=-120.0)]
    )
    assert json.loads(artifact.model_dump_json()) == {
        "schema_version": 1,
        "window_s": 0.5,
        "windows": [{"start": 0.0, "end": 0.5, "rms_db": -120.0}],
    }
    restored = EnergyArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact


def test_contract_defaults_match_spec():
    assert SILENCE_NOISE_DB == -35.0
    assert SILENCE_MIN_DUR == 0.5
    assert ENERGY_WINDOW_S == 0.5
    assert SHOT_THRESHOLD == 27.0
    assert SilenceArtifact(regions=[]).noise_db == -35.0
    assert SilenceArtifact(regions=[]).min_dur == 0.5
    assert EnergyArtifact(windows=[]).window_s == 0.5
    assert ShotsArtifact(shots=[]).schema_version == 1
    assert SilenceArtifact(regions=[]).schema_version == 1
    assert EnergyArtifact(windows=[]).schema_version == 1


# ---------------------------------------------------------------------------
# parse_silence_log
# ---------------------------------------------------------------------------


def _ffmpeg_log(*pairs: tuple[float, float | None]) -> str:
    lines = []
    for start, end in pairs:
        lines.append(f"[silencedetect#0:0 @ 0x1] silence_start: {start}")
        if end is not None:
            lines.append(
                f"[silencedetect#0:0 @ 0x1] silence_end: {end} "
                f"| silence_duration: {end - start:.4f}"
            )
    return "\n".join(lines)


def test_parse_silence_log_multiple_regions():
    log = "\n".join(
        [
            "ffmpeg version 7.0",
            "Input #0, wav, from 'audio.wav':",
            _ffmpeg_log((1.25, 2.75), (4.1, 5.35), (9.0, 9.9)),
        ]
    )
    regions = parse_silence_log(log, duration_s=12.0)
    assert [(r.start, r.end) for r in regions] == [
        (1.25, 2.75),
        (4.1, 5.35),
        (9.0, 9.9),
    ]
    assert all(isinstance(r, SilenceRegion) for r in regions)


def test_parse_silence_log_trailing_unpaired_start():
    log = _ffmpeg_log((0.5, 1.5), (8.75, None))
    regions = parse_silence_log(log, duration_s=10.0)
    assert [(r.start, r.end) for r in regions] == [(0.5, 1.5), (8.75, 10.0)]


def test_parse_silence_log_no_silence():
    log = "Input #0, wav, from 'audio.wav':\nStream #0: Audio: pcm_s16le\n"
    assert parse_silence_log(log, duration_s=5.0) == []


def test_parse_silence_log_malformed_lines():
    log = "\n".join(
        [
            "garbage line with no markers",
            "silence_start: not-a-number",
            # end before any start: ignore
            "silence_end: 3.0 | silence_duration: 1.0",
            "silence_start: 1.0",
            "silence_end: 2.0 | silence_duration: 1.0",
            # second start while one is open: ignore
            "silence_start: 4.0",
            "silence_end: 5.5 | silence_duration: 1.5",
        ]
    )
    regions = parse_silence_log(log, duration_s=10.0)
    assert [(r.start, r.end) for r in regions] == [(1.0, 2.0), (4.0, 5.5)]


def test_parse_silence_log_clamping():
    log = _ffmpeg_log((-0.5, 1.0), (9.8, 110.0))
    regions = parse_silence_log(log, duration_s=10.0)
    assert [(r.start, r.end) for r in regions] == [(0.0, 1.0), (9.8, 10.0)]


def test_parse_silence_log_region_beyond_duration_dropped():
    # start beyond duration -> zero length after clamping -> dropped
    log = _ffmpeg_log((95.0, 96.0))
    assert parse_silence_log(log, duration_s=10.0) == []


# ---------------------------------------------------------------------------
# rms_windows
# ---------------------------------------------------------------------------


def test_rms_windows_zeros_hit_floor():
    windows = rms_windows(np.zeros(8000, dtype=np.float32), 16000)
    assert len(windows) == 1
    assert windows[0].start == 0.0
    assert windows[0].end == 0.5
    assert windows[0].rms_db == -120.0


def test_rms_windows_sine_amplitude_05():
    fs = 16000
    t = np.arange(fs * 3, dtype=np.float64) / fs
    samples = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    windows = rms_windows(samples, fs)
    assert len(windows) == 6  # 3s of audio at 0.5s windows
    expected = 20.0 * math.log10(0.5 / math.sqrt(2.0))  # ~ -9.03 dB
    for w in windows:
        assert w.rms_db == pytest.approx(expected, abs=0.15)


def test_rms_windows_tiling():
    fs = 16000
    n = int(2.5 * fs)
    windows = rms_windows(np.ones(n, dtype=np.float64), fs)
    assert [(w.start, w.end) for w in windows] == [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
    ]
    # Full-scale ones: rms == 1.0 -> 0.0 dB
    for w in windows:
        assert w.rms_db == 0.0


def test_rms_windows_partial_final_window_included():
    fs = 16000
    n = int(1.3 * fs)  # 1.0s full + 0.3s partial (>= 0.1s -> included)
    windows = rms_windows(np.zeros(n, dtype=np.float64), fs)
    assert [(w.start, w.end) for w in windows] == [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.3),
    ]


def test_rms_windows_partial_final_window_skipped():
    fs = 16000
    n = int(1.05 * fs)  # 0.05s partial (< 0.1s -> skipped)
    windows = rms_windows(np.zeros(n, dtype=np.float64), fs)
    assert [(w.start, w.end) for w in windows] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]


def test_rms_windows_empty_input():
    assert rms_windows(np.array([], dtype=np.float64), 16000) == []


def test_rms_windows_accepts_python_list():
    # 12000 samples @ 16000 Hz = 0.75s -> one full + one 0.25s partial window
    samples = [0.0, 1.0] * 6000
    windows = rms_windows(samples, 16000)
    assert [(w.start, w.end) for w in windows] == [(0.0, 0.5), (0.5, 0.75)]
    for w in windows:
        assert w.rms_db == pytest.approx(20.0 * math.log10(0.70710678), abs=0.01)


# ---------------------------------------------------------------------------
# Fixture adapters
# ---------------------------------------------------------------------------


def test_fixture_media_adapter_detect_silence_default():
    regions = FixtureMediaAdapter().detect_silence("audio.wav", duration_s=10.0)
    assert len(regions) == 1
    assert all(isinstance(r, SilenceRegion) for r in regions)
    r = regions[0]
    assert 0.0 <= r.start < r.end <= 10.0


def test_fixture_media_adapter_detect_silence_configurable():
    canned = [
        SilenceRegion(start=1.0, end=1.8),
        SilenceRegion(start=5.0, end=7.2),
    ]
    adapter = FixtureMediaAdapter(silence_regions=canned)
    regions = adapter.detect_silence(
        "audio.wav", duration_s=20.0, noise_db=-40.0, min_dur=0.7
    )
    assert regions == canned


def test_fixture_scene_adapter_default_single_full_shot():
    shots = FixtureSceneAdapter().detect_shots("proxy.mp4", 12.5)
    assert shots == [Shot(start=0.0, end=12.5)]
    assert all(isinstance(s, Shot) for s in shots)


def test_fixture_scene_adapter_configurable_shots():
    canned = [
        Shot(start=0.0, end=3.0),
        Shot(start=3.0, end=8.25),
        Shot(start=8.25, end=10.0),
    ]
    adapter = FixtureSceneAdapter(shots=canned)
    assert adapter.detect_shots("proxy.mp4", 10.0) == canned
