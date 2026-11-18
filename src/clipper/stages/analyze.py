"""
Analysis stage implementation for the clipper pipeline (M3b).

Runs the three analysis stages for a job, in order:

1. ``shots``   — ``SceneAdapter.detect_shots(proxy.mp4)`` -> ``shots.json``
2. ``silence`` — ``MediaAdapter.detect_silence(audio.wav)`` -> ``silence.json``
3. ``energy``  — stdlib ``wave`` read + ``rms_windows`` -> ``energy.json``

A stage is skipped only when the DB stage state is ``done`` AND the
referenced artifact file exists AND the artifact JSON has
``schema_version == 1``. On success the job returns to ``queued``; on
failure the current stage is marked ``failed`` (with the error text),
the job is set to ``failed``, and a clean ``StageError`` is raised.

Contracts and rules: docs/spec/m3.md. CPU-only; heavy libraries (numpy
allowed at module level; scenedetect/cv2 live behind the scenes adapter).
"""

import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from ..audio_analysis import rms_windows
from ..config import Config
from ..models import (
    EnergyArtifact,
    Shot,
    ShotsArtifact,
    SilenceArtifact,
)
from . import StageError

_SCHEMA_VERSION = 1
_EXPECTED_RATE = 16000
_EXPECTED_CHANNELS = 1
_EXPECTED_SAMPWIDTH = 2  # 16-bit PCM


def _now() -> str:
    return datetime.now().isoformat()


def _set_job_state(conn, job_id: str, state: str) -> None:
    conn.execute(
        "UPDATE jobs SET state = ?, updated_at = ? WHERE job_id = ?",
        (state, _now(), job_id),
    )
    conn.commit()


def _stage_row(conn, job_id: str, name: str):
    cur = conn.execute(
        "SELECT state, artifact_path FROM stages WHERE job_id = ? AND name = ?",
        (job_id, name),
    )
    return cur.fetchone()


def _start_stage(conn, job_id: str, name: str) -> None:
    conn.execute(
        """
        INSERT INTO stages
            (job_id, name, state, started_at, finished_at, artifact_path, error)
        VALUES (?, ?, 'running', ?, NULL, NULL, NULL)
        ON CONFLICT(job_id, name) DO UPDATE SET
            state = 'running',
            started_at = excluded.started_at,
            finished_at = NULL,
            artifact_path = NULL,
            error = NULL
        """,
        (job_id, name, _now()),
    )
    conn.commit()


def _finish_stage(
    conn,
    job_id: str,
    name: str,
    artifact_path: str | None = None,
    error: str | None = None,
) -> None:
    if error is not None:
        conn.execute(
            """
            UPDATE stages
            SET state = 'failed', error = ?, artifact_path = NULL, finished_at = ?
            WHERE job_id = ? AND name = ?
            """,
            (error, _now(), job_id, name),
        )
    else:
        conn.execute(
            """
            UPDATE stages
            SET state = 'done', artifact_path = ?, error = NULL, finished_at = ?
            WHERE job_id = ? AND name = ?
            """,
            (artifact_path, _now(), job_id, name),
        )
    conn.commit()


def _is_done(conn, job_id: str, name: str) -> tuple[bool, str | None]:
    """Return (skippable, artifact_path) for a stage.

    Skippable only when: state == 'done' AND the recorded artifact file
    exists AND its JSON has schema_version == 1.
    """
    row = _stage_row(conn, job_id, name)
    if row is None:
        return False, None
    state, artifact_path = row[0], row[1]
    if state != "done" or not artifact_path:
        return False, None
    path = Path(artifact_path)
    if not path.exists():
        return False, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, None
    if data.get("schema_version") != _SCHEMA_VERSION:
        return False, None
    return True, str(path)


def _write_artifact(path: Path, artifact) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact.model_dump(), f, indent=2)
    except OSError as e:
        raise StageError(f"Failed to write artifact {path}: {e}") from e


def _read_duration(job_dir: Path) -> float:
    """Read the video duration from the M1 probe.json (no re-probing)."""
    probe_path = job_dir / "probe.json"
    if not probe_path.exists():
        raise StageError(
            f"probe.json not found at {probe_path} (M1 probe stage incomplete)"
        )
    try:
        with open(probe_path, encoding="utf-8") as f:
            data = json.load(f)
        duration = float(data["duration_s"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise StageError(f"Failed to read duration from {probe_path}: {e}") from e
    if not duration > 0:
        raise StageError(
            f"Invalid duration_s in {probe_path}: {data.get('duration_s')}"
        )
    return duration


def _read_wav_samples(wav_path: Path) -> np.ndarray:
    """Read a 16kHz mono pcm_s16le WAV with stdlib wave.

    Returns float64 samples normalized to [-1, 1]. Raises a clean
    ``StageError`` for any other format (wrong rate/channels/bit-depth
    or compressed WAV).
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != _EXPECTED_CHANNELS:
                raise StageError(
                    f"audio.wav must be mono PCM (1 channel), got "
                    f"{wf.getnchannels()} channels"
                )
            if wf.getsampwidth() != _EXPECTED_SAMPWIDTH:
                raise StageError(
                    "audio.wav must be 16-bit PCM (pcm_s16le, sampwidth 2), "
                    f"got sampwidth {wf.getsampwidth()}"
                )
            if wf.getframerate() != _EXPECTED_RATE:
                raise StageError(
                    f"audio.wav must be 16kHz, got {wf.getframerate()} Hz"
                )
            if wf.getcomptype() != "NONE":
                raise StageError(
                    f"audio.wav must be uncompressed PCM, got "
                    f"'{wf.getcomptype()}'"
                )
            frames = wf.readframes(wf.getnframes())
    except StageError:
        raise
    except (wave.Error, OSError) as e:
        raise StageError(f"Failed to read WAV {wav_path}: {e}") from e

    if len(frames) % 2:
        raise StageError(f"audio.wav PCM payload is {len(frames)} bytes (odd)")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    return samples


def run_analysis(
    job_id: str,
    cfg: Config,
    conn,
    media,
    scenes,
) -> dict:
    """Run the shots, silence and energy analysis stages for a job.

    Args:
        job_id: The job ID (must exist in the DB).
        cfg: Configuration object (data_dir).
        conn: sqlite3 connection over the clipper database.
        media: A ``MediaAdapter`` (provides detect_silence).
        scenes: A ``SceneAdapter`` (provides detect_shots).

    Returns:
        Summary dict: job_id, duration_s, counts per artifact, artifact
        paths, and the list of skipped stage names.

    Raises:
        StageError: Clean, single-line error on any failure (missing job,
            missing M1 inputs, adapter failure, bad WAV format, ...). The
            failing stage and the job are marked failed in the DB.
    """
    row = conn.execute(
        "SELECT job_id FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    proxy_path = job_dir / "proxy.mp4"
    wav_path = job_dir / "audio.wav"

    summary = {
        "job_id": job_id,
        "duration_s": None,
        "shots": 0,
        "silence_regions": 0,
        "energy_windows": 0,
        "skipped": [],
        "artifacts": {},
    }
    current_stage: str | None = None

    try:
        if not proxy_path.exists():
            raise StageError(
                f"Missing M1 input {proxy_path} — run the ingest stage first"
            )
        if not wav_path.exists():
            raise StageError(
                f"Missing M1 input {wav_path} — run the ingest stage first"
            )

        duration = _read_duration(job_dir)
        summary["duration_s"] = duration
        _set_job_state(conn, job_id, "running")

        # ------------------------------------------------------------------
        # Stage 1: shots (proxy.mp4 -> shots.json)
        # ------------------------------------------------------------------
        current_stage = "shots"
        shots_path = job_dir / "shots.json"
        skip, stored = _is_done(conn, job_id, "shots")
        if skip:
            summary["skipped"].append("shots")
            with open(stored, encoding="utf-8") as f:
                summary["shots"] = len(json.load(f).get("shots", []))
            summary["artifacts"]["shots"] = stored
        else:
            _start_stage(conn, job_id, "shots")
            try:
                detected = scenes.detect_shots(str(proxy_path), duration)
            except Exception as e:
                raise StageError(f"Shot detection failed: {e}") from e

            shots = [Shot(start=s.start, end=s.end) for s in (detected or [])]
            if not shots:
                shots = [Shot(start=0.0, end=duration)]
            artifact = ShotsArtifact(schema_version=_SCHEMA_VERSION, shots=shots)
            _write_artifact(shots_path, artifact)
            _finish_stage(conn, job_id, "shots", str(shots_path))
            summary["shots"] = len(shots)
            summary["artifacts"]["shots"] = str(shots_path)

        # ------------------------------------------------------------------
        # Stage 2: silence (audio.wav -> silence.json)
        # ------------------------------------------------------------------
        current_stage = "silence"
        silence_path = job_dir / "silence.json"
        skip, stored = _is_done(conn, job_id, "silence")
        if skip:
            summary["skipped"].append("silence")
            with open(stored, encoding="utf-8") as f:
                summary["silence_regions"] = len(
                    json.load(f).get("regions", [])
                )
            summary["artifacts"]["silence"] = stored
        else:
            _start_stage(conn, job_id, "silence")
            try:
                regions = media.detect_silence(str(wav_path), duration)
            except Exception as e:
                raise StageError(f"Silence detection failed: {e}") from e

            artifact = SilenceArtifact(
                schema_version=_SCHEMA_VERSION, regions=list(regions or [])
            )
            _write_artifact(silence_path, artifact)
            _finish_stage(conn, job_id, "silence", str(silence_path))
            summary["silence_regions"] = len(artifact.regions)
            summary["artifacts"]["silence"] = str(silence_path)

        # ------------------------------------------------------------------
        # Stage 3: energy (audio.wav -> energy.json)
        # ------------------------------------------------------------------
        current_stage = "energy"
        energy_path = job_dir / "energy.json"
        skip, stored = _is_done(conn, job_id, "energy")
        if skip:
            summary["skipped"].append("energy")
            with open(stored, encoding="utf-8") as f:
                summary["energy_windows"] = len(json.load(f).get("windows", []))
            summary["artifacts"]["energy"] = stored
        else:
            _start_stage(conn, job_id, "energy")
            samples = _read_wav_samples(wav_path)
            windows = rms_windows(samples, _EXPECTED_RATE)

            artifact = EnergyArtifact(
                schema_version=_SCHEMA_VERSION, windows=windows
            )
            _write_artifact(energy_path, artifact)
            _finish_stage(conn, job_id, "energy", str(energy_path))
            summary["energy_windows"] = len(windows)
            summary["artifacts"]["energy"] = str(energy_path)

        _set_job_state(conn, job_id, "queued")
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"Analysis failed: {e}"
        )
        if current_stage is not None:
            try:
                _finish_stage(conn, job_id, current_stage, error=str(clean))
            except Exception:
                pass
        try:
            _set_job_state(conn, job_id, "failed")
        except Exception:
            pass
        if clean is e:
            raise
        raise clean from e
