"""
Ingestion pipeline for clipper.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime

from ..adapters.media import MediaAdapter, MediaError
from ..config import Config
from . import StageError


def compute_content_hash(path: str, max_bytes: int = 64 * 1024 * 1024) -> str:
    """
    Compute a content hash for the file based on size and first max_bytes of file.

    Args:
        path: Path to the file
        max_bytes: Maximum bytes to read from beginning of file

    Returns:
        Content hash string in format: "{size}:{sha256(first_max_bytes)}"
    """
    if not os.path.exists(path):
        raise StageError(f"Source file does not exist: {path}")

    size = os.path.getsize(path)

    # Read first max_bytes of file
    with open(path, 'rb') as f:
        data = f.read(max_bytes)

    # Calculate SHA256 hash of the data
    hash_obj = hashlib.sha256(data)
    hash_hex = hash_obj.hexdigest()

    return f"{size}:{hash_hex}"

def run_ingest(
    source: str,
    client: str,
    niche: str,
    cfg: Config,
    conn,
    media: MediaAdapter
) -> tuple[str, bool, bool]:
    """
    Run the ingest stage with idempotency checks.

    Returns:
        (job_id, is_new_job, should_retry)
    """
    # Check if source file exists and is readable
    if not os.path.exists(source):
        raise StageError(f"Source file does not exist: {source}")

    if not os.access(source, os.R_OK):
        raise StageError(f"Source file not readable: {source}")

    # Compute content hash
    try:
        content_hash = compute_content_hash(source)
    except Exception as e:
        raise StageError(f"Failed to compute content hash: {e}") from e

    # Check if we have a job with this content hash and state != failed
    job_id = str(uuid.uuid4().hex)
    is_retry = False
    try:
        # Query for existing job with the content hash
        cursor = conn.cursor()
        cursor.execute("""
            SELECT job_id, state FROM jobs WHERE content_hash = ?
        """, (content_hash,))
        result = cursor.fetchone()

        if result:
            existing_id, state = result

            # If it's already done or running, return existing job
            if state != 'failed':
                return existing_id, False, False

            # If it failed, re-use the job and re-run the pipeline below
            job_id = existing_id
            is_retry = True
            timestamp = datetime.now().isoformat()
            cursor.execute("DELETE FROM stages WHERE job_id = ?", (job_id,))
            cursor.execute("""
                UPDATE jobs SET state = 'queued', client = ?, niche = ?, updated_at = ?
                WHERE job_id = ?
            """, (client, niche, timestamp, job_id))
            conn.commit()

    except Exception as e:
        raise StageError(f"Database query failed: {e}") from e

    # Get data directory
    data_dir = cfg.resolved_data_dir

    # Create job directory
    job_dir = data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create the job in database
    try:
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()

        # Insert initial job with content hash
        cursor.execute("""
            INSERT OR IGNORE INTO jobs
            (
                job_id, source_path, client, niche, state,
                content_hash, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
        """, (
            job_id, os.path.abspath(source), client, niche,
            content_hash, timestamp, timestamp
        ))

        if cursor.rowcount == 0 and not is_retry:
            # Job already exists (and we're not re-running a failed one)
            cursor.execute(
                "SELECT job_id FROM jobs WHERE content_hash = ?", (content_hash,)
            )
            existing_job_id = cursor.fetchone()[0]
            return existing_job_id, False, False

        conn.commit()
    except Exception as e:
        raise StageError(f"Failed to create job in database: {e}") from e

    def check_stage_completed(stage_name: str) -> str | None:
        """
        Check if a stage is already completed.
        Returns artifact path if completed, None otherwise.
        """
        cursor = conn.cursor()
        cursor.execute("""
            SELECT artifact_path FROM stages
            WHERE job_id = ? AND name = ? AND state = 'completed'
        """, (job_id, stage_name))
        result = cursor.fetchone()
        return result[0] if result else None

    def mark_stage_failed(stage_name: str, error: str):
        """Mark a stage as failed."""
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO stages
                (job_id, name, state, error, finished_at)
                VALUES (?, ?, 'failed', ?, ?)
            """, (job_id, stage_name, error, timestamp))
            conn.commit()
        except Exception as e:
            raise StageError(f"Failed to mark stage {stage_name} as failed: {e}") from e

    def mark_stage_completed(stage_name: str, artifact_path: str = None):
        """Mark a stage as completed."""
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO stages
                (job_id, name, state, artifact_path, finished_at)
                VALUES (?, ?, 'completed', ?, ?)
            """, (job_id, stage_name, artifact_path, timestamp))
            conn.commit()
        except Exception as e:
            raise StageError(
                f"Failed to mark stage {stage_name} as completed: {e}"
            ) from e

    # Run stages in order
    try:
        # Stage 1: Ingest
        ingest_artifact_path = str(job_dir / "ingest.json")
        ingest_data = {
            "job_id": job_id,
            "source_path": source,
            "content_hash": content_hash,
            "client": client,
            "niche": niche,
            "schema_version": 1
        }

        with open(ingest_artifact_path, 'w') as f:
            json.dump(ingest_data, f, indent=2)

        mark_stage_completed("ingest", ingest_artifact_path)

        # Stage 2: Probe
        probe_artifact_path = str(job_dir / "probe.json")

        try:
            probe_result = media.probe(source)
        except MediaError as e:
            raise StageError(f"Failed to probe source: {e}") from e

        # Verify audio exists
        if not probe_result.has_audio:
            raise StageError("Source file has no audio stream")

        probe_data = {
            "duration_s": probe_result.duration_s,
            "width": probe_result.width,
            "height": probe_result.height,
            "fps": probe_result.fps,
            "vcodec": probe_result.vcodec,
            "acodec": probe_result.acodec,
            "has_audio": probe_result.has_audio,
            "size_bytes": probe_result.size_bytes,
            "schema_version": 1
        }

        with open(probe_artifact_path, 'w') as f:
            json.dump(probe_data, f, indent=2)

        mark_stage_completed("probe", probe_artifact_path)

        # Stage 3: Proxy
        proxy_artifact_path = str(job_dir / "proxy.json")
        proxy_video_path = str(job_dir / "proxy.mp4")

        try:
            proxy_info = media.make_proxy(source, proxy_video_path)
        except MediaError as e:
            raise StageError(f"Failed to create proxy: {e}") from e

        proxy_data = {
            "width": proxy_info.width,
            "height": proxy_info.height,
            "fps": proxy_info.fps,
            "path": proxy_info.path,
            "schema_version": 1
        }

        with open(proxy_artifact_path, 'w') as f:
            json.dump(proxy_data, f, indent=2)

        mark_stage_completed("proxy", proxy_artifact_path)

        # Stage 4: Audio extraction
        audio_artifact_path = str(job_dir / "audio.json")
        audio_wav_path = str(job_dir / "audio.wav")

        try:
            audio_info = media.extract_wav(source, audio_wav_path)
        except MediaError as e:
            raise StageError(f"Failed to extract audio: {e}") from e

        audio_data = {
            "sample_rate": audio_info.sample_rate,
            "channels": audio_info.channels,
            "duration_s": audio_info.duration_s,
            "path": audio_info.path,
            "schema_version": 1
        }

        with open(audio_artifact_path, 'w') as f:
            json.dump(audio_data, f, indent=2)

        mark_stage_completed("audio", audio_artifact_path)

        # Update job state to queued (pipeline continues)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            UPDATE jobs SET state = 'queued', updated_at = ? WHERE job_id = ?
        """, (timestamp, job_id))
        conn.commit()

        return job_id, not is_retry, is_retry  # is_new_job, should_retry

    except Exception as e:
        # Mark all stages as failed
        mark_stage_failed("ingest", str(e))
        mark_stage_failed("probe", str(e))
        mark_stage_failed("proxy", str(e))
        mark_stage_failed("audio", str(e))

        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            UPDATE jobs SET state = 'failed', updated_at = ? WHERE job_id = ?
        """, (timestamp, job_id))
        conn.commit()

        raise StageError(f"Ingest pipeline failed: {e}") from e
