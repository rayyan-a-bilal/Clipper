"""
Transcription stage implementation for clipper pipeline.
"""

import json
from pathlib import Path

from ..adapters.asr import ASRAdapter, get_adapter
from ..config import Config
from ..db import Database
from ..models import SentencesArtifact, Word, WordsArtifact
from ..sentences import build_sentences


def run_transcribe(job_id: str, cfg: Config, asr: ASRAdapter = None) -> bool:
    """
    Run the transcribe and sentences stages for a job.

    Args:
        job_id: The job ID to process
        cfg: Configuration object
        asr: ASR adapter (if None, uses get_adapter to create one)

    Returns:
        True if successful

    Raises:
        Exception: If the operation fails
    """
    db = Database(cfg)

    # Load the job data
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found in database")

    # Get the audio path produced by the M1 audio stage
    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    audio_wav_path = job_dir / "audio.wav"

    if not audio_wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")

    # Get adapter if not provided
    if asr is None:
        asr = get_adapter(cfg)

    transcribe_stage_name = "transcribe"
    sentences_stage_name = "sentences"
    current_stage = transcribe_stage_name

    try:
        # Stage 1: transcribe (skip if already completed with an existing artifact)
        transcribe_stage = db.get_stage(job_id, transcribe_stage_name)
        words_json_path = job_dir / "words.json"

        if (transcribe_stage
                and transcribe_stage["state"] == "completed"
                and transcribe_stage.get("artifact_path")
                and Path(transcribe_stage["artifact_path"]).exists()):
            words_json_path = Path(transcribe_stage["artifact_path"])
        else:
            db.create_stage(job_id, transcribe_stage_name)
            words = asr.transcribe(audio_wav_path)

            words_artifact = WordsArtifact(schema_version=1, words=words)
            with open(words_json_path, "w") as f:
                json.dump(words_artifact.model_dump(), f, indent=2)

            db.update_stage(job_id, transcribe_stage_name, "completed",
                            artifact_path=str(words_json_path))

        # Stage 2: build sentences from words (deterministic, cheap to redo)
        current_stage = sentences_stage_name
        db.create_stage(job_id, sentences_stage_name)

        with open(words_json_path) as f:
            words_data = json.load(f)

        words = [Word(**word_dict) for word_dict in words_data["words"]]
        sentences = build_sentences(words)

        sentences_artifact = SentencesArtifact(schema_version=1, sentences=sentences)
        sentences_json_path = job_dir / "sentences.json"
        with open(sentences_json_path, "w") as f:
            json.dump(sentences_artifact.model_dump(), f, indent=2)

        db.update_stage(job_id, sentences_stage_name, "completed",
                        artifact_path=str(sentences_json_path))

        # Job continues to the next pipeline stages
        db.update_job_state(job_id, "queued")
        return True

    except Exception as e:
        # Mark the failing stage and the job
        try:
            db.update_stage(job_id, current_stage, "failed", error=str(e))
        except Exception:
            pass  # make sure the job is still marked failed
        db.update_job_state(job_id, "failed")
        raise

    finally:
        # Always close the adapter
        try:
            asr.close()
        except Exception:
            pass


__all__ = ["run_transcribe"]
