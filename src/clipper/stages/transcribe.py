"""
Transcription stage implementation for clipper pipeline.
"""

import json
from pathlib import Path
from typing import List

from ..adapters.asr import ASRAdapter, get_adapter
from ..config import Config
from ..db import DB
from ..models import Word, Sentence, WordsArtifact, SentencesArtifact


def run_transcribe(job_id: str, cfg: Config, conn, asr: ASRAdapter = None) -> bool:
    """
    Run the transcribe stage for a job.
    
    Args:
        job_id: The job ID to process
        cfg: Configuration object  
        conn: Database connection
        asr: ASR adapter (if None, uses get_adapter to create one)
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        Exception: If the operation fails
    """
    # Create DB instance for job operations
    db = DB(conn)
    
    # Load the job data  
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found in database")
    
    # Get the audio path from previous stage (from M1)
    job_dir = cfg.data_dir / "jobs" / job_id
    audio_wav_path = job_dir / "audio.wav"
    
    if not audio_wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")
    
    # Get adapter if not provided
    if asr is None:
        asr = get_adapter(cfg)
        
    try:
        # First stage: transcribe
        transcribe_stage_name = "transcribe"
        sentences_stage_name = "sentences"
        
        # Check if we can skip the transcribe stage
        transcribe_stage = db.get_stage(job_id, transcribe_stage_name)
        if (transcribe_stage and 
            transcribe_stage["state"] == "completed" and 
            transcribe_stage.get("artifact_path") and
            Path(transcribe_stage["artifact_path"]).exists()):
            # Stage already completed and artifact exists - skip
            print(f"Stage {transcribe_stage_name} already completed, skipping...")
        else:
            # Run the actual transcription
            print(f"Running transcription for job {job_id}")
            words = asr.transcribe(audio_wav_path)
            
            # Write words.json artifact  
            words_artifact = WordsArtifact(schema_version=1, words=words)
            words_json_path = job_dir / "words.json"
            
            with open(words_json_path, "w") as f:
                json.dump(words_artifact.model_dump(), f, indent=2)
                
            # Mark stage as complete
            db.update_stage(job_id, transcribe_stage_name, "completed", 
                          artifact_path=str(words_json_path))
        
        # Second stage: build sentences
        # Check if we can skip the sentences stage
        sentences_stage = db.get_stage(job_id, sentences_stage_name)
        if (sentences_stage and 
            sentences_stage["state"] == "completed" and
            sentences_stage.get("artifact_path") and
            Path(sentences_stage["artifact_path"]).exists()):
            # Stage already completed and artifact exists - skip
            print(f"Stage {sentences_stage_name} already completed, skipping...")
        else:
            # Load words from transcribe stage
            if transcribe_stage and transcribe_stage.get("artifact_path"):
                words_json_path = Path(transcribe_stage["artifact_path"])
            else:
                # If transcription was skipped or artifact path not found, run it now
                words = asr.transcribe(audio_wav_path)
                words_artifact = WordsArtifact(schema_version=1, words=words)
                words_json_path = job_dir / "words.json"
                with open(words_json_path, "w") as f:
                    json.dump(words_artifact.model_dump(), f, indent=2)
            
            # Load words from file
            try:
                with open(words_json_path, "r") as f:
                    words_data = json.load(f)
                    
                words = [Word(**word_dict) for word_dict in words_data["words"]]
            
                # Build sentences
                from ..sentences import build_sentences
                sentences = build_sentences(words)
                
                # Write sentences.json artifact
                sentences_artifact = SentencesArtifact(schema_version=1, sentences=sentences)
                sentences_json_path = job_dir / "sentences.json"
                
                with open(sentences_json_path, "w") as f:
                    json.dump(sentences_artifact.model_dump(), f, indent=2)
                    
                # Mark stage as complete
                db.update_stage(job_id, sentences_stage_name, "completed", 
                              artifact_path=str(sentences_json_path))
                              
            except Exception as e:
                # If sentence building fails, mark the stage as failed and re-raise error
                db.update_stage(job_id, sentences_stage_name, "failed", error=str(e))
                raise
                
        # Update job state back to queued (as it will continue to next stages)
        db.update_job_state(job_id, "queued")
        
        return True
        
    except Exception as e:
        # If there's an error, mark failing stage and update job to failed state
        print(f"Error in transcription: {e}")
        # Determine which stage failed based on where the error occurred
        try:
            db.update_stage(job_id, transcribe_stage_name, "failed", error=str(e))
        except:
            pass  # If we can't update first stage, that's ok - just make sure job is marked failed
        db.update_job_state(job_id, "failed")
        raise
            
    finally:
        # Always close the adapter
        try:
            asr.close()
        except Exception as e:
            print(f"Warning: Failed to close ASR adapter: {e}")

# Exported function for ease of access
__all__ = ["run_transcribe"]