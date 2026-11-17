#!/usr/bin/env python3
"""
Manual transcription script for clipper jobs.
"""

import argparse
import os
from pathlib import Path

import typer

from clipper.config import Config
from clipper.db import DB
from clipper.stages.transcribe import run_transcribe


def main(job_id: str):
    """
    Transcribe a job manually.
    
    Args:
        job_id: The job ID to transcribe
    """
    # Load configuration
    cfg = Config.load_config()
    
    # Connect to database
    db_path = cfg.resolved_data_dir / "jobs.db"
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return 1
        
    conn = DB.connect(str(db_path))
    
    try:
        # Get adapter and run transcription
        from clipper.adapters.asr import get_adapter
        asr = get_adapter(cfg)
        
        print(f"Starting transcription for job {job_id}")
        success = run_transcribe(job_id, cfg, conn, asr)
        
        if success:
            # Report results 
            job_dir = cfg.data_dir / "jobs" / job_id
            
            # Count words
            words_file = job_dir / "words.json"
            word_count = 0
            if words_file.exists():
                import json
                with open(words_file) as f:
                    words_data = json.load(f)
                    word_count = len(words_data.get("words", []))
            
            # Count sentences 
            sentences_file = job_dir / "sentences.json"
            sentence_count = 0
            if sentences_file.exists():
                import json
                with open(sentences_file) as f:
                    sentences_data = json.load(f)
                    sentence_count = len(sentences_data.get("sentences", []))
            
            print(f"Job {job_id} transcription completed")
            print(f"  Words: {word_count}")
            print(f"  Sentences: {sentence_count}")
            print(f"  Words file: {words_file}")
            print(f"  Sentences file: {sentences_file}")
            return 0
        else:
            print(f"Failed to complete transcription for job {job_id}")
            return 1
            
    except Exception as e:
        print(f"Error transcribing job {job_id}: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a clipper job")
    parser.add_argument("job_id", help="Job ID to transcribe")
    
    args = parser.parse_args()
    exit(main(args.job_id))