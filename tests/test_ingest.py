"""
Tests for ingest stage functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from src.clipper.stages.ingest import run_ingest, compute_content_hash
from src.clipper.db import Database
from src.clipper.config import Config
from src.clipper.adapters.media import FixtureMediaAdapter, MediaError


def test_compute_content_hash():
    """Test content hash computation."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    try:
        # Test basic hash computation
        hash1 = compute_content_hash(temp_path)
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Test with different content produces different hash
        with open(temp_path, 'w') as f:
            f.write("different content")
        hash2 = compute_content_hash(temp_path)
        assert hash1 != hash2
        
        # Test with same content produces same hash  
        with open(temp_path, 'w') as f:
            f.write("test content")
        hash3 = compute_content_hash(temp_path)
        assert hash1 == hash3
        
    finally:
        os.unlink(temp_path)


def test_run_ingest_with_fixture_adapter():
    """Test ingest stage with fixture adapter."""
    # Create a mock configuration and database
    config = Config.load_config()
    
    # Create in-memory database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        config.paths.data_dir = tmpdir
        
        db = Database(config)
        conn = db.get_connection()
        
        try:
            # Test with fixture adapter
            adapter = FixtureMediaAdapter()
            
            # Run ingest on a fake source that doesn't exist (should fail gracefully)
            with pytest.raises(Exception):  # Will raise an error since file not found
                job_id, is_new, should_retry = run_ingest(
                    source="nonexistent.mp4",
                    client="test_client",
                    niche="test_niche",
                    cfg=config,
                    conn=conn,
                    media=adapter
                )
                
        finally:
            conn.close()


def test_run_ingest_with_mocked_probe():
    """Test ingest stage with mocked probe operations."""
    # Create a mock configuration and database
    config = Config.load_config()
    
    # Use in-memory DB for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        config.paths.data_dir = tmpdir
        
        db = Database(config)
        conn = db.get_connection()
        
        try:
            # Test with fixture adapter that has no audio (should fail at probe stage)
            adapter = FixtureMediaAdapter(has_audio=False)
            
            with pytest.raises(Exception):
                # This should raise a StageError because there's no audio
                job_id, is_new, should_retry = run_ingest(
                    source="/fake/path.mp4",
                    client="test_client",
                    niche="test_niche",
                    cfg=config,
                    conn=conn,
                    media=adapter
                )
                
        finally:
            conn.close()