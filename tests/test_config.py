"""
Tests for configuration module.
"""

import tempfile
import os
from pathlib import Path
from clipper.config import Config


def test_config_load():
    """Test that config loads without error."""
    config = Config.load_config()
    assert config is not None
    assert hasattr(config, 'paths')
    assert hasattr(config, 'data_dir')
    

def test_config_with_temp_data_dir():
    """Test config with temporary data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['CLIPPER_DATA_DIR'] = temp_dir
        config = Config.load_config()
        
        # Check that the override worked
        assert str(config.resolved_data_dir) == temp_dir
        
        # Clean up
        del os.environ['CLIPPER_DATA_DIR']


def test_database_migration():
    """Test database initialization."""
    from clipper.db import Database
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['CLIPPER_DATA_DIR'] = temp_dir
        config = Config.load_config()
        
        # Create database instance
        db = Database(config)
        
        # Check schema version
        version = db.get_schema_version()
        assert isinstance(version, int)
        
        # Clean up
        del os.environ['CLIPPER_DATA_DIR']
        

def test_contract_roundtrip():
    """Test that data contracts can be serialized and deserialized."""
    from clipper.models import Word, Sentence, Shot
    
    # Test Word contract
    word = Word(
        start=1.0,
        end=2.5,
        text="Hello world",
        prob=0.95
    )
    
    # Test serialization
    word_json = word.model_dump()
    assert isinstance(word_json, dict)
    
    # Test deserialization
    restored_word = Word(**word_json)
    assert restored_word.start == 1.0
    assert restored_word.text == "Hello world"
    
    # Test Sentence contract
    sentence = Sentence(
        start=1.0,
        end=4.5,
        text="Hello world this is a test",
        word_idx=(0, 3)
    )
    
    sent_json = sentence.model_dump()
    restored_sentence = Sentence(**sent_json)
    assert restored_sentence.text == "Hello world this is a test"
    
    # Test Shot contract
    shot = Shot(
        start=1.0,
        end=10.5
    )
    
    shot_json = shot.model_dump()
    restored_shot = Shot(**shot_json)
    assert restored_shot.start == 1.0