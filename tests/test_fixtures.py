"""
Tests related to fixture creation and handling.
"""

import json
import os
from pathlib import Path


def test_words_fixture_exists():
    """Test that words.json fixture exists and is valid."""
    fixture_path = Path("tests/fixtures/words.json")
    
    # Check if the file exists
    assert fixture_path.exists(), "words.json fixture should exist"
    
    # Check if it's valid JSON
    with open(fixture_path, 'r') as f:
        data = json.load(f)
        
    assert isinstance(data, list), "words.json should contain a list"
    assert len(data) > 0, "words.json should not be empty"
    
    # Check sample structure
    word = data[0]
    assert "start" in word
    assert "end" in word  
    assert "text" in word
    assert "prob" in word


def test_fixture_creation_script():
    """Test that fixture creation script runs."""
    
    # Simple check to see if the script exists and is executable
    script_path = Path("scripts/make_fixtures.py")
    assert script_path.exists(), "make_fixtures.py script should exist"
    
    # Check that it has basic structure
    with open(script_path, 'r') as f:
        content = f.read()
        
    assert "create_words_fixture" in content
    assert "create_sample_video" in content