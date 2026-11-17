"""
Test cases for ASR adapters.
"""

import json
from pathlib import Path

from clipper.adapters.asr import FixtureASR
from clipper.models import Word


def test_fixture_asr_loads_list_format():
    """Test that FixtureASR correctly loads list format from fixture."""
    fixture = FixtureASR(fixture_path=Path("tests/fixtures/words.json"))

    # Test the transcribe method
    words = fixture.transcribe(Path("dummy.wav"))

    assert isinstance(words, list)
    assert len(words) > 0
    for word in words:
        assert isinstance(word, Word)
        assert hasattr(word, 'start')
        assert hasattr(word, 'end')
        assert hasattr(word, 'text')
        assert hasattr(word, 'prob')


def test_fixture_asr_loads_dict_format():
    """Test that FixtureASR load dict format from fixture."""
    # Create a temporary dict-formatted fixture
    test_data = {
        "words": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "prob": 0.95},
            {"start": 1.0, "end": 2.5, "text": "world", "prob": 0.98}
        ]
    }

    # Temporary file approach
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        fixture = FixtureASR(fixture_path=Path(temp_path))
        words = fixture.transcribe(Path("dummy.wav"))

        assert isinstance(words, list)
        assert len(words) == 2
        assert words[0].text == "Hello"
        assert words[1].text == "world"
    finally:
        os.unlink(temp_path)


def test_fixture_asr_close_safety():
    """Test that FixtureASR close() method is safe."""
    fixture = FixtureASR(fixture_path=Path("tests/fixtures/words.json"))

    # Should not raise any exception
    fixture.close()


def test_fixture_asr_empty_words():
    """Test behavior with empty words list."""
    # Create a temporary empty words fixture
    import os
    import tempfile

    empty_data = []
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(empty_data, f)
        temp_path = f.name

    try:
        fixture = FixtureASR(fixture_path=Path(temp_path))
        words = fixture.transcribe(Path("dummy.wav"))

        assert isinstance(words, list)
        assert len(words) == 0
    finally:
        os.unlink(temp_path)


def test_fixture_asr_with_empty_text():
    """Test behavior with words that have empty text."""
    # Create a temporary fixture with empty text words
    import os
    import tempfile

    test_data = {
        "words": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "prob": 0.95},
            {"start": 1.0, "end": 2.5, "text": "", "prob": 0.98},
            {"start": 2.5, "end": 4.0, "text": "world", "prob": 0.90}
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        fixture = FixtureASR(fixture_path=Path(temp_path))
        words = fixture.transcribe(Path("dummy.wav"))

        assert isinstance(words, list)
        assert len(words) == 2  # Empty word should be filtered out
        assert words[0].text == "Hello"
        assert words[1].text == "world"
    finally:
        os.unlink(temp_path)


def test_fixture_asr_word_validation():
    """Test that words are properly validated by pydantic."""
    fixture = FixtureASR(fixture_path=Path("tests/fixtures/words.json"))

    words = fixture.transcribe(Path("dummy.wav"))

    # Each word should be a valid Word object with all required fields
    for word in words:
        assert isinstance(word.start, (int, float))
        assert isinstance(word.end, (int, float))
        assert isinstance(word.text, str)
        assert isinstance(word.prob, (int, float))
