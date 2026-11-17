"""
Test cases for sentence building logic.
"""

import json

from clipper.models import Word
from clipper.sentences import build_sentences


def test_empty_words():
    """Test building sentences from empty word list."""
    result = build_sentences([])
    assert result == []


def test_single_word_no_punctuation():
    """Test that a single word without punctuation becomes one sentence."""
    words = [
        Word(start=0.0, end=1.0, text="Hello", prob=0.95)
    ]
    result = build_sentences(words)
    assert len(result) == 1
    assert result[0].text == "Hello"
    assert result[0].start == 0.0
    assert result[0].end == 1.0


def test_single_word_with_punctuation():
    """Test that a single word with punctuation becomes one sentence."""
    words = [
        Word(start=0.0, end=1.0, text="Hello!", prob=0.95)
    ]
    result = build_sentences(words)
    assert len(result) == 1
    assert result[0].text == "Hello!"
    assert result[0].start == 0.0
    assert result[0].end == 1.0


def test_punctuation_split():
    """Test splitting on punctuation."""
    words = [
        Word(start=0.0, end=1.0, text="Hello", prob=0.95),
        Word(start=1.0, end=2.5, text="world", prob=0.98),
        Word(start=2.5, end=4.0, text="This", prob=0.90),
        Word(start=4.0, end=5.5, text="is", prob=0.85),
        Word(start=5.5, end=7.0, text="a.", prob=0.92),  # Sentence ends here
        Word(start=7.0, end=8.5, text="test", prob=0.93)
    ]
    result = build_sentences(words)

    assert len(result) == 2
    assert result[0].text == "Hello world This is a."
    assert result[1].text == "test"


def test_gap_split():
    """Test splitting on gap > 0.6s."""
    words = [
        Word(start=0.0, end=1.0, text="Hello", prob=0.95),
        Word(start=1.0, end=2.5, text="world", prob=0.98),
        Word(start=3.2, end=4.0, text="This", prob=0.90),  # Gap > 0.6s
        Word(start=4.0, end=5.5, text="is", prob=0.85)
    ]
    result = build_sentences(words)

    assert len(result) == 2
    assert result[0].text == "Hello world"
    assert result[1].text == "This is"


def test_words_fixture():
    """Test that we can process the fixture words."""
    with open("tests/fixtures/words.json") as f:
        words_data = json.load(f)

    # Convert to Word objects
    words = [Word(**word_data) for word_data in words_data]

    # Call build_sentences function
    result = build_sentences(words)

    assert isinstance(result, list)
    # Basic checks - we don't know exact output values but should have some sentences
    for sentence in result:
        assert hasattr(sentence, 'start')
        assert hasattr(sentence, 'end')
        assert hasattr(sentence, 'text')
        assert hasattr(sentence, 'word_idx')
