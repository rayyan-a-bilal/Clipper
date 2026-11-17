#!/usr/bin/env python3
"""
Make test fixtures for clipper.
"""

import json
from pathlib import Path


def create_words_fixture():
    """Create the words.json fixture file with sample data."""

    fixture_dir = Path("./tests/fixtures")
    fixture_dir.mkdir(exist_ok=True)

    words_fixture = fixture_dir / "words.json"

    if not words_fixture.exists():
        # Create a sample words.json file with test data
        words_data = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "prob": 0.95
            },
            {
                "start": 1.0,
                "end": 2.5,
                "text": "world",
                "prob": 0.98
            },
            {
                "start": 2.5,
                "end": 4.0,
                "text": "this",
                "prob": 0.90
            },
            {
                "start": 4.0,
                "end": 5.5,
                "text": "is",
                "prob": 0.85
            },
            {
                "start": 5.5,
                "end": 7.0,
                "text": "a",
                "prob": 0.80
            },
            {
                "start": 7.0,
                "end": 8.5,
                "text": "test",
                "prob": 0.92
            }
        ]

        with open(words_fixture, 'w') as f:
            json.dump(words_data, f, indent=2)

        print(f"Created words.json fixture at {words_fixture}")

    return True

def create_sample_video():
    """Create a basic sample MP4 test file."""

    fixture_dir = Path("./tests/fixtures")
    fixture_dir.mkdir(exist_ok=True)

    # Create a dummy test video file with minimal content
    test_video = fixture_dir / "sample.mp4"

    if not test_video.exists():
        print("Creating dummy test video...")
        # Just create a small dummy file (would normally be filled with actual video data)
        with open(test_video, 'w') as f:
            f.write("# Dummy MP4 test file\n")
            f.write("This is a placeholder for testing purposes\n")
            f.write("In a real installation, this would contain actual video content\n")

        print(f"Created dummy test video at {test_video}")

    return True

def create_all_fixtures():
    """Create all the required test fixtures."""
    print("Creating test fixtures...")

    success = True
    success &= create_words_fixture()
    success &= create_sample_video()

    if success:
        print("All test fixtures created successfully!")
    else:
        print("Failed to create some test fixtures.")

    return success

if __name__ == "__main__":
    success = create_all_fixtures()
    if not success:
        exit(1)
