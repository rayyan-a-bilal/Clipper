#!/usr/bin/env python3
"""
Script to create test fixtures for clipper.
"""

import json
import os
from pathlib import Path

# Create words.json fixture file
def create_words_fixture():
    """Create fake Word data for testing."""
    words_data = [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "Hello",
            "prob": 0.95
        },
        {
            "start": 1.2,
            "end": 2.1,
            "text": "world",
            "prob": 0.92
        },
        {
            "start": 2.1,
            "end": 3.0,
            "text": "this",
            "prob": 0.87
        },
        {
            "start": 3.0,
            "end": 4.5,
            "text": "is",
            "prob": 0.91
        },
        {
            "start": 4.5,
            "end": 6.0,
            "text": "a",
            "prob": 0.85
        },
        {
            "start": 6.0,
            "end": 7.8,
            "text": "test",
            "prob": 0.93
        }
    ]
    
    # Ensure fixtures directory exists
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)
    
    # Write words.json
    with open(fixtures_dir / "words.json", "w") as f:
        json.dump(words_data, f, indent=2)
        
    print("Created tests/fixtures/words.json")

def create_sample_video():
    """Create a simple test video using ffmpeg if available."""
    import subprocess
    try:
        # Check if ffmpeg is available
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Create a simple 10-second test video using testsrc filter
            print("Creating sample test video...")
            output_path = Path("tests/fixtures/sample_video.mp4")
            cmd = [
                "ffmpeg",
                "-f", "lavfi",
                "-i", "testsrc=duration=10:size=640x480:rate=30",
                "-c:v", "libx264",
                "-t", "10",
                "-y",
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Created {output_path}")
        else:
            print("ffmpeg not found - skipping video creation")
            
    except Exception as e:
        print(f"Failed to create test video: {e}")

if __name__ == "__main__":
    print("Creating test fixtures...")
    create_words_fixture()
    create_sample_video()
    print("Fixtures created successfully!")