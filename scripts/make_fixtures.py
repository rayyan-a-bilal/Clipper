#!/usr/bin/env python3
"""
Make test fixtures for clipper.
"""

import os
import shutil
import subprocess
from pathlib import Path

def create_test_file():
    """Create a simple test video file using ffmpeg."""
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install ffmpeg to create test fixtures.")
        return False
    
    # Create a basic test video file that can be used for testing
    fixture_dir = Path("./tests/fixtures")
    fixture_dir.mkdir(exist_ok=True)
    
    # Create a simple 5-second test video with a moving pattern using ffmpeg
    test_video = fixture_dir / "sample.mp4"
    
    if not test_video.exists():
        print("Creating sample test video...")
        # Create 10-second test video with some motion for testing
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=640x480:rate=30",
            "-c:v", "libx264",
            "-t", "5",
            str(test_video)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Created test video at {test_video}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create test video: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = create_test_file()
    if not success:
        exit(1)