"""
Media adapter implementations for clipper.
"""

from .media import MediaAdapter, FFmpegMediaAdapter, FixtureMediaAdapter, get_adapter

__all__ = ["MediaAdapter", "FFmpegMediaAdapter", "FixtureMediaAdapter", "get_adapter"]