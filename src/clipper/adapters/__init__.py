"""
Media adapter implementations for clipper.
"""

from .media import FFmpegMediaAdapter, FixtureMediaAdapter, MediaAdapter, get_adapter

__all__ = ["MediaAdapter", "FFmpegMediaAdapter", "FixtureMediaAdapter", "get_adapter"]
