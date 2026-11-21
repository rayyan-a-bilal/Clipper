"""
Adapter implementations for clipper (GPU/ffmpeg/network/model access).
"""

from .llm import LLMClient, LLMError, MockLLM, OllamaLLMClient, get_llm_client
from .media import FFmpegMediaAdapter, FixtureMediaAdapter, MediaAdapter, get_adapter
from .vision import (
    FFmpegVisionAdapter,
    FixtureVisionAdapter,
    VisionAdapter,
    get_vision_adapter,
)

__all__ = [
    "MediaAdapter",
    "FFmpegMediaAdapter",
    "FixtureMediaAdapter",
    "get_adapter",
    "LLMClient",
    "LLMError",
    "MockLLM",
    "OllamaLLMClient",
    "get_llm_client",
    "VisionAdapter",
    "FFmpegVisionAdapter",
    "FixtureVisionAdapter",
    "get_vision_adapter",
]
