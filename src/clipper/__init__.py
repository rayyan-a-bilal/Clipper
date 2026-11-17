"""
Clipper - Local, headless video clipping pipeline
"""

from .utils import retry_with_backoff, async_retry_with_backoff

__version__ = "0.1.0"
__author__ = "RBilal"

__all__ = ["retry_with_backoff", "async_retry_with_backoff"]