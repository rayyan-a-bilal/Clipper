"""
Pipeline stages for clipper.
"""

class StageError(Exception):
    """Custom exception for pipeline stage errors."""
    pass

from .ingest import run_ingest, compute_content_hash

__all__ = ["run_ingest", "compute_content_hash", "StageError"]