"""
Pipeline stages for clipper.
"""

class StageError(Exception):
    """Custom exception for pipeline stage errors."""
    pass

from .ingest import compute_content_hash, run_ingest

__all__ = ["run_ingest", "compute_content_hash", "StageError"]
