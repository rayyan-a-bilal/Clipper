"""
Pipeline stages for clipper.
"""

class StageError(Exception):
    """Custom exception for pipeline stage errors."""
    pass

# NOTE: import kept below the StageError definition to avoid a circular import
# (ingest.py does `from . import StageError`), which is why E402 is suppressed here.
from .ingest import compute_content_hash, run_ingest  # noqa: E402

__all__ = ["run_ingest", "compute_content_hash", "StageError"]
