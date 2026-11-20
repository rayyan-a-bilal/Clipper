"""
Pipeline stages for clipper.
"""

class StageError(Exception):
    """Custom exception for pipeline stage errors."""
    pass

# NOTE: imports kept below the StageError definition to avoid a circular import
# (ingest.py does `from . import StageError`), which is why E402 is suppressed here.
from .analyze import run_analysis  # noqa: E402
from .candidates import run_candidates  # noqa: E402
from .ingest import compute_content_hash, run_ingest  # noqa: E402
from .score import run_scoring  # noqa: E402
from .select import run_selection, run_selection_stage  # noqa: E402

__all__ = [
    "run_ingest",
    "compute_content_hash",
    "StageError",
    "run_analysis",
    "run_candidates",
    "run_scoring",
    "run_selection",
    "run_selection_stage",
]
