"""M9d — QC actions file I/O adapter (docs/spec/m9.md m9-4, m9-5 export half).

Centralizes ALL file I/O for `clipper.qc_actions` (same discipline as
`adapters/assets.py`, M9c-J3). Read/write surface, and nothing else:

- `clip_plans.json`  (read / append / write — the ONLY artifact written by QC)
- `sentences.json`   (read — clip-window transcript for fallback metadata)
- per-clip render outputs `final.mp4` / `thumbnail.jpg` (read / copy ONLY)
- `data/exports/<job_id>/` export destination (write)

NEVER touched (M5/M9 invariants): `scores.json`, `selection.json`,
`candidates.json`, `words.json` — these are immutable for QC actions
(m9-5: "words.json, scores.json, selection.json, candidates.json are
immutable"). No LLM, no render, no ffmpeg, no GPU — CPU-only I/O only.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from clipper.models import ClipPlansArtifact, SentencesArtifact

CLIP_PLANS_FILENAME = "clip_plans.json"
SENTENCES_FILENAME = "sentences.json"
FINAL_FILENAME = "final.mp4"
THUMBNAIL_FILENAME = "thumbnail.jpg"


def read_clip_plans(job_dir: Path | str) -> ClipPlansArtifact:
    """Read `clip_plans.json` (canonical form).

    A MISSING file is a valid EMPTY artifact (a job whose clip_plans stage
    never ran has zero plans — zero is valid at every stage). Present but
    unparsable / schema-mismatched content is a concrete error.
    """
    path = Path(job_dir) / CLIP_PLANS_FILENAME
    if not path.is_file():
        return ClipPlansArtifact(schema_version=1, plans=[])
    try:
        artifact = ClipPlansArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"invalid {path}: {exc}") from exc
    return artifact


def write_clip_plans(job_dir: Path | str, artifact: ClipPlansArtifact) -> Path:
    """Atomically write `clip_plans.json` (canonical
    `model_dump_json(indent=2)` + newline, same form as M5C-5).

    Temp file in the SAME directory + `os.replace` (same discipline as
    M9c `write_job_overrides`); parent directories created.
    """
    directory = Path(job_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        directory,
        CLIP_PLANS_FILENAME,
        artifact.model_dump_json(indent=2) + "\n",
    )
    return directory / CLIP_PLANS_FILENAME


def read_sentences(job_dir: Path | str) -> SentencesArtifact:
    """Read `sentences.json` (M2b). Missing/corrupt → concrete ValueError
    naming the artifact AND the producer (never a silent empty transcript).
    """
    path = Path(job_dir) / SENTENCES_FILENAME
    if not path.is_file():
        raise ValueError(
            f"missing {path} — produced by the M2b transcribe stage "
            "(scripts/transcribe_job.py)"
        )
    try:
        return SentencesArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"invalid {path}: {exc}") from exc


def has_file(path: Path | str) -> bool:
    """Existence check for a specific file (never a directory)."""
    return Path(path).is_file()


def copy_file(src: Path | str, dst: Path | str) -> Path:
    """Copy one existing file to `dst` (parents created). `src` must exist —
    the caller does concrete missing-file checks first; a race here is a
    clean error, not a silent skip."""
    source = Path(src)
    if not source.is_file():
        raise ValueError(f"missing file to copy: {source}")
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def write_text_file(path: Path | str, text: str) -> Path:
    """Atomically write a text file (e.g. the export `metadata.json`)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(target.parent, target.name, text)
    return target


def _write_atomic(directory: Path, name: str, payload: str) -> Path:
    target = directory / name
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return target
