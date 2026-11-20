"""
Selection + draft ClipPlan stage implementation for the clipper pipeline (M5c).

TWO stages, one artifact each (mirrors the M5b single-stage pattern):

``selection``
- requires (per-job artifacts from earlier milestones):
    * ``candidates.json`` — M4b candidates stage (``scripts/candidates_job.py``)
    * ``scores.json``     — M5b scoring stage  (``scripts/score_job.py``)
    * ``sentences.json``  — M2b transcribe stage (``scripts/transcribe_job.py``)
    * ``words.json``      — M2b transcribe stage (``scripts/transcribe_job.py``)
    * ``silence.json``    — M3b analysis stage  (``scripts/analyze_job.py``)
    * ``shots.json``      — M3b analysis stage  (``scripts/analyze_job.py``)
    * ``probe.json``      — M1 ingest stage     (``python -m clipper ingest``)
  (``schema_version == 1`` validated on every input; clean ``StageError``
  naming the missing artifact AND the milestone/script that produces it).
- :func:`clipper.selection.select_clips` ranks/gates/dedupes; then
  :func:`clipper.cuts.refine_cut` is applied to candidates in RANK ORDER.
  A ``CutError`` marks that decision's ``cut_reject_reason`` and
  BACKFILLS from the next eligible, non-overlapping candidate until the
  target is reached or the eligible list is exhausted.
- writes ``selection.json`` (``SelectionArtifact``, ``schema_version=1``)
  with one decision record per candidate.

``clip_plans``
- if ZERO clips are selected: writes an EMPTY ``ClipPlansArtifact`` and
  SKIPS THE LLM ENTIRELY (zero clips is a VALID success).
- otherwise ONE metadata call (``METADATA_SYSTEM`` -> ``MetadataResponse``)
  for ALL selected clips, validated (title <= 80 chars; hook_overlay <= 6
  words AND <= 40 chars; description <= 300 chars; 3-6 hashtags each
  starting with '#' and containing no spaces; exactly one item per clip).
  After retries the stage uses a DETERMINISTIC fallback (title = first
  8 transcript words, hook_overlay = first 5 words, description = first
  200 chars, hashtags = ``[f"#{job.niche}"]``) and logs a warning.
- builds DRAFT :class:`ClipPlan`s per selected clip (RANK ORDER):
  ``clip_id = f"{job_id}-clip{n:02d}"`` (n at 1), start/end from
  refine_cut, weighted_score, the metadata, ``layout = center_crop``,
  ``crop_path = []``, ``status = pending_qc`` (M6 owns reframe).
- writes ``clip_plans.json`` and upserts the ``clips`` rows
  (``plan_json``, ``status``; ``clip_id`` is the primary key).

Stage skip rule (BOTH stages): DB state = ``done`` AND the artifact file
exists AND has ``schema_version == 1``. Job goes ``queued -> running ->
queued`` on success; on failure the active stage + job are marked
``failed`` and a clean ``StageError`` is raised.

Client policy: ``client`` may be ``None`` (selection/cuts need no LLM).
If at least one clip is selected and ``client is None``, the ``selection``
stage completes normally and the ``clip_plans`` stage stays PENDING —
the summary carries ``clip_plans == "pending_llm"`` and the caller re-runs
with a real client (``scripts/select_job.py`` spawns the managed Ollama
ONLY in that case; with zero selected clips Ollama is never started).
"""

import json
import logging
from pathlib import Path

from ..adapters.llm import LLMClient, LLMError
from ..candidates import temporal_iou
from ..config import Config
from ..cuts import CutError, refine_cut
from ..models import (
    Candidate,
    CandidatesArtifact,
    ClipDecision,
    ClipMetadata,
    ClipPlan,
    ClipPlansArtifact,
    Layout,
    MetadataResponse,
    ProbeResult,
    ScoredClip,
    ScoresArtifact,
    SelectionArtifact,
    Sentence,
    SentencesArtifact,
    ShotsArtifact,
    SilenceArtifact,
    WordsArtifact,
)
from ..prompts import METADATA_SYSTEM
from ..selection import SELECTION_IOU, select_clips
from . import StageError
from .analyze import (
    _finish_stage,
    _is_done,
    _now,
    _set_job_state,
    _start_stage,
)

_SCHEMA_VERSION = 1
_STAGE_SELECTION = "selection"
_STAGE_PLANS = "clip_plans"
_SELECTION_NAME = "selection.json"
_PLANS_NAME = "clip_plans.json"

_LOG = logging.getLogger("clipper")

# Metadata bounds (M5c contract: METADATA_SYSTEM prompt + docs/spec/m5.md).
TITLE_MAX = 80
HOOK_MAX_WORDS = 6
HOOK_MAX_CHARS = 40
DESCRIPTION_MAX = 300
HASHTAG_MIN = 3
HASHTAG_MAX = 6

# (filename, expected artifact model, milestone hint for clean errors)
_REQUIRED_INPUTS = (
    (
        "candidates.json",
        CandidatesArtifact,
        "produced by the M4b candidates stage (scripts/candidates_job.py)",
    ),
    (
        "scores.json",
        ScoresArtifact,
        "produced by the M5b scoring stage (scripts/score_job.py)",
    ),
    (
        "sentences.json",
        SentencesArtifact,
        "produced by the M2b transcribe stage (scripts/transcribe_job.py)",
    ),
    (
        "words.json",
        WordsArtifact,
        "produced by the M2b transcribe stage (scripts/transcribe_job.py)",
    ),
    (
        "silence.json",
        SilenceArtifact,
        "produced by the M3b analysis stage (scripts/analyze_job.py)",
    ),
    (
        "shots.json",
        ShotsArtifact,
        "produced by the M3b analysis stage (scripts/analyze_job.py)",
    ),
    (
        "probe.json",
        ProbeResult,
        "produced by the M1 ingest stage (python -m clipper ingest)",
    ),
)


# ---------------------------------------------------------------------------
# Pure helpers (metadata validation + deterministic fallback)
# ---------------------------------------------------------------------------
def _validate_metadata(
    response: MetadataResponse, cand_ids: list[str],
) -> dict[str, ClipMetadata]:
    """Validate one metadata response; raise ValueError to trigger retry."""
    items = response.items
    if len(items) != len(cand_ids):
        raise ValueError(
            f"metadata returned {len(items)} item(s), expected {len(cand_ids)}"
        )

    by_id: dict[str, ClipMetadata] = {}
    for item in items:
        if item.cand_id in by_id:
            raise ValueError(f"duplicate metadata for cand_id {item.cand_id!r}")
        by_id[item.cand_id] = item

    expected = set(cand_ids)
    missing = sorted(expected - set(by_id))
    extra = sorted(set(by_id) - expected)
    if missing or extra:
        raise ValueError(
            f"metadata clip mismatch: missing={missing}, extra={extra}"
        )

    for item in items:
        _check_text(item, "title", item.title, TITLE_MAX)
        if (
            not item.hook_overlay.strip()
            or len(item.hook_overlay.split()) > HOOK_MAX_WORDS
            or len(item.hook_overlay) > HOOK_MAX_CHARS
        ):
            raise ValueError(
                f"{item.cand_id}: hook_overlay must be 1-{HOOK_MAX_WORDS} "
                f"words and <= {HOOK_MAX_CHARS} chars"
            )
        _check_text(item, "description", item.description, DESCRIPTION_MAX)

        hashtags = item.hashtags
        if not HASHTAG_MIN <= len(hashtags) <= HASHTAG_MAX:
            raise ValueError(
                f"{item.cand_id}: {len(hashtags)} hashtags "
                f"({HASHTAG_MIN}-{HASHTAG_MAX} required)"
            )
        for tag in hashtags:
            if not tag.startswith("#") or any(ch.isspace() for ch in tag):
                raise ValueError(
                    f"{item.cand_id}: invalid hashtag {tag!r} "
                    "(must start with '#' and contain no spaces)"
                )
    return by_id


def _check_text(
    item: ClipMetadata, name: str, value: str, maximum: int,
) -> None:
    if not value.strip():
        raise ValueError(f"{item.cand_id}: {name} is empty")
    if len(value) > maximum:
        raise ValueError(
            f"{item.cand_id}: {name} is {len(value)} chars (max {maximum})"
        )


def _fallback_metadata(
    cand_id: str, transcript: str, niche: str,
) -> ClipMetadata:
    """Deterministic fallback (M5c): first 8 words / first 5 words /
    first 200 chars / ``[f"#{niche}"]``."""
    words = transcript.split()
    return ClipMetadata(
        cand_id=cand_id,
        title=" ".join(words[:8]),
        hook_overlay=" ".join(words[:5]),
        description=transcript[:200],
        hashtags=[f"#{niche}"],
    )


def _clip_transcript(cand: Candidate, sentences: list[Sentence]) -> str:
    start_idx, end_idx = cand.sentence_range
    return " ".join(sentences[i].text for i in range(start_idx, end_idx + 1))


# ---------------------------------------------------------------------------
# Inputs / artifacts
# ---------------------------------------------------------------------------
def _load_inputs(job_dir: Path) -> dict[str, object]:
    """Load + schema-validate ALL required inputs; clean StageError on
    problems (naming the file and the milestone/script that produces it).

    Returns a dict of typed lists + scalar duration (all from artifacts on
    disk — fully deterministic).
    """
    payload: dict[str, object] = {}
    for name, model, hint in _REQUIRED_INPUTS:
        path = job_dir / name
        if not path.exists():
            raise StageError(
                f"Missing required input {path} — {hint}; run that stage first"
            )
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise StageError(f"Failed to read {name}: {e}") from e

        version = data.get("schema_version") if isinstance(data, dict) else None
        if version != _SCHEMA_VERSION:
            raise StageError(
                f"{name} has schema_version={version!r} "
                f"(expected {_SCHEMA_VERSION}) — {hint}"
            )
        try:
            payload[name] = model.model_validate(data)
        except Exception as e:
            raise StageError(f"{name} is not a valid artifact: {e}") from e

    return {
        "candidates": payload["candidates.json"].candidates,
        "scores": payload["scores.json"].scores,
        "sentences": payload["sentences.json"].sentences,
        "words": payload["words.json"].words,
        "silence": payload["silence.json"].regions,
        "shots": payload["shots.json"].shots,
        "duration_s": payload["probe.json"].duration_s,
    }


def _write_artifact(path: Path, artifact) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2) + "\n")
    except OSError as e:
        raise StageError(f"Failed to write artifact {path}: {e}") from e


def _artifacts_valid(path: Path, model) -> bool:
    try:
        model.model_validate_json(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _upsert_clips(conn, job_id: str, plans: list[ClipPlan]) -> None:
    ts = _now()
    for plan in plans:
        conn.execute(
            """
            INSERT INTO clips
                (clip_id, job_id, plan_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                job_id = excluded.job_id,
                plan_json = excluded.plan_json,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                plan.clip_id,
                job_id,
                plan.model_dump_json(),
                plan.status,
                ts,
                ts,
            ),
        )
    conn.commit()


def _job_niche(conn, job_id: str) -> str:
    row = conn.execute(
        "SELECT niche FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    return (row[0] if row else None) or "clip"


# ---------------------------------------------------------------------------
# Pure selection + refine computation (deterministic; used by both stages)
# ---------------------------------------------------------------------------
def _compute_selection(
    cfg: Config, inputs: dict[str, object],
) -> tuple[list[ClipDecision], dict[str, tuple[float, float]]]:
    """Rank/gate/dedupe via select_clips; refine in RANK ORDER with
    CutError backfill. Returns (all decisions in candidate order, cuts)."""
    candidates: list[Candidate] = inputs["candidates"]  # type: ignore[assignment]
    sentences: list[Sentence] = inputs["sentences"]  # type: ignore[assignment]
    scores: list[ScoredClip] = inputs["scores"]  # type: ignore[assignment]
    candidates_by_id = {c.cand_id: c for c in candidates}
    scores_by_id = {s.cand_id: s for s in scores}

    result = select_clips(candidates, scores, cfg)
    limit = max(0, min(cfg.clips.target_count, cfg.clips.max_count))

    # Refine in RANK ORDER; CutError -> mark + BACKFILL from the next
    # eligible candidate (skipping any that would overlap an already
    # successfully-cut clip at IoU > SELECTION_IOU) until target or
    # exhaustion.
    cuts: dict[str, tuple[float, float]] = {}
    cut_reject: dict[str, str] = {}
    chosen: list[str] = []
    for cand in result.ranked:
        if len(cuts) >= limit:
            break
        if any(
            temporal_iou(
                cand.start, cand.end,
                candidates_by_id[c].start, candidates_by_id[c].end,
            )
            > SELECTION_IOU
            for c in chosen
        ):
            continue
        try:
            cuts[cand.cand_id] = refine_cut(
                cand,
                scores_by_id[cand.cand_id],
                sentences,
                inputs["words"],  # type: ignore[arg-type]
                inputs["silence"],  # type: ignore[arg-type]
                inputs["shots"],  # type: ignore[arg-type]
                inputs["duration_s"],  # type: ignore[arg-type]
            )
            chosen.append(cand.cand_id)
        except CutError as e:
            cut_reject[cand.cand_id] = str(e)

    base = {d.cand_id: d for d in result.decisions}
    decisions = [
        ClipDecision(
            cand_id=c.cand_id,
            weighted=round(base[c.cand_id].weighted, 4),
            eligible=base[c.cand_id].eligible,
            gate_reasons=list(base[c.cand_id].gate_reasons),
            suppressed_by=base[c.cand_id].suppressed_by,
            selected=c.cand_id in cuts,
            cut_reject_reason=cut_reject.get(c.cand_id),
        )
        for c in candidates
    ]
    return decisions, cuts


# ---------------------------------------------------------------------------
# Metadata call (retry -> deterministic fallback)
# ---------------------------------------------------------------------------
def _metadata_call(
    client: LLMClient,
    cuts: dict[str, tuple[float, float]],
    sentences: list[Sentence],
    candidates_by_id: dict[str, Candidate],
    niche: str,
    cfg: Config,
) -> dict[str, ClipMetadata]:
    """ONE metadata call for ALL selected clips (retries); after exhaustion
    falls back to deterministic metadata and logs a warning."""
    clips = [
        {
            "cand_id": cand_id,
            "start": start,
            "end": end,
            "transcript": _clip_transcript(candidates_by_id[cand_id], sentences),
        }
        for cand_id, (start, end) in cuts.items()
    ]
    user_payload = json.dumps({"clips": clips}, ensure_ascii=False, indent=2)
    cand_ids = list(cuts.keys())

    attempts = max(1, cfg.llm.max_retries + 1)
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = client.generate_json(
                METADATA_SYSTEM, user_payload, MetadataResponse
            )
            return _validate_metadata(response, cand_ids)
        except (LLMError, ValueError) as e:
            last_error = e

    _LOG.warning(
        "clip metadata: LLM failed after %d attempt(s); using deterministic "
        "fallback for %d clip(s): %s",
        attempts,
        len(cuts),
        last_error,
    )
    return {
        cand_id: _fallback_metadata(
            cand_id,
            _clip_transcript(candidates_by_id[cand_id], sentences),
            niche,
        )
        for cand_id in cuts
    }


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------
def _stage_done(conn, job_id: str, name: str, path: Path, model, valid=True) -> bool:
    skip, stored = _is_done(conn, job_id, name)
    if not skip:
        return False
    # _is_done already checked existence + schema_version == 1; also enforce
    # full model validity so a corrupt artifact triggers a re-run.
    return _artifacts_valid(Path(stored) if stored else path, model) and valid


def run_selection(
    job_id: str, cfg: Config, conn, client: LLMClient | None = None,
) -> dict:
    """Run the M5c ``selection`` + ``clip_plans`` stages for a job.

    Args:
        job_id: The job ID (must exist in the DB, with M5b scoring outputs).
        cfg: Configuration object (data_dir, clips.*, llm.*).
        conn: sqlite3 connection over the clipper database.
        client: LLM client (real or MockLLM). May be ``None`` — the
            selection/cuts stages need no LLM; if >= 1 clip is selected and
            ``client is None`` the ``clip_plans`` stage stays pending and
            the summary carries ``clip_plans == "pending_llm"`` (the
            caller re-runs with a real client).

    Returns:
        Summary dict (job_id, selected, gate_failures, suppressed,
        cut_failures, plans, clip_plans status/path, skipped stages).

    Raises:
        StageError: Clean error on any failure (missing job/inputs, schema
            mismatch, write failure). The active stage and job are marked
            failed.
    """
    row = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    sel_path = job_dir / _SELECTION_NAME
    plans_path = job_dir / _PLANS_NAME

    summary: dict = {
        "job_id": job_id,
        "model": cfg.llm.model,
        "selected": 0,
        "gate_failures": 0,
        "suppressed": 0,
        "cut_failures": 0,
        "plans": 0,
        "clip_plans": str(plans_path),
        "selection": str(sel_path),
        "skipped": [],
    }

    active: str | None = None
    try:
        # ---- stage 1: selection (pure; no LLM needed) ----
        skip_sel = _stage_done(conn, job_id, _STAGE_SELECTION, sel_path,
                               SelectionArtifact)
        cuts: dict[str, tuple[float, float]] | None = None
        inputs: dict[str, object] | None = None
        if skip_sel:
            decisions = SelectionArtifact.model_validate_json(
                sel_path.read_text(encoding="utf-8")
            ).decisions
            summary["skipped"].append(_STAGE_SELECTION)
        else:
            active = _STAGE_SELECTION
            _set_job_state(conn, job_id, "running")
            _start_stage(conn, job_id, _STAGE_SELECTION)
            inputs = _load_inputs(job_dir)
            decisions, cuts = _compute_selection(cfg, inputs)
            _write_artifact(
                sel_path, SelectionArtifact(schema_version=1, decisions=decisions)
            )
            _finish_stage(conn, job_id, _STAGE_SELECTION, str(sel_path))
            active = None

        summary["selected"] = sum(1 for d in decisions if d.selected)
        summary["gate_failures"] = sum(
            1 for d in decisions if not d.eligible
        )
        summary["suppressed"] = sum(
            1 for d in decisions if d.suppressed_by is not None
        )
        summary["cut_failures"] = sum(
            1 for d in decisions if d.cut_reject_reason is not None
        )

        # ---- stage 2: clip_plans (LLM only if >= 1 clip selected) ----
        skip_plans = _stage_done(conn, job_id, _STAGE_PLANS, plans_path,
                                 ClipPlansArtifact)
        if skip_plans:
            plans = ClipPlansArtifact.model_validate_json(
                plans_path.read_text(encoding="utf-8")
            ).plans
            summary["skipped"].append(_STAGE_PLANS)
            summary["plans"] = len(plans)
            return summary

        if cuts is None:
            # selection stage was skipped (done): re-derive the cut list
            # deterministically from the same on-disk inputs.
            inputs = _load_inputs(job_dir)
            _, cuts = _compute_selection(cfg, inputs)

        if not cuts:
            # Zero selected clips is a VALID success: empty artifact,
            # NO LLM call.
            active = _STAGE_PLANS
            _set_job_state(conn, job_id, "running")
            _start_stage(conn, job_id, _STAGE_PLANS)
            _write_artifact(
                plans_path, ClipPlansArtifact(schema_version=1, plans=[])
            )
            _finish_stage(conn, job_id, _STAGE_PLANS, str(plans_path))
            active = None
            _set_job_state(conn, job_id, "queued")
            summary["plans"] = 0
            return summary

        if client is None:
            # Selection/cuts are DONE and persisted; the plans stage awaits
            # an LLM client. Leave it pending (NOT failed) — the caller
            # re-runs with a client (scripts/select_job.py).
            _set_job_state(conn, job_id, "queued")
            summary["clip_plans"] = "pending_llm"
            return summary

        if inputs is None:
            inputs = _load_inputs(job_dir)
        active = _STAGE_PLANS
        _set_job_state(conn, job_id, "running")
        _start_stage(conn, job_id, _STAGE_PLANS)
        niche = _job_niche(conn, job_id)
        candidates_by_id = {
            c.cand_id: c for c in inputs["candidates"]
        }
        metadata_by_id = _metadata_call(
            client, cuts,  # type: ignore[arg-type]
            inputs["sentences"],  # type: ignore[arg-type]
            candidates_by_id, niche, cfg,
        )
        weighted_by_id = {d.cand_id: d.weighted for d in decisions}

        plans: list[ClipPlan] = []
        for n, (cand_id, (start, end)) in enumerate(cuts.items(), start=1):
            meta = metadata_by_id[cand_id]
            plans.append(
                ClipPlan(
                    clip_id=f"{job_id}-clip{n:02d}",
                    job_id=job_id,
                    start=start,
                    end=end,
                    layout=Layout.center_crop,
                    crop_path=[],
                    weighted_score=weighted_by_id.get(cand_id, 0.0),
                    title=meta.title,
                    hook_overlay=meta.hook_overlay,
                    description=meta.description,
                    hashtags=list(meta.hashtags),
                    status="pending_qc",
                )
            )

        _write_artifact(plans_path, ClipPlansArtifact(schema_version=1, plans=plans))
        _upsert_clips(conn, job_id, plans)
        _finish_stage(conn, job_id, _STAGE_PLANS, str(plans_path))
        active = None

        _set_job_state(conn, job_id, "queued")
        summary["plans"] = len(plans)
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"M5c selection/clip_plans failed: {e}"
        )
        if active is not None:
            try:
                _finish_stage(conn, job_id, active, error=str(clean))
            except Exception:
                pass
            try:
                _set_job_state(conn, job_id, "failed")
            except Exception:
                pass
        if clean is e:
            raise
        raise clean from e


__all__ = ["run_selection"]
