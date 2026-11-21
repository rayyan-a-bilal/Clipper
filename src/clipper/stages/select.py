"""
Selection + draft ClipPlan stages for the clipper pipeline (M5C-4 / M5C-5).

``selection`` stage (M5C-4, NO LLM, NO metadata):

- required per-job artifacts from earlier milestones:
    * ``candidates.json`` — M4b candidates stage (``scripts/candidates_job.py``)
    * ``scores.json``     — M5b scoring stage  (``scripts/score_job.py``)
    * ``sentences.json``  — M2b transcribe stage (``scripts/transcribe_job.py``)
    * ``words.json``      — M2b transcribe stage (``scripts/transcribe_job.py``)
    * ``silence.json``    — M3b analysis stage  (``scripts/analyze_job.py``)
    * ``shots.json``      — M3b analysis stage  (``scripts/analyze_job.py``)
    * ``probe.json``      — M1 ingest stage     (``python -m clipper ingest``)
  (every input is schema-validated with ``schema_version == 1``; a clean
  ``StageError`` names the missing artifact AND the milestone/script that
  produces it).
- :func:`clipper.selection.select_clips` ranks / gates / dedupes
  (pure, no LLM); then :func:`clipper.cuts.refine_cut` is applied to the
  selected candidates in RANK ORDER. A ``CutError`` records that
  candidate's ``cut_reject_reason`` and BACKFILLS from the next eligible,
  non-overlapping candidate (``result.ranked`` order) until
  ``cfg.clips.target_count`` (capped at ``cfg.clips.max_count``) is reached
  or the eligible list is exhausted.
- writes ``selection.json`` (``SelectionArtifact``, ``schema_version=1``)
  with one decision record per candidate (ALL candidates, artifact order),
  including the refined ``(start, end)`` on every SELECTED decision
  (``refined_start`` / ``refined_end``) so M5C-5 can reuse it verbatim.
- stage skip rule: DB state = ``done`` AND ``selection.json`` exists AND
  its ``schema_version == 1``. Job goes ``queued -> running -> queued`` on
  success; on failure the stage + job are marked ``failed`` and a clean
  ``StageError`` (naming the offending artifact and its producer) is
  raised.
- ZERO selected clips is a VALID success (never fabricate clips).

``clip_plans`` stage (M5C-5 — ONE optional metadata call per JOB):

- requires ``selection.json`` (M5C-4 ``selection`` stage,
  ``scripts/select_job.py``), ``candidates.json`` (M4b) and
  ``sentences.json`` (M2b) — clean ``StageError`` naming the missing
  artifact AND the milestone/script that produces it; every input is
  ``schema_version == 1``.
- skip rule: DB state = ``done`` AND ``clip_plans.json`` exists AND its
  ``schema_version == 1``. Job goes ``queued -> running -> queued`` on
  success; failure marks stage + job ``failed`` and raises a clean
  ``StageError``.
- zero selected clips: writes an EMPTY ``ClipPlansArtifact`` and does NOT
  call the LLM at all (VALID success).
- selected > 0: ONE metadata call via
  ``client.generate_json(METADATA_SYSTEM, request_json, MetadataResponse)``
  where ``request_json = build_metadata_request(selected)`` — the prompt
  contains ONLY the selected clips' transcripts (never the full video).
  Retries up to ``cfg.llm.max_retries`` on ``LLMError``/``ValueError``
  (incl. schema-validation failures and timeouts); after retries are
  exhausted OR a timeout, EVERY clip gets deterministic
  ``fallback_metadata`` + a logged warning — metadata failure must NEVER
  fail the job (docs/spec/m5.md).
- The client's ``num_predict`` is capped at ``METADATA_NUM_PREDICT_CAP``
  (512) for the metadata call (cfg default is 1024).
- Builds one DRAFT :class:`ClipPlan` per selected clip in RANK order:
  ``clip_id = f"{job_id}-clip{n:02d}"`` (n 1-based), start/end from the
  selection stage's refined cut, ``layout = Layout.center_crop``,
  ``crop_path = []``, ``status = "pending_qc"`` (M6 replaces
  layout/crop_path). Writes ``clip_plans.json`` and upserts the ``clips``
  rows (``plan_json`` / ``status``).
"""

import json
import logging
from pathlib import Path

from ..adapters.llm import LLMClient, LLMError
from ..candidates import temporal_iou
from ..config import Config
from ..cuts import CutError, refine_cut
from ..metadata import (
    build_metadata_request,
    fallback_metadata,
    validate_metadata,
)
from ..models import (
    Candidate,
    CandidatesArtifact,
    ClipDecision,
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
    SilenceRegion,
    Word,
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

logger = logging.getLogger("clipper.stages.select")

_SCHEMA_VERSION = 1
_STAGE_NAME = "selection"
_ARTIFACT_NAME = "selection.json"

# --- clip_plans stage (M5C-5) ---
_PLANS_STAGE_NAME = "clip_plans"
_PLANS_ARTIFACT_NAME = "clip_plans.json"
# The metadata call is short-form text: cap num_predict at 512 tokens
# (cfg.llm.num_predict is 1024 for scoring) — M5C-5 card.
METADATA_NUM_PREDICT_CAP = 512

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


def _load_inputs(job_dir: Path) -> dict[str, object]:
    """Load + schema-validate ALL required inputs (clean ``StageError``
    naming the missing/offending artifact AND its producer milestone).

    Returns a dict of typed values (lists + ``duration_s``), entirely from
    the on-disk artifacts: deterministic input for pure selection/cuts.
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
    """Canonical artifact form: ``model_dump_json(indent=2)`` + newline."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2) + "\n")
    except OSError as e:
        raise StageError(f"Failed to write artifact {path}: {e}") from e


def _compute_selection(
    cfg: Config, inputs: dict[str, object],
) -> tuple[list[ClipDecision], dict[str, tuple[float, float]]]:
    """Rank/gate/dedupe via pure selection; refine in RANK ORDER with
    CutError backfill.

    Returns ``(decisions, cuts)``: one :class:`ClipDecision` per candidate
    (ALL candidates, original artifact order) and the refined
    ``{cand_id: (start, end)}`` map for the chosen candidates in selection
    order.
    """
    candidates: list[Candidate] = inputs["candidates"]  # type: ignore[assignment]
    sentences: list[Sentence] = inputs["sentences"]  # type: ignore[assignment]
    words: list[Word] = inputs["words"]  # type: ignore[assignment]
    silence: list[SilenceRegion] = inputs["silence"]  # type: ignore[assignment]
    shots = inputs["shots"]
    duration_s: float = inputs["duration_s"]  # type: ignore[assignment]
    scores: list[ScoredClip] = inputs["scores"]  # type: ignore[assignment]
    candidates_by_id = {c.cand_id: c for c in candidates}
    scores_by_id = {s.cand_id: s for s in scores}

    result = select_clips(candidates, scores, cfg)
    limit = max(0, min(cfg.clips.target_count, cfg.clips.max_count))

    # Refine in RANK ORDER (ranked = ALL eligible, weighted DESC / cheap
    # DESC / start ASC). A CutError marks that candidate's
    # cut_reject_reason and the walk continues (BACKFILL) with the next
    # eligible candidate not overlapping an already-cut choice (IoU >
    # SELECTION_IOU), until ``limit`` is reached or the list is exhausted.
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
            ) > SELECTION_IOU
            for c in chosen
        ):
            continue
        try:
            cuts[cand.cand_id] = refine_cut(
                cand,
                scores_by_id[cand.cand_id],
                sentences,
                words,
                silence,
                shots,
                duration_s,
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
            suppressed_by=(
                None
                if c.cand_id in cuts
                else base[c.cand_id].suppressed_by
            ),
            selected=c.cand_id in cuts,
            cut_reject_reason=cut_reject.get(c.cand_id),
            refined_start=cuts[c.cand_id][0] if c.cand_id in cuts else None,
            refined_end=cuts[c.cand_id][1] if c.cand_id in cuts else None,
        )
        for c in candidates
    ]
    return decisions, cuts


def run_selection_stage(job_id: str, cfg: Config, conn) -> dict:
    """Run the ``selection`` stage for a job (pure; NO LLM / NO Ollama).

    Args:
        job_id: The job ID (must exist in the DB, with M5b scoring outputs).
        cfg: Configuration object (data_dir, clips.*).
        conn: sqlite3 connection over the clipper database.

    Returns:
        Summary dict: job_id, selected, eligible, gate_failures,
        suppressed, cut_failures, selected_ids (rank order), skipped,
        artifact (path to selection.json).

    Raises:
        StageError: Clean error on any failure (missing job/inputs, schema
            mismatch, write failure). The stage and job are marked failed.
    """
    row = conn.execute(
        "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    artifact_path = job_dir / _ARTIFACT_NAME

    summary: dict = {
        "job_id": job_id,
        "selected": 0,
        "eligible": 0,
        "gate_failures": 0,
        "suppressed": 0,
        "cut_failures": 0,
        "selected_ids": [],
        "skipped": False,
        "artifact": str(artifact_path),
    }

    def _field(decision, key: str):
        if isinstance(decision, dict):
            return decision[key]
        return getattr(decision, key)

    def _fill_summary(decisions) -> None:
        summary["selected"] = sum(
            1 for d in decisions if _field(d, "selected")
        )
        summary["eligible"] = sum(
            1 for d in decisions if _field(d, "eligible")
        )
        summary["gate_failures"] = sum(
            1 for d in decisions if not _field(d, "eligible")
        )
        summary["suppressed"] = sum(
            1 for d in decisions if _field(d, "suppressed_by") is not None
        )
        summary["cut_failures"] = sum(
            1 for d in decisions if _field(d, "cut_reject_reason") is not None
        )
        summary["selected_ids"] = [
            _field(d, "cand_id") for d in decisions if _field(d, "selected")
        ]

    skip, stored = _is_done(conn, job_id, _STAGE_NAME)
    if skip:
        # Skip only when DB stage = done AND the artifact exists AND
        # schema_version == 1 (all checked by _is_done).
        with open(stored, encoding="utf-8") as f:
            data = json.load(f)
        _fill_summary(data.get("decisions", []))
        summary["skipped"] = True
        return summary

    _set_job_state(conn, job_id, "running")
    try:
        _start_stage(conn, job_id, _STAGE_NAME)

        inputs = _load_inputs(job_dir)
        decisions, cuts = _compute_selection(cfg, inputs)

        artifact = SelectionArtifact(schema_version=_SCHEMA_VERSION,
                                     decisions=decisions)
        _write_artifact(artifact_path, artifact)

        _finish_stage(conn, job_id, _STAGE_NAME, str(artifact_path))
        _set_job_state(conn, job_id, "queued")

        _fill_summary(decisions)
        summary["selected_ids"] = list(cuts.keys())  # rank (selection) order
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"Selection stage failed: {e}"
        )
        try:
            _finish_stage(conn, job_id, _STAGE_NAME, error=str(clean))
        except Exception:
            pass
        try:
            _set_job_state(conn, job_id, "failed")
        except Exception:
            pass
        if clean is e:
            raise
        raise clean from e


# Backwards-compatible alias for the M5c WIP callers (scripts/select_job.py);
# the client parameter is IGNORED — the selection stage needs no LLM.
def run_selection(job_id: str, cfg: Config, conn, client=None) -> dict:
    """M5C-4 alias: ``run_selection_stage`` (client ignored, NOT called)."""
    return run_selection_stage(job_id, cfg, conn)


# ---------------------------------------------------------------------------
# clip_plans stage (M5C-5) — ONE optional metadata call + DRAFT ClipPlans
# ---------------------------------------------------------------------------
_PLANS_REQUIRED_INPUTS = (
    (
        "selection.json",
        SelectionArtifact,
        "produced by the M5C-4 selection stage (scripts/select_job.py)",
    ),
    (
        "candidates.json",
        CandidatesArtifact,
        "produced by the M4b candidates stage (scripts/candidates_job.py)",
    ),
    (
        "sentences.json",
        SentencesArtifact,
        "produced by the M2b transcribe stage (scripts/transcribe_job.py)",
    ),
)


def _load_plans_inputs(job_dir: Path) -> tuple[SelectionArtifact, list, list]:
    """Load + schema-validate the ``clip_plans`` inputs (clean ``StageError``
    naming the missing/offending artifact AND its producer milestone)."""
    payload: dict[str, object] = {}
    for name, model, hint in _PLANS_REQUIRED_INPUTS:
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

    return (
        payload["selection.json"],  # type: ignore[return-value]
        payload["candidates.json"].candidates,  # type: ignore[union-attr]
        payload["sentences.json"].sentences,  # type: ignore[union-attr]
    )


def _rank_selected(selected, candidates_by_id: dict) -> list:
    """Selected decisions in RANK order (the same key ``select_clips``
    uses: weighted DESC, cheap_score DESC, start ASC, cand_id)."""

    def _key(d):
        cand = candidates_by_id[d.cand_id]
        return (-d.weighted, -cand.cheap_score, cand.start, d.cand_id)

    return sorted(selected, key=_key)


def _clip_transcript(sentences: list, start: float, end: float) -> str:
    """Transcript of the CLIP window only (sentences overlapping
    ``[start, end]``) — NEVER the full video transcript."""
    return " ".join(
        s.text for s in sentences if s.start < end and s.end > start
    )


def _cap_metadata_num_predict(client: LLMClient) -> None:
    """Cap the metadata request's ``num_predict`` at 512 (token cap for the
    short-form metadata call; ``cfg.llm.num_predict`` is 1024). Clients
    without a settable ``num_predict`` (e.g. ``MockLLM``) are untouched."""
    current = getattr(client, "num_predict", None)
    if isinstance(current, int) and not isinstance(current, bool):
        client.num_predict = min(current, METADATA_NUM_PREDICT_CAP)


def _metadata_for_selected(
    cfg: Config,
    niche: str,
    ranked: list,
    sentences: list,
    candidate_cuts: dict[str, tuple[float, float]],
    client: LLMClient | None,
    summary: dict,
) -> list:
    """ONE metadata call for all selected clips (retry -> deterministic
    fallback). Metadata failure must NEVER fail the job."""
    cand_ids = [d.cand_id for d in ranked]

    if client is None:
        logger.warning(
            "clip_plans: metadata client is None — using deterministic "
            "fallback metadata for all %d selected clip(s) (job continues)",
            len(ranked),
        )
        summary["fallback_reason"] = "no LLM client"
        return [
            fallback_metadata(
                d.cand_id, _clip_transcript(sentences, *candidate_cuts[d.cand_id]),
                niche,
            )
            for d in ranked
        ]

    request_json = build_metadata_request(
        [
            {
                "cand_id": d.cand_id,
                "transcript": _clip_transcript(
                    sentences, *candidate_cuts[d.cand_id]
                ),
                "niche": niche,
                "duration_s": round(
                    candidate_cuts[d.cand_id][1] - candidate_cuts[d.cand_id][0],
                    3,
                ),
            }
            for d in ranked
        ]
    )

    _cap_metadata_num_predict(client)
    attempts = max(1, cfg.llm.max_retries + 1)
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = client.generate_json(
                METADATA_SYSTEM, request_json, MetadataResponse
            )
            items = validate_metadata(response, cand_ids)
            summary["metadata_source"] = "llm"
            return items
        except (LLMError, ValueError) as e:
            # LLMError covers transport errors AND timeouts from the
            # OllamaLLMClient; ValueError covers schema-validation and
            # semantic metadata violations.
            last_error = e

    logger.warning(
        "clip_plans: metadata call failed after %d attempt(s) (%s) — "
        "using deterministic fallback metadata for all %d selected clip(s); "
        "the job continues and does NOT fail",
        attempts, last_error, len(ranked),
    )
    summary["fallback_reason"] = str(last_error)
    return [
        fallback_metadata(
            d.cand_id, _clip_transcript(sentences, *candidate_cuts[d.cand_id]),
            niche,
        )
        for d in ranked
    ]


def _build_plans(
    job_id: str, ranked: list, metadata: list
) -> list[ClipPlan]:
    """DRAFT ClipPlans in RANK order: center_crop / empty crop_path /
    pending_qc (M6 fills layout/crop_path later)."""
    meta_by_id = {m.cand_id: m for m in metadata}
    plans: list[ClipPlan] = []
    for n, d in enumerate(ranked, start=1):
        meta = meta_by_id[d.cand_id]
        assert d.refined_start is not None and d.refined_end is not None
        plans.append(
            ClipPlan(
                clip_id=f"{job_id}-clip{n:02d}",
                job_id=job_id,
                start=float(d.refined_start),
                end=float(d.refined_end),
                layout=Layout.center_crop,
                crop_path=[],
                weighted_score=float(d.weighted),
                title=meta.title,
                hook_overlay=meta.hook_overlay,
                description=meta.description,
                hashtags=list(meta.hashtags),
                status="pending_qc",
            )
        )
    return plans


def _upsert_clips(conn, job_id: str, plans: list[ClipPlan]) -> None:
    """Upsert one ``clips`` row per DRAFT plan (clip_id is the PK)."""
    now = _now()
    for p in plans:
        conn.execute(
            """
            INSERT INTO clips
                (clip_id, job_id, plan_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                plan_json = excluded.plan_json,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (p.clip_id, job_id, p.model_dump_json(), p.status, now, now),
        )
    conn.commit()


def run_clip_plans_stage(
    job_id: str,
    cfg: Config,
    conn,
    client: LLMClient | None,
) -> dict:
    """Run the ``clip_plans`` stage for a job (M5C-5).

    Args:
        job_id: The job ID (must exist in the DB, with a ``selection.json``
            from the M5C-4 ``selection`` stage).
        cfg: Configuration object (data_dir, llm.*).
        conn: sqlite3 connection over the clipper database.
        client: LLM client for the ONE metadata call (``OllamaLLMClient``
            or ``MockLLM``); ``None`` => deterministic fallback metadata
            for all selected clips (logged warning, job still succeeds).

    Returns:
        Summary dict: job_id, plans, metadata_source (``"llm"`` /
        ``"fallback"`` / ``None`` when zero selected), fallback_reason,
        skipped, artifact (path to clip_plans.json).

    Raises:
        StageError: Clean error on a NON-metadata failure (missing job /
            inputs, schema mismatch, write failure). The stage and job are
            marked failed. A metadata failure NEVER raises — it falls back.
    """
    row = conn.execute(
        "SELECT niche FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise StageError(f"Job {job_id} not found in database")
    niche = (row[0] or "").strip()

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    artifact_path = job_dir / _PLANS_ARTIFACT_NAME

    summary: dict = {
        "job_id": job_id,
        "plans": 0,
        "metadata_source": None,
        "fallback_reason": None,
        "skipped": False,
        "artifact": str(artifact_path),
    }

    skip, stored = _is_done(conn, job_id, _PLANS_STAGE_NAME)
    if skip:
        with open(stored, encoding="utf-8") as f:
            data = json.load(f)
        summary["plans"] = len(data.get("plans", []))
        summary["skipped"] = True
        return summary

    _set_job_state(conn, job_id, "running")
    try:
        _start_stage(conn, job_id, _PLANS_STAGE_NAME)

        selection, candidates, sentences = _load_plans_inputs(job_dir)
        selected = [d for d in selection.decisions if d.selected]

        if not selected:
            # ZERO selected: empty artifact, NO LLM call, VALID success.
            artifact = ClipPlansArtifact(
                schema_version=_SCHEMA_VERSION, plans=[]
            )
            _write_artifact(artifact_path, artifact)
            _finish_stage(
                conn, job_id, _PLANS_STAGE_NAME, str(artifact_path)
            )
            _set_job_state(conn, job_id, "queued")
            return summary

        candidates_by_id = {c.cand_id: c for c in candidates}
        ranked = _rank_selected(selected, candidates_by_id)
        cuts = {
            d.cand_id: (d.refined_start, d.refined_end) for d in ranked
        }
        metadata = _metadata_for_selected(
            cfg, niche, ranked, sentences, cuts, client, summary
        )
        if summary.get("metadata_source") != "llm":
            summary["metadata_source"] = "fallback"

        plans = _build_plans(job_id, ranked, metadata)
        artifact = ClipPlansArtifact(
            schema_version=_SCHEMA_VERSION, plans=plans
        )
        _write_artifact(artifact_path, artifact)

        _upsert_clips(conn, job_id, plans)

        _finish_stage(
            conn, job_id, _PLANS_STAGE_NAME, str(artifact_path)
        )
        _set_job_state(conn, job_id, "queued")

        summary["plans"] = len(plans)
        return summary

    except Exception as e:
        clean = e if isinstance(e, StageError) else StageError(
            f"clip_plans stage failed: {e}"
        )
        try:
            _finish_stage(conn, job_id, _PLANS_STAGE_NAME, error=str(clean))
        except Exception:
            pass
        try:
            _set_job_state(conn, job_id, "failed")
        except Exception:
            pass
        if clean is e:
            raise
        raise clean from e


__all__ = [
    "METADATA_NUM_PREDICT_CAP",
    "run_clip_plans_stage",
    "run_selection",
    "run_selection_stage",
]
