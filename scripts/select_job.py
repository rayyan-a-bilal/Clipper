#!/usr/bin/env python3
"""
Manual selection + draft ClipPlan script for clipper jobs (M5C-5).

Runs the M5c stages on an already-scored job:

  1. ``selection``  — rank/gate/dedupe (clipper.selection) + cut refinement
     with backfill (clipper.cuts). NO LLM needed (run FIRST).
  2. ``clip_plans`` — ONE metadata call for all selected clips, then DRAFT
     ClipPlans (center_crop / no crop_path / pending_qc) + `clips` rows.

Client selection (only consulted when >= 1 clip was selected):

  --no-llm : NO Ollama spawn, NO client — the clip_plans stage uses the
             deterministic fallback metadata for every clip (warning).
  --mock   : NO Ollama spawn — MockLLM with canned metadata (wiring test
             ONLY, never a real judgement).
  default  : ONE managed ephemeral Ollama (docs/spec/llm.md; NEVER the
             main service on 11434). If the managed runtime fails to start
             (LLMError), the script logs a warning and reruns the stage
             with NO client (fallback metadata) — it does NOT fail the job.

With ZERO selected clips the stage writes an empty clip_plans.json and
Ollama is NEVER spawned (zero clips is a VALID success — exit 0).

Usage:
    python scripts/select_job.py <job-id>
    python scripts/select_job.py <job-id> --mock
    python scripts/select_job.py <job-id> --no-llm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from clipper.adapters.llm import LLMError, MockLLM, get_llm_client
from clipper.config import Config
from clipper.db import Database
from clipper.llm_runtime import managed_ollama
from clipper.metadata import fallback_metadata
from clipper.models import ClipMetadata, MetadataResponse, SelectionArtifact
from clipper.stages.select import run_clip_plans_stage, run_selection_stage

log = logging.getLogger("clipper.select_job")

# --mock canned hashtags: 3 valid tags (validate_metadata requires 3-6).
_MOCK_HASHTAGS = ["#clipper", "#shorts", "#demo"]


def _selected_decisions(job_dir: Path) -> list:
    sel = SelectionArtifact.model_validate_json(
        (job_dir / "selection.json").read_text(encoding="utf-8")
    )
    return [d for d in sel.decisions if d.selected]


def _sentence_text_range(job_dir: Path) -> list[dict]:
    return json.loads(
        (job_dir / "sentences.json").read_text(encoding="utf-8")
    )["sentences"]


def _clip_transcript(sentences: list[dict], start: float, end: float) -> str:
    return " ".join(
        s["text"] for s in sentences if s["start"] < end and s["end"] > start
    )


def _mock_metadata_client(job_dir: Path, niche: str) -> MockLLM:
    """``--mock``: MockLLM with canned metadata (wiring test only).

    Per selected clip (rank order): the deterministic M5C-3 fallback shape
    (first 8 words title, first 5 words hook, first 200 chars description)
    with the fixed 3-tag ``#clipper #shorts #demo`` hashtag set (the
    fallback's single-``#niche`` set does not satisfy the 3-6 hashtag
    validation the stage applies to LLM metadata). Zero network I/O.
    """
    selected = _selected_decisions(job_dir)
    if not selected:
        return MockLLM()
    sentences = _sentence_text_range(job_dir)

    items: list[ClipMetadata] = []
    for d in selected:
        if not (d.refined_start is not None and d.refined_end is not None):
            continue
        base = fallback_metadata(
            d.cand_id,
            _clip_transcript(sentences, d.refined_start, d.refined_end),
            niche,
        )
        items.append(
            ClipMetadata(
                cand_id=base.cand_id,
                title=base.title,
                hook_overlay=base.hook_overlay,
                description=base.description,
                hashtags=list(_MOCK_HASHTAGS),
            )
        )
    client = MockLLM()
    client.enqueue(MetadataResponse(items=items))
    return client


def _print_selected(job_dir: Path) -> None:
    plans_path = job_dir / "clip_plans.json"
    if not plans_path.exists():
        print("clip_plans.json not found")
        return
    plans = json.loads(plans_path.read_text(encoding="utf-8"))["plans"]
    if not plans:
        print("selected: 0 clip(s) — VALID success (no clips fabricated)")
        return
    print(f"selected: {len(plans)} clip(s)")
    for p in plans:
        dur = p["end"] - p["start"]
        print(
            f"  {p['clip_id']}  [{p['start']:.2f}-{p['end']:.2f}]  "
            f"{dur:.1f}s  w={p['weighted_score']:.2f}  {p['title']!r}"
        )


def _print_rejections(job_dir: Path) -> None:
    """Rejection summary: gate failures, suppressions, cut failures."""
    sel = SelectionArtifact.model_validate_json(
        (job_dir / "selection.json").read_text(encoding="utf-8")
    )
    not_sel = [d for d in sel.decisions if not d.selected]
    if not sel.decisions:
        print("rejections: none recorded (no candidates scored)")
        return
    if not not_sel:
        print("rejections: none (all candidates selected)")
        return
    gates = [d for d in not_sel if not d.eligible]
    supps = [d for d in not_sel if d.suppressed_by is not None]
    cuts = [d for d in not_sel if d.cut_reject_reason is not None]
    if gates:
        print("gate failures:")
        for d in gates:
            print(
                f"  {d.cand_id}: {'; '.join(d.gate_reasons) or 'ineligible'}"
            )
    if supps:
        print("suppressed (temporal IoU > 0.30):")
        for d in supps:
            print(f"  {d.cand_id} <- {d.suppressed_by}")
    if cuts:
        print("cut failures (refine_cut CutError):")
        for d in cuts:
            print(f"  {d.cand_id}: {d.cut_reject_reason}")
    left = [
        d
        for d in not_sel
        if d.eligible
        and d.suppressed_by is None
        and d.cut_reject_reason is None
    ]
    if left:
        print(
            f"not selected: {len(left)} further eligible candidate(s) "
            "beyond target/cap"
        )


def _print_selection_note(job_dir: Path) -> None:
    sel = SelectionArtifact.model_validate_json(
        (job_dir / "selection.json").read_text(encoding="utf-8")
    )
    decisions = sel.decisions
    selected = sum(1 for d in decisions if d.selected)
    eligible = sum(1 for d in decisions if d.eligible)
    print(f"selected: {selected}/{len(decisions)} candidates (eligible: {eligible})")


def main(
    job_id: str, use_mock: bool = False, no_llm: bool = False,
) -> int:
    """Run the M5c selection + clip_plans stages and print a summary.

    Args:
        job_id: The job ID to process.
        use_mock: --mock => canned metadata via MockLLM (NO spawn).
        no_llm: --no-llm => no client at all (fallback metadata, NO spawn).

    Returns:
        0 on success (INCLUDING zero selected clips and ANY metadata
        fallback), 1 on non-metadata failure (missing job/inputs, ...).
    """
    cfg = Config.load_config()

    db = Database(cfg)
    job = db.get_job(job_id)
    if job is None:
        print(f"Error: job {job_id} not found")
        return 1
    niche = (job.get("niche") or "").strip()

    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    conn = db.get_connection()
    try:
        # Stage 1: selection + refined cuts FIRST (NO LLM).
        sel_summary = run_selection_stage(job_id, cfg, conn)
        if sel_summary.get("skipped"):
            print("note:     selection stage already done (skipped)")

        selected = _selected_decisions(job_dir)

        # Stage 2: clip_plans (ONE metadata call iff >= 1 selected).
        if selected and not no_llm and not use_mock:
            # ONE managed ephemeral Ollama (docs/spec/llm.md): killed again
            # on success AND failure. A startup failure is NOT a job
            # failure: fall back to deterministic metadata.
            try:
                with managed_ollama(cfg) as base_url:
                    client = get_llm_client(cfg, base_url)
                    plan_summary = run_clip_plans_stage(
                        job_id, cfg, conn, client
                    )
            except LLMError as e:
                log.warning(
                    "managed Ollama unavailable (%s) — running clip_plans "
                    "with fallback metadata; the job does NOT fail", e,
                )
                print(f"warning:  {e}")
                print("note:     using deterministic fallback metadata")
                plan_summary = run_clip_plans_stage(job_id, cfg, conn, None)
        else:
            client = (
                _mock_metadata_client(job_dir, niche) if use_mock else None
            )
            plan_summary = run_clip_plans_stage(job_id, cfg, conn, client)

        print(f"job:      {job_id}")
        if use_mock:
            print("note:     --mock canned metadata (wiring test only)")
        if no_llm:
            print("note:     --no-llm deterministic fallback metadata")
        if plan_summary.get("skipped"):
            print("note:     clip_plans stage already done (skipped)")
        if plan_summary.get("fallback_reason"):
            print(
                "note:     fallback metadata "
                f"({plan_summary['fallback_reason']})"
            )
        print(f"selection:  {job_dir / 'selection.json'}")

        _print_selection_note(job_dir)
        _print_selected(job_dir)
        _print_rejections(job_dir)
        print(f"clip_plans: {job_dir / 'clip_plans.json'}")
        return 0
    except Exception as e:
        print(f"Error: M5c selection/clip_plans failed: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the M5c selection + clip_plans stages for a clipper job"
    )
    parser.add_argument("job_id", help="Job ID to run selection on")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Canned metadata via MockLLM (wiring test only, no Ollama spawn)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="No LLM client at all: deterministic fallback metadata, no spawn",
    )
    args = parser.parse_args()
    if args.mock and args.no_llm:
        parser.error("--mock and --no-llm are mutually exclusive")
    sys.exit(main(args.job_id, use_mock=args.mock, no_llm=args.no_llm))
