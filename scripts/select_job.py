#!/usr/bin/env python3
"""
Manual selection + draft ClipPlan script for clipper jobs (M5c).

Runs the M5c stages on an already-scored job:

  1. ``selection``  — rank/gate/dedupe (clipper.selection) + cut refinement
     with backfill (clipper.cuts). NO LLM needed.
  2. ``clip_plans`` — ONE metadata call for all selected clips, then DRAFT
     ClipPlans (center_crop / no crop_path / pending_qc) + `clips` rows.

Default path (LLM metadata):
    selection/cuts run FIRST with no LLM. ONLY if at least one clip was
    selected, ONE managed ephemeral Ollama process (docs/spec/llm.md) is
    spawned, the metadata call runs against it, and the child is killed
    again (success AND failure). With zero selected clips Ollama is
    NEVER spawned (zero clips is a VALID success — exit 0).

Mock path (--mock):
    SELECTION test double ONLY — a MockLLM with canned metadata derived
    from each selected clip's transcript (first 8 words title, first 5
    words hook, first 200 chars description, fixed hashtags). Never a
    real judgement.

Usage:
    python scripts/select_job.py <job-id>
    python scripts/select_job.py <job-id> --mock
"""

import argparse
import json
import sys
from pathlib import Path

from clipper.adapters.llm import LLMClient, MockLLM, get_llm_client
from clipper.config import Config
from clipper.db import Database
from clipper.llm_runtime import managed_ollama
from clipper.models import (
    ClipMetadata,
    MetadataResponse,
    SelectionArtifact,
)
from clipper.stages.select import run_selection


def _mock_metadata_client(job_dir: Path) -> LLMClient:
    """``--mock``: MockLLM with canned metadata (wiring test only).

    Canned per selected clip (in rank order): title = first 8 transcript
    words, hook_overlay = first 5 words, description = first 200 chars,
    hashtags = fixed deterministic set. Zero network I/O.
    """
    decisions = SelectionArtifact.model_validate_json(
        (job_dir / "selection.json").read_text(encoding="utf-8")
    ).decisions
    selected = [d for d in decisions if d.selected]
    if not selected:
        return MockLLM()

    cands = json.loads(
        (job_dir / "candidates.json").read_text(encoding="utf-8")
    )
    sents = json.loads(
        (job_dir / "sentences.json").read_text(encoding="utf-8")
    )
    by_id = {c["cand_id"]: c for c in cands["candidates"]}
    texts = sents["sentences"]

    items: list[ClipMetadata] = []
    for d in selected:
        c = by_id[d.cand_id]
        start_idx, end_idx = c["sentence_range"]
        transcript = " ".join(
            t["text"] for t in texts[start_idx : end_idx + 1]
        )
        words = transcript.split()
        items.append(
            ClipMetadata(
                cand_id=d.cand_id,
                title=" ".join(words[:8]),
                hook_overlay=" ".join(words[:5]),
                description=transcript[:200],
                hashtags=["#clipper", "#shorts", "#demo"],
            )
        )
    client = MockLLM()
    client.enqueue(MetadataResponse(items=items))
    return client


def _print_results(job_id: str, summary: dict, job_dir: Path, use_mock: bool) -> None:
    """Print selected clips + rejection breakdown (for humans)."""
    sel = SelectionArtifact.model_validate_json(
        (job_dir / "selection.json").read_text(encoding="utf-8")
    )
    plans_artifact_path = job_dir / "clip_plans.json"
    if not plans_artifact_path.exists():
        print("clip_plans.json not found")
        return

    plans = json.loads(plans_artifact_path.read_text(encoding="utf-8"))["plans"]

    print(f"job:      {job_id}")
    print(f"model:    {summary.get('model')}")
    if summary.get("skipped"):
        skipped = ", ".join(summary["skipped"])
        print(f"note:     stage(s) already done, skipped: {skipped}")
    if use_mock:
        print("note:     --mock canned metadata (wiring test only)")

    if plans:
        print(f"selected: {len(plans)} clip(s)")
        for p in plans:
            dur = p["end"] - p["start"]
            print(
                f"  {p['clip_id']}  [{p['start']:.2f}-{p['end']:.2f}]  "
                f"{dur:.1f}s  w={p['weighted_score']:.2f}  {p['title']!r}"
            )
    else:
        print("selected: 0 clip(s) — VALID success (no clips fabricated)")

    not_sel = [d for d in sel.decisions if not d.selected]
    gates = [d for d in not_sel if not d.eligible]
    supps = [d for d in not_sel if d.suppressed_by is not None]
    cuts = [d for d in not_sel if d.cut_reject_reason is not None]
    if not not_sel:
        if sel.decisions:
            print("rejections: none (all candidates selected)")
        else:
            print("rejections: none recorded (no candidates scored)")
    else:
        if gates:
            print("gate failures:")
            for d in gates:
                print(f"  {d.cand_id}: {'; '.join(d.gate_reasons) or 'ineligible'}")
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
            print(f"not selected: {len(left)} further eligible candidate(s) "
                  f"beyond target/cap")
    print(f"selection:  {job_dir / 'selection.json'}")
    print(f"clip_plans: {plans_artifact_path}")


def main(job_id: str, use_mock: bool = False) -> int:
    """Run the M5c stages for a job and print a short summary.

    Args:
        job_id: The job ID to process.
        use_mock: --mock => canned metadata via MockLLM (wiring test only).

    Returns:
        0 on success (INCLUDING zero selected clips), 1 on failure.
    """
    cfg = Config.load_config()

    db = Database(cfg)
    if db.get_job(job_id) is None:
        print(f"Error: job {job_id} not found")
        return 1

    conn = db.get_connection()
    job_dir = cfg.resolved_data_dir / "jobs" / job_id
    try:
        # Pass 1: selection + refined cuts FIRST (no LLM needed).
        summary = run_selection(job_id, cfg, conn, None)

        if summary.get("clip_plans") == "pending_llm":
            # >= 1 clip selected -> one metadata call (LLM stage 2).
            if use_mock:
                client = _mock_metadata_client(job_dir)
                summary = run_selection(job_id, cfg, conn, client)
            else:
                # ONE managed ephemeral Ollama for the metadata call;
                # killed again on success AND failure (docs/spec/llm.md).
                with managed_ollama(cfg) as base_url:
                    client = get_llm_client(cfg, base_url)
                    summary = run_selection(job_id, cfg, conn, client)
        _print_results(job_id, summary, job_dir, use_mock)
        return 0
    except Exception as e:
        print(f"Error: M5c selection/clip_plans failed: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the M5c selection + clip_plans stages for a clipper job"
    )
    parser.add_argument("job_id", help="Job ID to run selection on")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Canned metadata via MockLLM (wiring test only, never real judgement)",
    )
    args = parser.parse_args()
    sys.exit(main(args.job_id, use_mock=args.mock))
