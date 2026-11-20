# M5C-0 — Audit Status (2026-11-20)

Audit-only pass (M5C-0). No new features implemented. M5b complete (commit
`2751918`); all M5c work is present on disk but **uncommitted**
(`git status`: modified `models/__init__.py`, `prompts.py`, `stages/__init__.py`;
untracked `src/clipper/selection.py`, `src/clipper/cuts.py`,
`src/clipper/models/selection_artifacts.py`, `src/clipper/stages/select.py`,
`scripts/select_job.py`, `tests/test_selection.py`, `tests/test_cuts.py`).

## Implemented (all present, all tests green)

| Piece | File | State |
|---|---|---|
| Pure selection (rank/gate/dedupe, no LLM) | `src/clipper/selection.py` | done — delegates weights to `scoring.weighted_score`, re-exports M4 `temporal_iou`; decisions for ALL candidates in candidate order |
| Pure cut refinement (no LLM) | `src/clipper/cuts.py` | done — all 6 spec constants verbatim (8.0 / 1.0 / 0.120 / 0.250 / 0.300 / 0.400), strict order clamp → sentence snap → word padding → silence snap → shot snap → validate 15..75s LAST, `CutError` with specific reason |
| Metadata contracts | `src/clipper/models/selection_artifacts.py` | done — `ClipMetadata`, `MetadataResponse`, `ClipDecision`, `SelectionArtifact`, `ClipPlansArtifact`; registered in `models/__init__.py` |
| `METADATA_SYSTEM` prompt | `src/clipper/prompts.py` | done — bounds match the stage (title ≤80; hook ≤6 words & ≤40 chars; description ≤300; 3-6 hashtags `#`-prefixed, no spaces) |
| Pure metadata helpers (M5C-3) | `src/clipper/metadata.py` | done — `validate_metadata` / `fallback_metadata` / `build_metadata_request` (pure, constants `TITLE_MAX`/`HOOK_MAX_*`/`DESCRIPTION_MAX`/`HASHTAG_*`/`FALLBACK_*` mirror the prompt; 35 CPU tests in `tests/test_metadata.py`) |
| Stage orchestration | `src/clipper/stages/select.py` | done — 2 stages (`selection`, `clip_plans`); CutError → `cut_reject_reason` + backfill from next eligible rank-ordered candidate (IoU-checked); metadata retry (`max_retries+1` attempts) → deterministic fallback + warning (job NEVER fails on metadata); zero clips → empty `ClipPlansArtifact`, NO LLM call, job success |
| Run script | `scripts/select_job.py` | done — selection/cuts first with no LLM; managed ephemeral Ollama spawned ONLY when ≥1 clip selected and not `--mock`; `--mock` = MockLLM canned metadata |
| Tests | `tests/test_selection.py` (17), `tests/test_cuts.py` (14) | all 31 pass, CPU-only |

## Missing

- **Stage-level tests** (no `tests/test_selection_stage.py`): not covered by
  tests — backfill-after-CutError through `run_selection`, the zero-clip
  success path incl. "no LLM call / no Ollama", metadata fallback after
  retries, metadata validation rejections, stage skip-idempotency, `clips`
  table upsert. Unit layers (selection/cuts) are fully tested; the glue is not.

## Failing

- **Tests: none.** 224/224 pass (`pytest -m "not gpu"`, 4m14s). The 33
  selection/cut/metadata-matching tests all pass; collection is clean
  (224 collected, no syntax/import errors).
- **Ruff: 19 style errors** (the DoD "ruff clean" is not met). ALL are in
  new M5c files, none in `src/`:
  - `scripts/select_job.py:107` E501
  - `tests/test_cuts.py`: I001, F401 (`Config`, `PathsConfig` imported unused)
  - `tests/test_selection.py`: I001, F841 (unused `w`), C408, 14× E501
  - 4 auto-fixable with `ruff --fix` (I001/F401); the rest are E501/F841/C408
    line reflows — mechanical, no logic change needed.

## Drift vs docs/spec/m5.md

| Spec requirement | Verdict |
|---|---|
| selection/cuts must not call LLM | OK — `selection.py`/`cuts.py` import no LLM; the only `generate_json` call is in the `clip_plans` stage for metadata |
| metadata failure must fall back, not fail the job | OK — retry-then-deterministic fallback (`_metadata_call`), warning logged, stage completes |
| gate weights/thresholds | OK — 0.30/0.25/0.20/0.15/0.10 in `scoring.py`; gate 7.2 / hook 6 / self_contained 7 in `selection.py` (1e-9 float epsilon at the 7.2 boundary, benign) |
| cut constants & order | OK — all constants verbatim; strict order incl. last `15 <= dur <= 75` validation |
| backfill after CutError | OK — `stages/select.py::_compute_selection` walks `result.ranked`, on `CutError` records `cut_reject_reason` and continues to the next eligible non-overlapping candidate |
| zero-clip success path | OK — empty `ClipPlansArtifact`, no LLM, exit 0 in the script |
| selection.json decisions for ALL candidates | OK — one `ClipDecision` per candidate in original order (gate/suppressed/selected/cut reasons) |

Two judgment notes (not spec violations, but discretionary extensions worth
flagging):

1. `cuts.py::_snap_sentence_start/_end` extend the spec's "snap to nearest
   sentence boundary within 1.0s" with a fallback (containing sentence, then
   nearest sentence) when nothing is within `SENTENCE_SNAP`. Spec doesn't
   define beyond-1.0s behavior; the code always snaps to *some* sentence,
   which can move a boundary more than 1.0s before word padding re-anchors
   it. Tests encode this behavior.
2. Backfill bookkeeping: `suppressed_by` is computed in `select_clips`
   against the initial pick list; if a picked candidate later dies with
   `CutError` and a previously-suppressed one is backfilled in, that
   candidate's decision row can carry BOTH `suppressed_by` set AND
   `selected=True`. Cosmetic/audit-trail only — final selects and cuts are
   correct; the script's human summary may under-report such a clip under
   "suppressed".

## Timeout source

- No evidence in the code of a misconfigured HTTP timeout: the httpx client
  is built with `timeout=cfg.llm.timeout_s` (`adapters/llm.py` lines
  50-51/78; default 300s, `llm.md` conformance).
- The metadata prompt contains ONLY selected-clip transcripts: each clip's
  text is derived from `cand.sentence_range` over `sentences.json`
  (`_clip_transcript`), not the full video transcript. Prompt size is thus
  proportional to selected-clip count — bounded.
- Since the M5c code is presently COMPLETE and the suite is green, an HTTP/
  Ollama timeout inside clipper is not the blocker. The most probable
  cause of the repeated "M5C failure" is the **PI harness session/
  code-generation timeout**: the whole M5c surface is one large uncommitted
  WIP spread over 7+ files, so a single interrupted generation round trips
  the work back to "partially implemented". (Prior-session logs are not
  visible from this audit session, so this is inference, not a captured
  stack trace.)

## Recommended next chunk (M5C-1, no new features)

1. `ruff --fix` + manual E501/F841/C408 cleanups in the 3 flagged files,
   verify `ruff check .` clean.
2. Add `tests/test_selection_stage.py` (CPU-only, MockLLM/monkeypatch):
   - backfill: top pick raises `CutError` → next eligible selected,
     `cut_reject_reason` recorded, target still met
   - zero-clip: all candidates gate-fail → `clip_plans.json` empty, `plans==0`,
     client never invoked (assert `MockLLM.calls == []`), job not failed
   - metadata: bad-then-good `MockLLM` queue → retry succeeds; all-bad queue
     → deterministic fallback used, stage still `done`
   - metadata validation: wrong item count / empty title / >80 chars /
     bad hashtag → ValueError retry path
   - stage skip: re-run with artifacts present → both stages skipped, no writes
   - `clips` table upsert shape (clip_id `{job_id}-clipNN`, DRAFT fields)
3. Commit M5c (implementation + tests) once both are green.

Explicitly NOT to do: start M6; implement the decision-flag cosmetic fix or
the beyond-1.0s snap change without a spec decision first.
