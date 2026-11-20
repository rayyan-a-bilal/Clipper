"""Selection / draft ClipPlan contracts for the M5c stage (additive).

Pure CPU data contracts; no selection or cut logic lives here. Existing
models in `clipper.models` are reused (incl. `ClipPlan`), not modified.

- `ClipMetadata` — one selected clip's LLM metadata (title/hook/description/
  hashtags), the unit the metadata call returns per clip.
- `MetadataResponse` — exactly what the model must return for the metadata
  call: one `ClipMetadata` per selected clip (ordered, same order as input).
- `ClipDecision` — the per-candidate decision record written by the
  `selection` stage (gate, suppression, selection, cut rejection).
- `SelectionArtifact` — the `selection.json` artifact (schema_version=1).
- `ClipPlansArtifact` — the `clip_plans.json` artifact (schema_version=1),
  holding DRAFT `ClipPlan`s (center_crop / no crop_path / pending_qc; M6
  replaces layout/crop_path).
"""

from pydantic import BaseModel, Field

from . import ClipPlan


class ClipMetadata(BaseModel):
    """Metadata for one selected clip (bounded, validated by the stage)."""

    cand_id: str
    title: str
    hook_overlay: str
    description: str
    hashtags: list[str]


class MetadataResponse(BaseModel):
    """What the model returns for the metadata call: one item per clip."""

    items: list[ClipMetadata]


class ClipDecision(BaseModel):
    """Per-candidate decision record (selection.json row).

    - ``gate_reasons``: why the candidate was ineligible (empty when
      eligible); covers scored-out / axis / gate violations.
    - ``suppressed_by``: cand_id of the already-selected clip whose temporal
      IoU (> 0.30) suppressed this one, if any.
    - ``cut_reject_reason``: specific :class:`clipper.cuts.CutError` reason
      when the suggestion could not be refined into a valid cut (M5c).
    - ``refined_start`` / ``refined_end``: the final ``(start, end)`` cut
      computed by ``clipper.cuts.refine_cut`` for SELECTED candidates only
      (``None`` otherwise). Persisted so the later ``clip_plans`` stage
      (M5C-5) can reuse the exact refined cut without re-running the
      refinement (M5C-4, additive; absent in older ``selection.json``).
    """

    cand_id: str
    weighted: float
    eligible: bool
    gate_reasons: list[str] = Field(default_factory=list)
    suppressed_by: str | None = None
    selected: bool = False
    cut_reject_reason: str | None = None
    refined_start: float | None = None
    refined_end: float | None = None


class SelectionArtifact(BaseModel):
    """selection.json artifact (schema_version=1)."""

    schema_version: int = 1
    decisions: list[ClipDecision]


class ClipPlansArtifact(BaseModel):
    """clip_plans.json artifact (schema_version=1).

    Holds DRAFT `ClipPlan`s: `layout = center_crop`, `crop_path = []`,
    `status = pending_qc`. M6 (reframe) replaces layout/crop_path.
    Zero plans is a VALID artifact.
    """

    schema_version: int = 1
    plans: list[ClipPlan]
