"""M5b — pure LLM-scoring helpers (NO I/O, NO LLM calls).

Implements the fixed M5b rules from ``docs/spec/m5.md``:

- ``build_context(candidate, sentences, cfg) -> CandidateContext``
  transcript = sentences[sentence_range] joined by single spaces
  (inclusive); ``context_before`` / ``context_after`` are the sentences
  overlapping ``[start - context_before_s, start)`` /
  ``(end, end + context_after_s]`` (windows from ``cfg.llm``).
- ``estimate_tokens(text) -> max(1, len(text) // 4)`` — the cheap
  ~4-chars-per-token estimate used for batch packing.
- ``pack_batches(contexts, batch_size, max_batch_tokens)`` — greedy,
  order-preserving bin packing with a 1500-token prompt-overhead reserve;
  an oversized single candidate loses context_before/context_after
  FIRST, never the transcript.
- ``validate_scores(response, batch) -> list[ScoredClip]`` — exactly one
  score per candidate (no extras/duplicates), axes int 0..10 (structural
  violations => ``ValueError`` => batch retry); an out-of-bounds suggested
  in/out keeps the scores but sets ``reject_reason="suggestion_out_of_bounds"``.
- ``weighted_score(clip)`` — the M5b weighted formula (selection itself is
  M5c; this helper only computes the documented weights).
"""

from __future__ import annotations

from clipper.config import Config
from clipper.models import (
    Candidate,
    CandidateContext,
    ScoredClip,
    ScoringBatch,
    ScoringResponse,
)

# -- docs/spec/m5.md + task constants (module-level, do NOT drift) ------------
PROMPT_OVERHEAD_TOKENS = 1500  # system prompt + schema reserved per batch
PER_CANDIDATE_OVERHEAD_TOKENS = 60  # cand_id + bounds + fields per candidate
SUGGEST_SLACK = 8.0  # LLM suggestion may deviate from candidate by <= 8s

OOB_REJECT_REASON = "suggestion_out_of_bounds"

# Weighted-score axis weights (docs/spec/m5.md, verbatim).
W_HOOK = 0.30
W_PAYOFF = 0.25
W_SELF_CONTAINED = 0.20
W_EMOTION = 0.15
W_CLARITY = 0.10

_AXES = ("hook", "payoff", "self_contained", "emotion", "clarity")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 characters per token, minimum 1."""
    return max(1, len(text) // 4)


def build_context(
    candidate: Candidate, sentences: list, cfg: Config
) -> CandidateContext:
    """Build the LLM-facing context for one candidate.

    - transcript: ``sentences[start_idx..end_idx]`` (``Candidate.sentence_range``,
      INCLUSIVE) joined by single spaces.
    - context_before: sentences overlapping
      ``[start - cfg.llm.context_before_s, start)``.
    - context_after: sentences overlapping ``(end, end + cfg.llm.context_after_s]``.
    """
    start_idx, end_idx = candidate.sentence_range
    transcript = " ".join(
        sentences[i].text for i in range(start_idx, end_idx + 1)
    )

    before_floor = candidate.start - cfg.llm.context_before_s
    context_before = " ".join(
        s.text
        for s in sentences
        if s.end > before_floor and s.start < candidate.start
    )

    after_ceil = candidate.end + cfg.llm.context_after_s
    context_after = " ".join(
        s.text
        for s in sentences
        if s.start <= after_ceil and s.end > candidate.end
    )

    return CandidateContext(
        cand_id=candidate.cand_id,
        start=candidate.start,
        end=candidate.end,
        transcript=transcript,
        context_before=context_before,
        context_after=context_after,
    )


def _candidate_cost(ctx: CandidateContext) -> int:
    return (
        estimate_tokens(ctx.transcript)
        + estimate_tokens(ctx.context_before)
        + estimate_tokens(ctx.context_after)
        + PER_CANDIDATE_OVERHEAD_TOKENS
    )


def _fit_to_budget(ctx: CandidateContext, budget: int) -> CandidateContext:
    """Shrink context_before/context_after until the candidate fits.

    The transcript is NEVER truncated. The remaining budget is given to
    context_before first, with the remainder to context_after.
    """
    if _candidate_cost(ctx) <= budget:
        return ctx

    # Room left for the two context fields after the (untouchable)
    # transcript + overhead. Note estimate_tokens("") == 1, so an empty
    # field still costs one token; we give context_before the room first
    # and reserve at least one token for the context_after field.
    allowed = budget - estimate_tokens(ctx.transcript) - PER_CANDIDATE_OVERHEAD_TOKENS
    if allowed <= 2:
        return ctx.model_copy(update={"context_before": "", "context_after": ""})
    before_tokens = allowed - 1
    before = ctx.context_before[: min(len(ctx.context_before), before_tokens * 4)]
    remaining = max(0, allowed - estimate_tokens(before))
    after = ctx.context_after[: max(0, min(len(ctx.context_after), remaining * 4))]
    return ctx.model_copy(update={"context_before": before, "context_after": after})


def pack_batches(
    contexts: list[CandidateContext],
    batch_size: int,
    max_batch_tokens: int,
) -> list[ScoringBatch]:
    """Greedy, order-preserving batch packing.

    A batch starts a new entry when it already holds ``batch_size`` candidates
    OR adding the next candidate would exceed
    ``max_batch_tokens - PROMPT_OVERHEAD_TOKENS``. Every batch holds >= 1
    candidate. A single candidate that exceeds the whole budget first loses
    context_before/context_after (never the transcript).
    """
    budget = max(1, max_batch_tokens - PROMPT_OVERHEAD_TOKENS)

    batches: list[ScoringBatch] = []
    batch_costs: list[int] = []
    for ctx in contexts:
        ctx = _fit_to_budget(ctx, budget)
        cost = _candidate_cost(ctx)
        if (
            not batches
            or len(batches[-1].candidates) >= batch_size
            or batch_costs[-1] + cost > budget
        ):
            batches.append(ScoringBatch(schema_version=1, candidates=[ctx]))
            batch_costs.append(cost)
        else:
            batches[-1].candidates.append(ctx)
            batch_costs[-1] += cost
    return batches


def validate_scores(
    response: ScoringResponse, batch: ScoringBatch
) -> list[ScoredClip]:
    """Validate one model response against its batch.

    Structural violations raise ``ValueError`` (the caller retries the batch):
    - not exactly one score per ``cand_id`` in the batch (missing, extra, or
      duplicate ids);
    - any axis not an ``int`` in ``0..10``;
    - a score entry that is not a ``ScoredClip``.

    A non-structural issue does NOT raise: ``suggested_in/out`` outside
    ``[cand.start - SUGGEST_SLACK, cand.end + SUGGEST_SLACK]`` keeps the
    scores but sets ``reject_reason = OOB_REJECT_REASON``.

    Returns the scored clips in BATCH CANDIDATE ORDER.
    """
    if not isinstance(response, ScoringResponse) or not isinstance(
        response.scores, list
    ):
        raise ValueError("response is not a ScoringResponse with a scores list")

    by_id: dict[str, ScoredClip] = {}
    for s in response.scores:
        if not isinstance(s, ScoredClip):
            raise ValueError(f"score entry is not a ScoredClip: {s!r}")
        if s.cand_id in by_id:
            raise ValueError(f"duplicate score for cand_id {s.cand_id!r}")
        by_id[s.cand_id] = s

    expected = [c.cand_id for c in batch.candidates]
    expected_set = set(expected)
    missing = [cid for cid in expected if cid not in by_id]
    extra = [cid for cid in by_id if cid not in expected_set]
    if missing or extra:
        raise ValueError(f"batch cand_id mismatch: missing={missing}, extra={extra}")

    out: list[ScoredClip] = []
    for cand in batch.candidates:
        s = by_id[cand.cand_id]
        for axis in _AXES:
            value = getattr(s, axis)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 10
            ):
                raise ValueError(
                    f"{axis}={value!r} out of range for {cand.cand_id!r} (int 0..10)"
                )
        lower = cand.start - SUGGEST_SLACK
        upper = cand.end + SUGGEST_SLACK
        out_of_bounds = (
            s.suggested_in < lower
            or s.suggested_in > upper
            or s.suggested_out < lower
            or s.suggested_out > upper
        )
        if out_of_bounds:
            s = s.model_copy(update={"reject_reason": OOB_REJECT_REASON})
        out.append(s)
    return out


def weighted_score(clip: ScoredClip) -> float:
    """M5b weighted score: 0.30*hook + 0.25*payoff + 0.20*self_contained
    + 0.15*emotion + 0.10*clarity (docs/spec/m5.md)."""
    return (
        W_HOOK * clip.hook
        + W_PAYOFF * clip.payoff
        + W_SELF_CONTAINED * clip.self_contained
        + W_EMOTION * clip.emotion
        + W_CLARITY * clip.clarity
    )


__all__ = [
    "PROMPT_OVERHEAD_TOKENS",
    "PER_CANDIDATE_OVERHEAD_TOKENS",
    "SUGGEST_SLACK",
    "OOB_REJECT_REASON",
    "weighted_score",
    "estimate_tokens",
    "build_context",
    "pack_batches",
    "validate_scores",
]
