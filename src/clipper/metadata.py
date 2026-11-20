"""M5c — clip metadata: pure validation, deterministic fallback, request builder.

M5C-3 scope (PURE — no I/O, no LLM, no stage/script wiring):

- `validate_metadata(response, selected_cand_ids)` — enforce the metadata
  contract exactly: one `ClipMetadata` per selected `cand_id`, no extras or
  duplicates, all field limits (title / hook_overlay / description /
  hashtags) bounded. Structural violations raise `ValueError` (the future
  `clip_plans` stage uses this to trigger LLM retry).
- `fallback_metadata(cand_id, transcript, niche)` — deterministic, within-limits
  fallback used after retries are exhausted. Metadata failure must NEVER
  fail the job (docs/spec/m5.md, M5 rules).
- `build_metadata_request(selected)` — compact JSON request payload
  containing ONLY `cand_id`, `transcript`, `niche`, `duration_s` per clip.
  The full-video transcript must never appear here: the metadata call sees
  only the selected clips' final transcripts.

Field limits mirror `prompts.py::METADATA_SYSTEM` (single source of the
prompt; these constants are the single source of code-side enforcement).
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from .models import ClipMetadata, MetadataResponse

# --- Field limits (must match prompts.METADATA_SYSTEM) ---------------------
TITLE_MAX = 80
HOOK_MAX_WORDS = 6
HOOK_MAX_CHARS = 40
DESCRIPTION_MAX = 300
HASHTAG_MIN = 3
HASHTAG_MAX = 6

# --- Fallback shapes (deterministic, within limits) ------------------------
FALLBACK_TITLE_WORDS = 8
FALLBACK_HOOK_WORDS = 5
FALLBACK_DESCRIPTION_CHARS = 200

# Whitelist of per-clip fields allowed in the metadata request payload.
REQUEST_CLIP_KEYS = ("cand_id", "duration_s", "niche", "transcript")


def _check_text(cand_id: str, name: str, value: str, maximum: int) -> None:
    """Non-empty, within `maximum` chars — ValueError otherwise."""
    if not value.strip():
        raise ValueError(f"{cand_id}: {name} is empty")
    if len(value) > maximum:
        raise ValueError(
            f"{cand_id}: {name} is {len(value)} chars (max {maximum})"
        )


def _check_hook(cand_id: str, value: str) -> None:
    if (
        not value.strip()
        or len(value.split()) > HOOK_MAX_WORDS
        or len(value) > HOOK_MAX_CHARS
    ):
        raise ValueError(
            f"{cand_id}: hook_overlay must be 1-{HOOK_MAX_WORDS} words "
            f"and <= {HOOK_MAX_CHARS} chars"
        )


def _check_hashtags(cand_id: str, hashtags: list[str]) -> None:
    if not HASHTAG_MIN <= len(hashtags) <= HASHTAG_MAX:
        raise ValueError(
            f"{cand_id}: {len(hashtags)} hashtags "
            f"({HASHTAG_MIN}-{HASHTAG_MAX} required)"
        )
    for tag in hashtags:
        if len(tag) < 2 or not tag.startswith("#") or any(c.isspace() for c in tag):
            raise ValueError(
                f"{cand_id}: invalid hashtag {tag!r} "
                "(must start with '#' and contain no spaces)"
            )


def validate_metadata(
    response: MetadataResponse, selected_cand_ids: list[str]
) -> list[ClipMetadata]:
    """Validate a metadata response against the selected clips.

    Requires EXACTLY one item per selected `cand_id` (no extras, no
    duplicates, no omissions) and enforces every field limit.
    Returns the items as a list; raises `ValueError` on any violation.
    """
    items = response.items
    if len(items) != len(selected_cand_ids):
        raise ValueError(
            f"metadata returned {len(items)} item(s), "
            f"expected {len(selected_cand_ids)}"
        )

    by_id: dict[str, ClipMetadata] = {}
    for item in items:
        if item.cand_id in by_id:
            raise ValueError(f"duplicate metadata for cand_id {item.cand_id!r}")
        by_id[item.cand_id] = item

    expected = set(selected_cand_ids)
    missing = sorted(expected - set(by_id))
    extra = sorted(set(by_id) - expected)
    if missing or extra:
        raise ValueError(
            f"metadata clip mismatch: missing={missing}, extra={extra}"
        )

    for item in items:
        _check_text(item.cand_id, "title", item.title, TITLE_MAX)
        _check_hook(item.cand_id, item.hook_overlay)
        _check_text(item.cand_id, "description", item.description, DESCRIPTION_MAX)
        _check_hashtags(item.cand_id, item.hashtags)
    return items


def fallback_metadata(cand_id: str, transcript: str, niche: str) -> ClipMetadata:
    """Deterministic within-limits metadata fallback (no LLM).

    - title: first 8 words (char-clipped to TITLE_MAX as a safety net)
    - hook_overlay: first 5 words (char-clipped to HOOK_MAX_CHARS)
    - description: first 200 chars (always <= DESCRIPTION_MAX)
    - hashtags: ['#<niche>'] when niche is non-empty, else ['#clip']
    """
    words = transcript.split()
    return ClipMetadata(
        cand_id=cand_id,
        title=" ".join(words[:FALLBACK_TITLE_WORDS])[:TITLE_MAX],
        hook_overlay=" ".join(words[:FALLBACK_HOOK_WORDS])[:HOOK_MAX_CHARS],
        description=transcript[:FALLBACK_DESCRIPTION_CHARS],
        hashtags=[f"#{niche}"] if niche.strip() else ["#clip"],
    )


def build_metadata_request(selected: list[Mapping]) -> str:
    """Compact JSON request payload for the metadata call.

    `selected` is a list of per-clip mappings; ONLY the whitelisted keys
    (`cand_id`, `transcript`, `niche`, `duration_s`) are carried through —
    anything else (e.g. the full video transcript) is dropped. Input order
    is preserved; the output is a single compact JSON string.
    """
    clips = [
        {key: clip[key] for key in REQUEST_CLIP_KEYS if key in clip}
        for clip in selected
    ]
    return json.dumps({"clips": clips}, separators=(",", ":"))
