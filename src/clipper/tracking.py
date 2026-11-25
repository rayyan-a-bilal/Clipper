"""Pure face tracking and track statistics (M6c — docs/spec/m6.md).

No I/O, no detector imports, no GPU, no numpy.
Unit-testable with synthetic `FaceDetection` objects only.
"""
from __future__ import annotations

from clipper.models import FaceDetection, FaceTrack, TrackSummary

# --- Module constants (M6c) ---
CORNER_AFFINITY_MIN_FRACTION = 0.8  # >= 80% of frames in same quadrant
CORNER_AFFINITY_MAX_AREA = 0.04     # mean_area must be < this
PERSISTENT_THRESHOLD = 0.6          # persistence >= 0.6 => "persistent"
FACECAM_PERSISTENCE = 0.8           # persistence >= 0.8 for facecam candidate
FACECAM_MAX_AREA = 0.04             # mean_area < 0.04 for facecam candidate
SIDE_BY_SIDE_MAX_RATIO = 2.0        # max/min area ratio for side_by_side pair


def _box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU of two (x, y, w, h) normalized boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0.0 else 0.0


def _centroid_quadrant(cx: float, cy: float) -> str:
    """Quadrant label for a normalized centroid (cx, cy) in [0, 1]."""
    if cx < 0.5:
        return "tl" if cy < 0.5 else "bl"
    return "tr" if cy < 0.5 else "br"


def associate(
    detections_by_frame: list[list[FaceDetection]],
    iou_threshold: float = 0.3,
    max_gap_frames: int = 3,
) -> list[FaceTrack]:
    """Group per-frame detections into face tracks via greedy IoU matching.

    Track IDs are assigned 0, 1, 2, … in order of first appearance.
    A track survives up to `max_gap_frames` consecutive frames without a
    match before being finalized; any re-appearance then creates a new track.

    Args:
        detections_by_frame: list of frames; each frame is a list of
            FaceDetection (normalized x/y/w/h in [0, 1]).
        iou_threshold: minimum IoU (against the track's last box) needed to
            link a detection to an existing track.
        max_gap_frames: maximum number of consecutive unmatched frames a track
            may go through before being finalized.

    Returns:
        FaceTrack objects ordered by track_id (= first-appearance order).
    """
    if not detections_by_frame:
        return []

    # Invariant: tracks_data[tid]["boxes"] is non-empty whenever tid is in keys.
    tracks_data: dict[int, dict] = {}
    active: set[int] = set()
    next_id = 0

    for detections in detections_by_frame:
        if not detections:
            for tid in list(active):
                tracks_data[tid]["gap"] += 1
                if tracks_data[tid]["gap"] > max_gap_frames:
                    active.discard(tid)
            continue

        # Greedy IoU matching against each active track's most recent box
        candidates: list[tuple[float, int, int]] = []
        for tid in sorted(active):
            boxes = tracks_data[tid]["boxes"]
            if not boxes:
                continue
            lb = boxes[-1]  # (t, x, y, w, h)
            l_xywh = (lb[1], lb[2], lb[3], lb[4])
            for d_idx, det in enumerate(detections):
                d_xywh = (det.x, det.y, det.w, det.h)
                iou = _box_iou(l_xywh, d_xywh)
                if iou >= iou_threshold:
                    candidates.append((iou, tid, d_idx))

        # Greedy: highest IoU first; tie-break (tid asc, d_idx asc)
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        matched_tids: set[int] = set()
        matched_dets: set[int] = set()
        matches: dict[int, int] = {}  # d_idx -> tid

        for _, tid, d_idx in candidates:
            if tid in matched_tids or d_idx in matched_dets:
                continue
            matched_tids.add(tid)
            matched_dets.add(d_idx)
            matches[d_idx] = tid

        for d_idx, det in enumerate(detections):
            box = (det.t, det.x, det.y, det.w, det.h)
            if d_idx in matches:
                tid = matches[d_idx]
                tracks_data[tid]["boxes"].append(box)
                tracks_data[tid]["gap"] = 0
            else:
                tid = next_id
                next_id += 1
                tracks_data[tid] = {"boxes": [box], "gap": 0}
                active.add(tid)

        # Unmatched active tracks: count the gap, finalize if exceeded
        for tid in list(active):
            if tid not in matched_tids:
                tracks_data[tid]["gap"] += 1
                if tracks_data[tid]["gap"] > max_gap_frames:
                    active.discard(tid)

    return [
        FaceTrack(
            track_id=tid,
            boxes=tracks_data[tid]["boxes"],
            persistence=0.0,
            corner_affinity=None,
        )
        for tid in sorted(tracks_data)
    ]


def track_stats(
    track: FaceTrack,
    frame_count: int,
    frame_w: int = 0,  # noqa: ARG001 — boxes are normalized; kept for API symmetry
    frame_h: int = 0,  # noqa: ARG001
) -> dict:
    """Compute persistence, mean_area, corner_affinity, and mean_cx for a track.

    Args:
        track: FaceTrack with boxes as (t, x, y, w, h) normalized tuples.
        frame_count: total frames in the vision segment.
        frame_w / frame_h: vision frame dimensions (unused; boxes are
            already normalized — kept for API symmetry).

    Returns:
        {
            "persistence": float,
            "mean_area": float,
            "corner_affinity": "tl" | "tr" | "bl" | "br" | None,
            "mean_cx": float,
        }
    """
    if not track.boxes or frame_count <= 0:
        return {
            "persistence": 0.0,
            "mean_area": 0.0,
            "corner_affinity": None,
            "mean_cx": 0.5,
        }

    n = len(track.boxes)
    persistence = n / frame_count

    areas = [b[3] * b[4] for b in track.boxes]
    mean_area = sum(areas) / n

    # Corner affinity: >= CORNER_AFFINITY_MIN_FRACTION in one quadrant
    # AND mean_area < CORNER_AFFINITY_MAX_AREA
    counts: dict[str, int] = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
    cx_sum = 0.0
    for b in track.boxes:
        cx = b[1] + b[3] / 2.0
        cy = b[2] + b[4] / 2.0
        counts[_centroid_quadrant(cx, cy)] += 1
        cx_sum += cx

    mean_cx = cx_sum / n
    best_q, best_n = max(counts.items(), key=lambda kv: kv[1])

    corner_affinity: str | None = None
    if (
        best_n / n >= CORNER_AFFINITY_MIN_FRACTION
        and mean_area < CORNER_AFFINITY_MAX_AREA
    ):
        corner_affinity = best_q

    return {
        "persistence": persistence,
        "mean_area": mean_area,
        "corner_affinity": corner_affinity,
        "mean_cx": mean_cx,
    }


def summarize(
    tracks: list[FaceTrack],
    frame_count: int,
    frame_w: int = 0,  # noqa: ARG001
    frame_h: int = 0,  # noqa: ARG001
) -> TrackSummary:
    """Build a TrackSummary from a set of face tracks.

    Args:
        tracks: FaceTrack list (order-independent; track_id identifies each).
        frame_count: total frames in the vision segment.
        frame_w / frame_h: passed through to track_stats (unused).

    Returns:
        TrackSummary:
          dominant_track_id   — highest persistence, tie-break larger mean_area
          persistent_track_ids — persistence >= PERSISTENT_THRESHOLD
          facecam_candidate_track_ids — corner_affinity set, persistence >=
                                FACECAM_PERSISTENCE, mean_area < FACECAM_MAX_AREA
          side_by_side_pair   — two persistent tracks, horizontally separated
                                (mean_cx on opposite sides of 0.5), mean_area
                                within SIDE_BY_SIDE_MAX_RATIO; None if no pair
    """
    if not tracks or frame_count <= 0:
        return TrackSummary(
            dominant_track_id=None,
            persistent_track_ids=[],
            facecam_candidate_track_ids=[],
            side_by_side_pair=None,
            dominant_persistence=0.0,
        )

    stats: list[tuple[int, dict]] = [
        (t.track_id, track_stats(t, frame_count, frame_w, frame_h))
        for t in tracks
    ]

    # Dominant: highest persistence; tie-break: larger mean_area; then smallest tid
    dominant_tid = max(
        stats, key=lambda s: (s[1]["persistence"], s[1]["mean_area"], -s[0])
    )[0]

    # Persistent tracks (sorted by track_id for determinism)
    persistent = sorted(
        tid for tid, s in stats if s["persistence"] >= PERSISTENT_THRESHOLD
    )

    # Facecam candidates (sorted by track_id for determinism)
    facecam = sorted(
        tid
        for tid, s in stats
        if s["corner_affinity"] is not None
        and s["persistence"] >= FACECAM_PERSISTENCE
        and s["mean_area"] < FACECAM_MAX_AREA
    )

    # Side-by-side pair:
    #   both persistent,
    #   horizontally separated (one cx < 0.5, other cx >= 0.5),
    #   mean_area within SIDE_BY_SIDE_MAX_RATIO of each other.
    # Pick the pair with highest combined persistence; tie-break smallest tid sum.
    side_by_side: tuple[int, int] | None = None
    if len(persistent) >= 2:
        by_tid = dict(stats)
        pids = persistent
        best_key: tuple[float, float, int] = (-1.0, -1.0, -1)
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                tid_a, tid_b = pids[i], pids[j]
                sa, sb = by_tid[tid_a], by_tid[tid_b]
                if sa["mean_area"] <= 0 or sb["mean_area"] <= 0:
                    continue
                # Horizontally separated?
                sep = (
                    (sa["mean_cx"] < 0.5 and sb["mean_cx"] >= 0.5)
                    or (sa["mean_cx"] >= 0.5 and sb["mean_cx"] < 0.5)
                )
                if not sep:
                    continue
                # Area within ratio?
                lo = min(sa["mean_area"], sb["mean_area"])
                hi = max(sa["mean_area"], sb["mean_area"])
                if hi / lo > SIDE_BY_SIDE_MAX_RATIO:
                    continue
                combined_p = sa["persistence"] + sb["persistence"]
                key = (
                    combined_p,
                    min(sa["mean_area"], sb["mean_area"]),
                    -(tid_a + tid_b),
                )
                if key > best_key:
                    best_key = key
                    side_by_side = (tid_a, tid_b)

    # M6f: record the dominant track's persistence for decide_layout
    dominant_persistence = 0.0
    for tid, s in stats:
        if tid == dominant_tid:
            dominant_persistence = s["persistence"]
            break

    return TrackSummary(
        dominant_track_id=dominant_tid,
        persistent_track_ids=persistent,
        facecam_candidate_track_ids=facecam,
        side_by_side_pair=side_by_side,
        dominant_persistence=dominant_persistence,
    )
