"""M6c — Face tracking and track statistics (docs/spec/m6.md).

CPU-only tests with synthetic FaceDetection objects; no GPU, no I/O,
no detector, no model files.
"""
from __future__ import annotations

import pytest

from clipper import tracking
from clipper.models import FaceDetection, FaceTrack, TrackSummary

# --- test helpers -----------------------------------------------------------


def det(
    t: float, x: float, y: float, w: float, h: float, score: float = 0.95
) -> FaceDetection:
    return FaceDetection(t=t, x=x, y=y, w=w, h=h, score=score)


def face_moving_smooth(
    frame_count: int = 30, x0: float = 0.4, vx: float = 0.001
) -> list[list[FaceDetection]]:
    """One face moving smoothly across frames (e.g. x=0.4 -> 0.68 over 28 frames)."""
    frames = []
    w, h = 0.15, 0.18
    for fi in range(frame_count):
        x = x0 + vx * fi
        t = fi / 30.0
        frames.append([det(t, x, 0.3, w, h)])
    return frames


def two_faces(frame_count: int = 30) -> list[list[FaceDetection]]:
    """Two faces side by side, present in all frames."""
    frames = []
    for fi in range(frame_count):
        t = fi / 30.0
        frames.append([
            det(t, 0.15, 0.3, 0.18, 0.20),   # left face
            det(t, 0.65, 0.3, 0.18, 0.20),   # right face
        ])
    return frames


def small_corner_face(
    frame_count: int = 30, quadrant: str = "tl"
) -> list[list[FaceDetection]]:
    """A small face pinned in a corner quadrant, present in all frames."""
    # small: w*h < 0.04 — use w=0.10, h=0.10 → area=0.01
    w, h = 0.10, 0.10
    cx_map = {"tl": 0.2, "tr": 0.8, "bl": 0.2, "br": 0.8}
    cy_map = {"tl": 0.2, "tr": 0.2, "bl": 0.8, "br": 0.8}
    cx = cx_map[quadrant]
    cy = cy_map[quadrant]
    x = cx - w / 2
    y = cy - h / 2
    frames = []
    for fi in range(frame_count):
        t = fi / 30.0
        frames.append([det(t, x, y, w, h)])
    return frames


def large_centered_face(frame_count: int = 30) -> list[list[FaceDetection]]:
    """A large face in the center, present in all frames."""
    # large: w*h > 0.04 — use w=0.4, h=0.5 → area=0.20
    w, h = 0.4, 0.5
    x = 0.5 - w / 2  # 0.3
    y = 0.5 - h / 2  # 0.25
    frames = []
    for fi in range(frame_count):
        t = fi / 30.0
        frames.append([det(t, x, y, w, h)])
    return frames


def occluded_frames(gap: int = 2) -> list[list[FaceDetection]]:
    """Face in frames 0-9, absent frames 10..10+gap-1, back at 10+gap..29."""
    x, y, w, h = 0.4, 0.3, 0.15, 0.18
    frames: list[list[FaceDetection]] = []
    for fi in range(30):
        if 10 <= fi < 10 + gap:
            frames.append([])
        else:
            frames.append([det(fi / 30.0, x, y, w, h)])
    return frames


# --- associate: single face -------------------------------------------------


def test_single_face_smooth_one_track_persistence_1():
    frames = face_moving_smooth(frame_count=30)
    tracks = tracking.associate(frames)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.track_id == 0
    assert len(track.boxes) == 30


def test_single_face_persistence_is_one():
    frames = face_moving_smooth(frame_count=20)
    tracks = tracking.associate(frames)
    assert len(tracks) == 1
    s = tracking.track_stats(tracks[0], frame_count=20)
    assert s["persistence"] == pytest.approx(1.0)


# --- associate: two faces side by side --------------------------------------


def test_two_faces_side_by_side_two_tracks():
    frames = two_faces(frame_count=30)
    tracks = tracking.associate(frames)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert ids == {0, 1}


def test_two_faces_stable_id_ordering():
    """Left face appears first → track_id=0; right face → track_id=1."""
    frames = two_faces(frame_count=30)
    tracks = tracking.associate(frames)
    left_track = next(t for t in tracks if t.track_id == 0)
    right_track = next(t for t in tracks if t.track_id == 1)
    # Left face has x ≈ 0.15, right face x ≈ 0.65
    assert left_track.boxes[0][1] < 0.5
    assert right_track.boxes[0][1] >= 0.5


# --- summarize: side_by_side -------------------------------------------------


def test_summarize_side_by_side_detected():
    frames = two_faces(frame_count=30)
    tracks = tracking.associate(frames)
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert summary.side_by_side_pair is not None
    assert set(summary.side_by_side_pair) == {0, 1}


def test_summarize_persistent_tracks():
    frames = two_faces(frame_count=30)
    tracks = tracking.associate(frames)
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert sorted(summary.persistent_track_ids) == [0, 1]


# --- associate: short occlusion within max_gap_frames ----------------------


def test_short_occlusion_within_gap_same_track():
    # 2-frame gap, max_gap=3 → one track
    frames = occluded_frames(gap=2)
    tracks = tracking.associate(frames, max_gap_frames=3)
    assert len(tracks) == 1
    assert len(tracks[0].boxes) == 28  # 30 - 2 missing


def test_short_occlusion_within_gap_persistence():
    # 2-frame gap in 30 frames → 28 frames present
    frames = occluded_frames(gap=2)
    tracks = tracking.associate(frames, max_gap_frames=3)
    s = tracking.track_stats(tracks[0], frame_count=30)
    assert s["persistence"] == pytest.approx(28 / 30)


# --- associate: long gap → new track ----------------------------------------


def test_long_gap_creates_new_track():
    # 5-frame gap, max_gap=3 → track ends, new track starts
    frames = occluded_frames(gap=5)
    # max_gap=3 means: after 3 consecutive misses, finalize track.
    # 5 > 3, so track 0 is finalized before the face reappears at frame 15.
    tracks = tracking.associate(frames, max_gap_frames=3)
    assert len(tracks) == 2


def test_long_gap_determine_ids():
    frames = occluded_frames(gap=5)
    tracks = tracking.associate(frames, max_gap_frames=3)
    assert {t.track_id for t in tracks} == {0, 1}


# --- corner affinity: small corner face -------------------------------------


def test_small_tl_face_corner_affinity_tl():
    frames = small_corner_face(frame_count=30, quadrant="tl")
    tracks = tracking.associate(frames)
    assert len(tracks) == 1
    s = tracking.track_stats(tracks[0], frame_count=30)
    assert s["corner_affinity"] == "tl"
    assert s["mean_area"] < 0.04


def test_small_tr_face_corner_affinity_tr():
    frames = small_corner_face(frame_count=30, quadrant="tr")
    tracks = tracking.associate(frames)
    s = tracking.track_stats(tracks[0], frame_count=30)
    assert s["corner_affinity"] == "tr"


def test_small_bl_face_corner_affinity_bl():
    frames = small_corner_face(frame_count=30, quadrant="bl")
    tracks = tracking.associate(frames)
    s = tracking.track_stats(tracks[0], frame_count=30)
    assert s["corner_affinity"] == "bl"


def test_small_br_face_corner_affinity_br():
    frames = small_corner_face(frame_count=30, quadrant="br")
    tracks = tracking.associate(frames)
    s = tracking.track_stats(tracks[0], frame_count=30)
    assert s["corner_affinity"] == "br"


# --- summarize: facecam candidate -------------------------------------------


def test_summarize_facecam_candidate_small_corner():
    frames = small_corner_face(frame_count=30, quadrant="tl")
    tracks = tracking.associate(frames)
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert 0 in summary.facecam_candidate_track_ids


def test_summarize_facecam_not_candidate_large_face():
    frames = large_centered_face(frame_count=30)
    tracks = tracking.associate(frames)
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert summary.facecam_candidate_track_ids == []


# --- large centered face: no corner affinity --------------------------------


def test_large_centered_face_corner_affinity_none():
    frames = large_centered_face(frame_count=30)
    tracks = tracking.associate(frames)
    assert len(tracks) == 1
    s = tracking.track_stats(tracks[0], frame_count=30)
    assert s["corner_affinity"] is None
    assert s["mean_area"] > 0.04  # large face


def test_large_centered_face_not_facecam_candidate():
    frames = large_centered_face(frame_count=30)
    tracks = tracking.associate(frames)
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert summary.facecam_candidate_track_ids == []


# --- empty input -------------------------------------------------------------


def test_no_detections_empty_tracks():
    tracks = tracking.associate([])
    assert tracks == []


def test_no_detections_empty_summary():
    summary = tracking.summarize([], frame_count=30, frame_w=960, frame_h=540)
    assert summary.dominant_track_id is None
    assert summary.persistent_track_ids == []
    assert summary.facecam_candidate_track_ids == []
    assert summary.side_by_side_pair is None


def test_all_empty_frames_no_tracks():
    """All frames empty → zero tracks, no crash."""
    frames = [[] for _ in range(30)]
    tracks = tracking.associate(frames)
    assert tracks == []
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert summary.dominant_track_id is None


# --- determinism and stable track_id ordering --------------------------------


def test_determinism_same_input_same_output():
    """Two calls with identical input produce identical output."""
    frames = two_faces(frame_count=30)

    tracks1 = tracking.associate(frames)
    tracks2 = tracking.associate(frames)

    assert len(tracks1) == len(tracks2)
    for t1, t2 in zip(tracks1, tracks2, strict=True):
        assert t1.track_id == t2.track_id
        assert t1.boxes == t2.boxes


def test_track_ids_assigned_by_first_appearance():
    """Face A appears in frame 0, face B appears in frame 5: A=0, B=1."""
    frames: list[list[FaceDetection]] = []
    # Frames 0-4: only face A at (x=0.2, y=0.3)
    for fi in range(5):
        frames.append([det(fi / 30.0, 0.2, 0.3, 0.15, 0.18)])
    # Frames 5-29: both faces
    for fi in range(5, 30):
        frames.append([
            det(fi / 30.0, 0.2, 0.3, 0.15, 0.18),   # face A
            det(fi / 30.0, 0.7, 0.3, 0.15, 0.18),   # face B (appears at frame 5)
        ])
    tracks = tracking.associate(frames)
    assert len(tracks) == 2
    # Face A appeared first → track_id=0
    assert any(t.track_id == 0 for t in tracks)
    # Face B appeared later → track_id=1
    assert any(t.track_id == 1 for t in tracks)
    # Confirm: track 0 has 30 boxes, track 1 has 25 boxes
    t0 = next(t for t in tracks if t.track_id == 0)
    t1 = next(t for t in tracks if t.track_id == 1)
    assert len(t0.boxes) == 30
    assert len(t1.boxes) == 25


def test_determinism_identical_stats_on_repeated_calls():
    """summarize is deterministic across repeated calls."""
    frames = face_moving_smooth(frame_count=30)
    tracks = tracking.associate(frames)
    s1 = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    s2 = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    assert s1 == s2


# --- track_stats basic properties --------------------------------------------


def test_track_stats_returns_all_keys():
    frames = face_moving_smooth(frame_count=10)
    tracks = tracking.associate(frames)
    s = tracking.track_stats(tracks[0], frame_count=10)
    assert set(s.keys()) == {"persistence", "mean_area", "corner_affinity", "mean_cx"}


def test_track_stats_zero_frame_count():
    track = FaceTrack(
        track_id=0,
        boxes=[(0.0, 0.4, 0.3, 0.15, 0.18)],
        persistence=0.0,
        corner_affinity=None,
    )
    s = tracking.track_stats(track, frame_count=0)
    assert s["persistence"] == 0.0
    assert s["mean_area"] == 0.0
    assert s["corner_affinity"] is None


def test_track_stats_empty_track():
    track = FaceTrack(track_id=0, boxes=[], persistence=0.0, corner_affinity=None)
    s = tracking.track_stats(track, frame_count=30)
    assert s["persistence"] == 0.0
    assert s["mean_area"] == 0.0
    assert s["corner_affinity"] is None


# --- summarize: dominant track -----------------------------------------------


def test_summarize_dominant_is_more_persistent():
    """Track with 30/30 frames beats track with 15/30 frames."""
    frames: list[list[FaceDetection]] = []
    for fi in range(30):
        dets = [det(fi / 30.0, 0.2, 0.3, 0.15, 0.18)]        # face A: all frames
        if fi >= 15:
            dets.append(det(fi / 30.0, 0.7, 0.3, 0.15, 0.18)) # face B: frames 15-29
        frames.append(dets)
    tracks = tracking.associate(frames)
    summary = tracking.summarize(tracks, frame_count=30, frame_w=960, frame_h=540)
    # face A (tid=0) present 30/30 frames; face B (tid=1) present 15/30 frames
    assert summary.dominant_track_id == 0


# --- corner affinity: >= 80% rule --------------------------------------------


def test_corner_affinity_at_80_percent_boundary():
    """Exactly 80% in one quadrant → corner_affinity set (>= comparison).

    Uses a hand-crafted FaceTrack (24 boxes in 'tl', 6 boxes in 'tr') to
    avoid IoU-matching issues at the quadrant boundary. All boxes share the
    same y, w, h; x is chosen so that IoU is maintained but we build the
    track directly (no associate needed)."""
    w, h = 0.10, 0.10
    boxes: list[tuple[float, float, float, float, float]] = []
    # 24 boxes in top-left quadrant (cx≈0.2 < 0.5)
    for i in range(24):
        boxes.append((i / 30.0, 0.2 - w / 2, 0.2 - h / 2, w, h))
    # 6 boxes in top-right quadrant (cx≈0.8 >= 0.5)
    for i in range(24, 30):
        boxes.append((i / 30.0, 0.8 - w / 2, 0.2 - h / 2, w, h))
    track = FaceTrack(track_id=0, boxes=boxes, persistence=0.0, corner_affinity=None)
    s = tracking.track_stats(track, frame_count=30)
    assert s["mean_area"] == pytest.approx(0.01)
    assert s["corner_affinity"] == "tl"  # 24/30 = 0.8 >= 0.8, area=0.01 < 0.04


def test_corner_affinity_below_80_none():
    """75% in one quadrant → corner_affinity None.

    Hand-crafted track: 22 boxes in 'tl', 8 boxes in 'tr'.
    22/30 = 0.733 < 0.8 → no corner affinity."""
    w, h = 0.10, 0.10
    boxes: list[tuple[float, float, float, float, float]] = []
    for i in range(22):
        boxes.append((i / 30.0, 0.2 - w / 2, 0.2 - h / 2, w, h))
    for i in range(22, 30):
        boxes.append((i / 30.0, 0.8 - w / 2, 0.2 - h / 2, w, h))
    track = FaceTrack(track_id=0, boxes=boxes, persistence=0.0, corner_affinity=None)
    s = tracking.track_stats(track, frame_count=30)
    assert s["corner_affinity"] is None  # 22/30 = 0.733 < 0.8
    assert s["mean_area"] == pytest.approx(0.01)


# --- IoU threshold controls matching -----------------------------------------


def test_iou_threshold_high_prevents_match():
    """With a very high IoU threshold, a slightly shifted face won't match."""
    # Face at x=0.40 for 5 frames, then shifts to x=0.70
    # With iou_threshold=0.99, the shift is too large → new track
    frames: list[list[FaceDetection]] = []
    w, h = 0.15, 0.18
    for fi in range(5):
        frames.append([det(fi / 30.0, 0.40, 0.3, w, h)])
    for fi in range(5, 10):
        frames.append([det(fi / 30.0, 0.70, 0.3, w, h)])
    tracks = tracking.associate(frames, iou_threshold=0.99, max_gap_frames=3)
    assert len(tracks) == 2


def test_iou_threshold_low_allows_match():
    """With a low IoU threshold, a slightly shifted face still matches."""
    # Small shift: x=0.40 → x=0.45 (IoU should still be high)
    frames: list[list[FaceDetection]] = []
    w, h = 0.15, 0.18
    for fi in range(5):
        frames.append([det(fi / 30.0, 0.40, 0.3, w, h)])
    for fi in range(5, 10):
        frames.append([det(fi / 30.0, 0.45, 0.3, w, h)])
    tracks = tracking.associate(frames, iou_threshold=0.3, max_gap_frames=3)
    assert len(tracks) == 1  # still one track


# --- summarize: zero tracks no crash ----------------------------------------


def test_summarize_zero_tracks_no_crash():
    summary = tracking.summarize([], frame_count=100, frame_w=960, frame_h=540)
    assert isinstance(summary, TrackSummary)
    assert summary.dominant_track_id is None


# --- TrackSummary model round-trip --------------------------------------------


def test_track_summary_model_roundtrip():
    summary = TrackSummary(
        dominant_track_id=0,
        persistent_track_ids=[0, 1],
        facecam_candidate_track_ids=[0],
        side_by_side_pair=(0, 1),
    )
    data = summary.model_dump_json()
    summary2 = TrackSummary.model_validate_json(data)
    assert summary == summary2
