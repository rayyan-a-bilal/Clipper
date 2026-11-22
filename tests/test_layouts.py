"""CPU-only golden tests for M6d layout decision + crop path (pure).

No GPU, no I/O, no detector, no model files — synthetic FaceTrack boxes
only (M6a/M6c contracts).
"""
from clipper.layouts import (
    ASPECT,
    DEAD_ZONE,
    EMA_TAU_S,
    FACECAM_ENLARGE,
    MAX_PAN_FRAC_PER_S,
    build_tracks,
    decide_layout,
)
from clipper.models import FaceTrack, Layout, TrackSummary

SW, SH = 1920, 1080  # 16:9 source
FPS = 30.0
DUR = 2.0
N_KEYS = round(DUR * FPS) + 1  # 61


def summary(
    dominant: int | None = None,
    persistent: list[int] | None = None,
    facecam: list[int] | None = None,
    pair: tuple[int, int] | None = None,
) -> TrackSummary:
    return TrackSummary(
        dominant_track_id=dominant,
        persistent_track_ids=persistent or [],
        facecam_candidate_track_ids=facecam or [],
        side_by_side_pair=pair,
    )


def box(t: float, cx_n: float, cy_n: float = 0.5, size: float = 0.2) -> tuple:
    """FaceTrack box (t, x, y, w, h) centered at normalized (cx_n, cy_n)."""
    return (t, cx_n - size / 2.0, cy_n - size / 2.0, size, size)


def track(tid: int, boxes: list[tuple]) -> FaceTrack:
    return FaceTrack(
        track_id=tid, boxes=boxes, persistence=1.0, corner_affinity=None
    )


def moving_face(cxs, ts=None, size: float = 0.2) -> FaceTrack:
    ts = ts or [i * 0.2 for i in range(len(cxs))]
    return track(0, [box(t, cx) for t, cx in zip(ts, cxs, strict=True)])


def track_window_wh():
    w = min(int(round(SH * ASPECT)), SW)
    return w, SH  # w=608, h=1080


# --------------------------------------------------------------------- decide


def test_no_faces_is_normal_and_gives_center_crop():
    assert (
        decide_layout(summary(), SW, SH) is Layout.center_crop
    )  # NO FACE IS NORMAL


def test_facecam_candidate_wins():
    assert (
        decide_layout(summary(dominant=0, persistent=[0], facecam=[0]), SW, SH)
        is Layout.facecam
    )


def test_facecam_precedence_over_split_screen():
    s = summary(dominant=0, persistent=[0, 1], facecam=[1], pair=(0, 1))
    assert decide_layout(s, SW, SH) is Layout.facecam


def test_split_screen_when_pair_present():
    assert (
        decide_layout(summary(dominant=0, persistent=[0, 1], pair=(0, 1)), SW, SH)
        is Layout.split_screen
    )


def test_face_track_when_one_persistent_track():
    assert (
        decide_layout(summary(dominant=0, persistent=[0]), SW, SH)
        is Layout.face_track
    )


def test_dominant_but_not_persistent_is_center_crop():
    # dominant exists but its persistence is below the 0.6 threshold
    assert (
        decide_layout(summary(dominant=0, persistent=[]), SW, SH) is Layout.center_crop
    )


# ------------------------------------------------------------- common invariants


def keys_of(tracks):
    return [k.model_dump() for t_ in tracks for k in t_.keys]


def assert_common_invariants(tracks):
    for t_ in tracks:
        assert len(t_.keys) == N_KEYS, "key count == round(duration*fps) + 1"
        assert t_.keys[0].t == 0.0, "t is clip-relative, starts at 0.0"
        assert [isinstance(k.x, int) for k in t_.keys] == [True] * N_KEYS
        ws = {k.w for k in t_.keys}
        hs = {k.h for k in t_.keys}
        assert len(ws) == 1 and len(hs) == 1, "w/h consistent per track"
        w, h = ws.pop(), hs.pop()
        for k in t_.keys:
            assert 0 <= k.x and k.x + w <= SW, "x within source bounds"
            assert 0 <= k.y and k.y + h <= SH, "y within source bounds"


def assert_deterministic(build):
    a = [r.model_dump_json() for r in build()]
    b = [r.model_dump_json() for r in build()]
    assert a == b, "pure logic must be deterministic"


# -------------------------------------------------------------- center_crop


def test_center_crop_static_identical_keys():
    s = summary()
    tracks = build_tracks(Layout.center_crop, s, SW, SH, FPS, DUR, [], [])
    assert [t_.role for t_ in tracks] == ["main"]
    keys = keys_of(tracks)
    uniq = {(k["x"], k["y"], k["w"], k["h"]) for k in keys}
    assert len(uniq) == 1, "static path: all keys identical"
    assert_common_invariants(tracks)


def test_center_crop_exact_9_16():
    tracks = build_tracks(Layout.center_crop, summary(), SW, SH, FPS, DUR, [], [])
    k = tracks[0].keys[0]
    assert k.w * 16 == k.h * 9, "EXACT 9:16"
    assert k.x == (SW - k.w) // 2 and k.y == (SH - k.h) // 2, "centered"
    assert_deterministic(
        lambda: build_tracks(Layout.center_crop, summary(), SW, SH, FPS, DUR, [], [])
    )


def test_center_crop_portrait_source_fills_frame():
    W, H = 1080, 1920
    tracks = build_tracks(
        Layout.center_crop, summary(), W, H, FPS, DUR, [], []
    )
    k = tracks[0].keys[0]
    assert (k.w, k.h, k.x, k.y) == (W, H, 0, 0)


# --------------------------------------------------------------- face_track


def test_face_track_static_when_face_still():
    s = summary(dominant=0, persistent=[0])
    tr = track(0, [box(i * 0.1, 0.5) for i in range(10)])
    tracks = build_tracks(Layout.face_track, s, SW, SH, FPS, DUR, [], [tr])
    assert [t_.role for t_ in tracks] == ["main"]
    xs = {k["x"] for k in keys_of(tracks)}
    assert len(xs) == 1, "still face -> static path"
    assert_common_invariants(tracks)


def test_face_track_window_geometry():
    tracks = build_tracks(
        Layout.face_track,
        summary(dominant=0, persistent=[0]),
        SW, SH, FPS, DUR, [],
        [moving_face([0.3, 0.5, 0.7])],
    )
    k = tracks[0].keys[0]
    w, h = track_window_wh()
    assert k.h == SH, "window h = source_h"
    assert k.w == w == int(round(SH * ASPECT)), "window w = round(h * 9/16)"
    assert k.y == 0


def test_face_track_moving_path_moves_and_stays_in_bounds():
    s = summary(dominant=0, persistent=[0])
    tracks = build_tracks(
        Layout.face_track, s, SW, SH, FPS, DUR, [],
        [moving_face([0.3, 0.4, 0.5, 0.6, 0.7], ts=[0.0, 0.2, 0.4, 0.6, 0.8])],
    )
    xs = [k["x"] for k in keys_of(tracks)]
    assert max(xs) - min(xs) > 0, "path moves"
    hi = SW - int(round(SH * ASPECT))
    assert min(xs) >= 0 and max(xs) <= hi, "stays in [0, source_w - w]"
    assert_common_invariants(tracks)
    assert_deterministic(
        lambda: build_tracks(
            Layout.face_track, s, SW, SH, FPS, DUR, [],
            [moving_face([0.3, 0.4, 0.5, 0.6, 0.7], ts=[0.0, 0.2, 0.4, 0.6, 0.8])],
        )
    )


def test_ema_smoothing_gradual_approach():
    # face jumps hard LEFT at t=0, then target stays far RIGHT at t=0.5
    dt = 1 / 30.0
    tr = track(
        0,
        [box(0.0, 0.16)] + [box(i * dt, 0.84) for i in range(1, 46)],
    )
    tracks = build_tracks(
        Layout.face_track,
        summary(dominant=0, persistent=[0]),
        SW, SH, FPS, DUR, [], [tr],
    )
    xs = [k["x"] for k in keys_of(tracks)]
    w = int(round(SH * ASPECT))
    target = SW - w - 1  # far from start
    assert xs[0] < target, "no jump to target on first frame after step input"
    assert xs[-1] < target, "approach is gradual, not a jump"
    deltas = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    assert all(d >= -1 for d in deltas), "monotonic approach"
    max_step = MAX_PAN_FRAC_PER_S * w * 1.0
    assert max(deltas) <= max_step + 1.0
    # every per-frame increment respects EMA alpha of tau and the velocity clamp
    import math

    alpha = 1.0 - math.exp(-dt / EMA_TAU_S)
    first_increment = xs[1] - xs[0]
    expected = min(alpha * (target + 1 - xs[0]), MAX_PAN_FRAC_PER_S * w * dt)
    assert abs(first_increment - expected) <= 1.5


def test_dead_zone_small_jitter_zero_motion():
    # centroid sits on the window center; jitter <= 20% of window -> static
    w = int(round(SH * ASPECT))
    jit = 0.02 * w  # 12 px, well inside the 0.2*w/2 = 60.8 px dead half-width
    assert jit < DEAD_ZONE * w / 2
    ts = [i * 0.1 for i in range(20)]
    offsets = [(jit if i % 2 else -jit) / SW for i in range(20)]  # ±12px in SOURCE px
    cxs = [0.5 + o for o in offsets]
    tr = track(
        0, [box(0.0, 0.5)] + [box(t, cx) for t, cx in zip(ts[1:], cxs[1:], strict=True)]
    )
    tracks = build_tracks(
        Layout.face_track,
        summary(dominant=0, persistent=[0]),
        SW, SH, FPS, DUR, [], [tr],
    )
    xs = {k["x"] for k in keys_of(tracks)}
    assert len(xs) == 1, "dead zone: small centroid jitter produces ZERO motion"


def test_velocity_clamp_rate_limits_large_jump():
    w = int(round(SH * ASPECT))
    tr = track(
        0,
        [box(0.0, w / 2.0 / SW), box(1.0, (SW - w / 2.0) / SW)],
    )  # delta target = SW - w, one second apart
    tracks = build_tracks(
        Layout.face_track,
        summary(dominant=0, persistent=[0]),
        SW, SH, FPS, DUR, [], [tr],
    )
    k0 = {round(k.t, 6): k for k in tracks[0].keys}
    x0, x1 = k0[0.0].x, k0[1.0].x
    assert x1 > x0, "it moves toward the new centroid..."
    assert x1 - x0 <= MAX_PAN_FRAC_PER_S * w * 1.0 + 1.0, "...but is rate-limited"


def test_shot_boundary_hard_reset():
    # face left, then a shot cut, then the SAME face far right:
    # position must jump at the boundary (no EMA blend)
    s = summary(dominant=0, persistent=[0])
    tr = track(0, [box(0.5, 0.2), box(1.5, 0.8)])
    tracks = build_tracks(Layout.face_track, s, SW, SH, FPS, DUR, [1.0], [tr])
    at = {round(k.t, 6): k for k in tracks[0].keys}
    x_left = at[0.9].x
    x_at_cut = at[1.0].x
    assert x_at_cut > x_left, "path advanced after the cut"
    # hard reset lands exactly on the new target (centered on cx=0.8)
    w = int(round(SH * ASPECT))
    assert x_at_cut == min(round(0.8 * SW - w / 2), SW - w)
    assert at[1.5].x == x_at_cut and at[2.0].x == x_at_cut, "holds the reset position"


def test_face_track_requires_dominant_track():
    import pytest

    with pytest.raises(ValueError):
        build_tracks(
            Layout.face_track, summary(dominant=None, persistent=[0]),
            SW, SH, FPS, DUR, [], [track(0, [box(0, 0.5)])],
        )


def test_face_track_missing_track_raises():
    import pytest

    with pytest.raises(ValueError):
        build_tracks(
            Layout.face_track, summary(dominant=0, persistent=[0]),
            SW, SH, FPS, DUR, [], [],
        )


# ------------------------------------------------------------- split_screen


def make_pair(cxs):
    return [
        track(0, [box(0.0, cxs[0]), box(0.5, cxs[0])]),
        track(1, [box(0.0, cxs[1]), box(0.5, cxs[1])]),
    ]


def test_split_screen_two_static_tracks_top_bottom():
    s = summary(dominant=0, persistent=[0, 1], pair=(0, 1))
    tracks = build_tracks(
        Layout.split_screen, s, SW, SH, FPS, DUR, [], make_pair((0.25, 0.75))
    )
    assert [t_.role for t_ in tracks] == ["top", "bottom"]
    w = int(round(SH * ASPECT))
    kt = tracks[0].keys
    kb = tracks[1].keys
    assert len({(k.x, k.y, k.w, k.h) for k in kt}) == 1, "top is static"
    assert len({(k.x, k.y, k.w, k.h) for k in kb}) == 1, "bottom is static"
    assert kt[0].w == kb[0].w == w and kt[0].h == kb[0].h == SH
    assert kt[0].x == min(round(0.25 * SW - w / 2), SW - w), "centered on speaker A"
    assert kb[0].x == min(round(0.75 * SW - w / 2), SW - w), "centered on speaker B"
    assert kt[0].x != kb[0].x
    assert_common_invariants(tracks)
    def build():
        return build_tracks(
            Layout.split_screen, s, SW, SH, FPS, DUR, [], make_pair((0.25, 0.75))
        )

    assert_deterministic(build)


# ------------------------------------------------------------------- facecam


def test_facecam_small_corner_face_main_plus_face():
    s = summary(dominant=0, persistent=[0], facecam=[0])
    tr = track(0, [box(0.0, 0.1, 0.1, 0.1), box(0.5, 0.1, 0.1, 0.1)])  # small TL face
    tracks = build_tracks(Layout.facecam, s, SW, SH, FPS, DUR, [], [tr])
    assert [t_.role for t_ in tracks] == ["main", "face"]
    main, face = tracks[0], tracks[1]
    # main: static exact-9:16 content window
    mk = main.keys[0]
    assert mk.w * 16 == mk.h * 9
    assert len({(k.x, k.y, k.w, k.h) for k in main.keys}) == 1, "main is static"
    # face: static square, enlarged around the corner face, in bounds
    fk = face.keys[0]
    assert fk.w == fk.h, "facecam is a square"
    side = int(round(0.1 * SW * FACECAM_ENLARGE))
    assert fk.w == min(side, min(SW, SH))
    assert len({(k.x, k.y, k.w, k.h) for k in face.keys}) == 1, "face is static"
    assert fk.x + fk.w > 0.1 * SW, "covers the face's side"
    assert_common_invariants(tracks)
    assert_deterministic(
        lambda: build_tracks(
            Layout.facecam, s, SW, SH, FPS, DUR, [],
            [track(0, [box(0.0, 0.1, 0.1, 0.1), box(0.5, 0.1, 0.1, 0.1)])],
        )
    )


# --------------------------------------------------------------- key grid + guards


def test_key_grid_all_layouts():
    s_face = summary(dominant=0, persistent=[0])
    s_pair = summary(dominant=0, persistent=[0, 1], pair=(0, 1))
    s_facecam = summary(dominant=0, persistent=[0], facecam=[0])
    for layout, s, tracks_in in (
        (Layout.center_crop, summary(), []),
        (Layout.face_track, s_face, [moving_face([0.3, 0.5, 0.7])]),
        (Layout.split_screen, s_pair, make_pair((0.25, 0.75))),
        (Layout.facecam, s_facecam, [track(0, [box(0.0, 0.1, 0.1, 0.1)])]),
    ):
        tracks = build_tracks(layout, s, SW, SH, FPS, DUR, [], tracks_in)
        assert_common_invariants(tracks)


def test_key_count_with_odd_fps_and_duration():
    tracks = build_tracks(Layout.center_crop, summary(), SW, SH, 24.0, 1.7, [], [])
    assert len(tracks[0].keys) == round(1.7 * 24.0) + 1
    assert tracks[0].keys[0].t == 0.0


def test_bad_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        build_tracks(Layout.center_crop, summary(), SW, SH, 0.0, DUR, [], [])
    with pytest.raises(ValueError):
        build_tracks(Layout.center_crop, summary(), 0, SH, FPS, DUR, [], [])
