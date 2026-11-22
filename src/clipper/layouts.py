"""Pure layout decision and crop-track generation (M6d — docs/spec/m6.md).

No I/O, no adapters, no GPU, no numpy. Unit-testable with synthetic
`FaceTrack` objects and a `TrackSummary` only.

Layout decision (highest precedence first):
  1. facecam candidate present        -> facecam
  2. side_by_side pair                -> split_screen
  3. a persistent track (>= 0.6)      -> face_track
  4. otherwise                        -> center_crop   (NO FACE IS NORMAL)

`build_tracks` emits `RegionTrack`s whose keys are the M0 `CropKey`:
  - CLIP-RELATIVE timestamps starting at 0.0
  - round(duration_s * native_fps) + 1 keys per track (NATIVE fps grid)
  - coordinates are SOURCE pixels, integers
  - motion segments are LINEAR INTERPOLATION between computed positions
"""
from __future__ import annotations

import math

from clipper.models import CropKey, FaceTrack, Layout, RegionTrack, TrackSummary

# --- Module constants (M6d) ---
EMA_TAU_S = 0.8               # EMA time constant for face_track panning
DEAD_ZONE = 0.20              # no motion while centroid within central 20% of window
MAX_PAN_FRAC_PER_S = 0.30     # velocity clamp, fraction of window width per second
ASPECT = 9 / 16               # output aspect (exact in binary floating point)
FACECAM_ENLARGE = 4.0         # facecam square side = max(mean face w, h) * this


def decide_layout(
    summary: TrackSummary,
    source_w: int,  # noqa: ARG001 — geometry is irrelevant to the decision
    source_h: int,  # noqa: ARG001 — kept on the card's signature
) -> Layout:
    """Decide the layout from the per-clip track summary.

    Precedence: facecam > split_screen > face_track > center_crop.
    A faceless clip (empty summary) is a NORMAL state -> center_crop.
    """
    if summary.facecam_candidate_track_ids:
        return Layout.facecam
    if summary.side_by_side_pair is not None:
        return Layout.split_screen
    if summary.persistent_track_ids:
        return Layout.face_track
    return Layout.center_crop


# --- Geometry helpers (SOURCE pixels) ---

def _check_dims(
    source_w: int, source_h: int, native_fps: float, duration_s: float
) -> None:
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"source dims must be positive, got {source_w}x{source_h}")
    if native_fps <= 0:
        raise ValueError(f"native_fps must be positive, got {native_fps}")
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s}")


def _center_window(source_w: int, source_h: int) -> tuple[int, int, int, int]:
    """Largest EXACT 9:16 integer window fitting inside the source, centered.

    w = 9k, h = 16k, k = min(source_w // 9, source_h // 16) (9/16 reduced),
    so w/h == ASPECT exactly. Centering uses integer floors; the window
    always fits: 0 <= (sw - w) // 2 and 0 <= (sh - h) // 2 (w <= sw, h <= sh).
    """
    k = min(source_w // 9, source_h // 16)
    if k < 1:
        raise ValueError(f"source {source_w}x{source_h} too small for one 9x16 cell")
    w, h = 9 * k, 16 * k
    x = (source_w - w) // 2
    y = (source_h - h) // 2
    return x, y, w, h


def _track_window(source_w: int, source_h: int) -> tuple[int, int]:
    """Full-height 9:16 window: h = source_h, w = round(h * ASPECT).

    Safety clamp: w cannot exceed source_w (very portrait sources).
    """
    h = source_h
    w = min(int(round(source_h * ASPECT)), source_w)
    return w, h


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _key_times(native_fps: float, duration_s: float) -> list[float]:
    n = round(duration_s * native_fps)
    return [i / native_fps for i in range(n + 1)]


def _static_track(
    role: str,
    x: int,
    y: int,
    w: int,
    h: int,
    native_fps: float,
    duration_s: float,
) -> RegionTrack:
    keys = [
        CropKey(t=t, x=x, y=y, w=w, h=h) for t in _key_times(native_fps, duration_s)
    ]
    return RegionTrack(role=role, keys=keys)


def _find_track(tracks: list[FaceTrack], track_id: int, layout: Layout) -> FaceTrack:
    for track in tracks:
        if track.track_id == track_id:
            return track
    raise ValueError(
        f"layout {layout.value} requires track {track_id} but it is not in `tracks`"
    )


def _mean_centroid_and_size(
    track: FaceTrack, source_w: int, source_h: int
) -> tuple[float, float, float, float]:
    """Mean SOURCE-pixel centroid (cx, cy) and mean SOURCE-pixel box (bw, bh)."""
    boxes = track.boxes
    n = len(boxes)
    cx = sum((b[1] + b[3] / 2.0) * source_w for b in boxes) / n
    cy = sum((b[2] + b[4] / 2.0) * source_h for b in boxes) / n
    bw = sum(b[3] for b in boxes) / n * source_w
    bh = sum(b[4] for b in boxes) / n * source_h
    return cx, cy, bw, bh


def _sample(events: list[tuple[float, float]], i: int, t: float) -> float:
    """Linear interpolation of events[(t_i, v_i)] (t increasing) at time t.

    `i` is an index with events[i][0] <= t < events[i+1][0] (caller walks a
    two-pointer).
    """
    if t <= events[0][0]:
        return events[0][1]
    if t >= events[-1][0]:
        return events[-1][1]
    e0, e1 = events[i], events[i + 1]
    if e1[0] == e0[0]:
        return e1[1]
    frac = (t - e0[0]) / (e1[0] - e0[0])
    return e0[1] + frac * (e1[1] - e0[1])


def _build_face_track(
    track: FaceTrack,
    source_w: int,
    source_h: int,
    native_fps: float,
    duration_s: float,
    shot_boundaries: list[float],
) -> RegionTrack:
    """Single full-height 9:16 window following the dominant face centroid.

    x follows the per-detection target (window centered on the centroid,
    clamped to [0, source_w - w]) with:
      - EMA smoothing: alpha = 1 - exp(-dt / EMA_TAU_S)
      - dead zone: no motion while the centroid is within the central
        DEAD_ZONE fraction of the crop window
      - velocity clamp: |dx| <= MAX_PAN_FRAC_PER_S * w * dt
      - hard reset at shot boundaries (jump straight to the new target,
        no smoothing)
    Output keys: linear interpolation between computed positions on the
    native-fps grid.
    """
    w, h = _track_window(source_w, source_h)
    lo, hi = 0, source_w - w

    # Per-detection centroids in SOURCE pixels, in time order
    dets = sorted(
        ((b[0], (b[1] + b[3] / 2.0) * source_w) for b in track.boxes),
        key=lambda e: e[0],
    )
    bounds = sorted(b for b in shot_boundaries if b > 0 and b < duration_s)

    events: list[tuple[float, float]] = []
    state: float | None = None
    prev_t: float | None = None
    for t, cx in dets:
        target = _clamp(cx - w / 2.0, float(lo), float(hi))
        crossed = [
            b
            for b in bounds
            if (prev_t if prev_t is not None else float("-inf")) < b <= t
        ]
        if state is None:
            state = target  # initialize (hard) at the first detection
        elif crossed:
            # Hard reset AT each shot boundary: jump to the new target
            state = target
            for b in crossed:
                events.append((b, state))
        else:
            dt = t - (prev_t or 0.0)
            if dt > 0 and abs(cx - (state + w / 2.0)) > DEAD_ZONE * w / 2.0:
                alpha = 1.0 - math.exp(-dt / EMA_TAU_S)
                step = alpha * (target - state)
                max_step = MAX_PAN_FRAC_PER_S * w * dt
                if step > max_step:
                    step = max_step
                elif step < -max_step:
                    step = -max_step
                state = _clamp(state + step, float(lo), float(hi))
        events.append((t, state))
        prev_t = t

    if not events:
        # Defensive: a face_track always comes from a non-empty track;
        # fall back to a static centered window.
        events = [(0.0, float(_clamp((source_w - w) / 2.0, float(lo), float(hi))))]

    keys: list[CropKey] = []
    i = 0
    for t in _key_times(native_fps, duration_s):
        while i < len(events) - 1 and events[i + 1][0] < t:
            i += 1
        v = _sample(events, i, t)
        x = int(round(_clamp(v, float(lo), float(hi))))
        keys.append(CropKey(t=t, x=x, y=0, w=w, h=h))
    return RegionTrack(role="main", keys=keys)


def _build_split_screen(
    summary: TrackSummary,
    tracks: list[FaceTrack],
    source_w: int,
    source_h: int,
    native_fps: float,
    duration_s: float,
) -> list[RegionTrack]:
    """Two STATIC full-height 9:16 windows, one per speaker.

    `top` = first member of the side_by_side pair, `bottom` = second
    (deterministic: pair order from the summary). Each window is centered
    on that speaker's mean centroid, clamped to [0, source_w - w].
    """
    pair = summary.side_by_side_pair
    assert pair is not None  # only reached via decide_layout / explicit request
    w, h = _track_window(source_w, source_h)
    hi = source_w - w
    out: list[RegionTrack] = []
    for role, tid in (("top", pair[0]), ("bottom", pair[1])):
        track = _find_track(tracks, tid, Layout.split_screen)
        cx, _cy, _bw, _bh = _mean_centroid_and_size(track, source_w, source_h)
        x = int(_clamp(round(cx - w / 2.0), 0, hi))
        out.append(_static_track(role, x, 0, w, h, native_fps, duration_s))
    return out


def _build_facecam(
    summary: TrackSummary,
    tracks: list[FaceTrack],
    source_w: int,
    source_h: int,
    native_fps: float,
    duration_s: float,
) -> list[RegionTrack]:
    """`main` = static EXACT 9:16 centered content window; `face` = STATIC
    enlarged facecam square on the candidate's mean centroid.

    Face window: square side = round(max(mean face w, h) *
    FACECAM_ENLARGE), clamped to [1, min(source_w, source_h)], centered on
    the mean centroid and clamped to stay inside the source.
    """
    tid = min(summary.facecam_candidate_track_ids)  # deterministic (ids sorted)
    track = _find_track(tracks, tid, Layout.facecam)
    cx, cy, bw, bh = _mean_centroid_and_size(track, source_w, source_h)

    s = max(1, min(int(round(max(bw, bh) * FACECAM_ENLARGE)), min(source_w, source_h)))
    fx = int(_clamp(round(cx - s / 2.0), 0, source_w - s))
    fy = int(_clamp(round(cy - s / 2.0), 0, source_h - s))

    mx, my, mw, mh = _center_window(source_w, source_h)
    return [
        _static_track("main", mx, my, mw, mh, native_fps, duration_s),
        _static_track("face", fx, fy, s, s, native_fps, duration_s),
    ]


def build_tracks(
    layout: Layout,
    summary: TrackSummary,
    source_w: int,
    source_h: int,
    native_fps: float,
    duration_s: float,
    shot_boundaries: list[float],
    tracks: list[FaceTrack],
) -> list[RegionTrack]:
    """Build the per-layout crop tracks for one clip (pure math).

    Judgment note (docs/spec/m6.md M6d): the card's signature carries only
    `summary`, but `TrackSummary` holds track IDS, not the per-frame boxes
    that face_track EMA / split-facecam centering need — so `tracks` (the
    `FaceTrack` list from `clipper.tracking.associate`) is REQUIRED as the
    last parameter.

    Every returned track has exactly `round(duration_s * native_fps) + 1`
    keys at clip-relative times `0.0, 1/fps, …`, integer SOURCE-pixel
    coords inside the source, and per-track-consistent w/h.
    """
    _check_dims(source_w, source_h, native_fps, duration_s)
    if layout == Layout.center_crop:
        x, y, w, h = _center_window(source_w, source_h)
        return [_static_track("main", x, y, w, h, native_fps, duration_s)]
    if layout == Layout.face_track:
        tid = summary.dominant_track_id
        if tid is None:
            raise ValueError("face_track layout requires a dominant track")
        return [_build_face_track(
            _find_track(tracks, tid, Layout.face_track),
            source_w, source_h, native_fps, duration_s, shot_boundaries,
        )]
    if layout == Layout.split_screen:
        return _build_split_screen(
            summary, tracks, source_w, source_h, native_fps, duration_s)
    if layout == Layout.facecam:
        return _build_facecam(
            summary, tracks, source_w, source_h, native_fps, duration_s)
    raise ValueError(f"unknown layout: {layout!r}")  # pragma: no cover
