"""
YuNet decode + NMS helpers (M6b — docs/spec/m6.md).

PURE functions: no I/O, no cv2, no onnxruntime import, no model files.
Unit-testable on CPU with synthetic tensors.

YuNet anchor/prior constants
----------------------------
The YuNet ONNX (OpenCV Zoo, face_detection_yunet_2023mar.onnx) is a
single-branch SSD-style detector. It emits raw box deltas `loc`,
class logits `conf` (background, foreground) and `iou` (soft-NMS hint
— kept at the model outputs, unused by our hard NMS) per anchor:

- feature level 0: stride 8,  min_size 16
- feature level 1: stride 16, min_size 32
- feature level 2: stride 32, min_size 64

For an L x L input the anchor counts are (L/8)^2 + (L/16)^2 + (L/32)^2
(8400 anchors at the default 640x640 `det_size`). Anchor centers sit
at half-pixel offsets of each cell of the level's feature map:

    cx = (j + 0.5) * stride,  cy = (i + 0.5) * stride,  w = h = min_size

SSD decoding (variance 1 on all axes, the YuNet convention):

    cx' = cx + loc_x * w          cy' = cy + loc_y * h
    w'  = w * exp(loc_w)          h'  = h * exp(loc_h)
    box = [cx' - w'/2, cy' - h'/2, cx' + w'/2, cy' + h'/2]

Foreground probability is a 2-class softmax over `conf` (same as
OpenCV's `FaceDetectorYN`):  p = e^fg / (e^bg + e^fg).
"""

from __future__ import annotations

import math

# --- YuNet anchor/prior constants (module-level, documented above) ---
YUNET_STRIDES = (8, 16, 32)
YUNET_MIN_SIZES = (16, 32, 64)
YUNET_ANCHOR_ASPECT = 1.0  # square anchors (w == h == min_size)
YUNET_DECODE_VARIANCE = 1.0  # loc deltas are already scaled by variance
# `conf` layout: [background, foreground] — foreground is index -1.
YUNET_FG_INDEX = -1


def anchor_grid(
    strides: tuple[int, ...] = YUNET_STRIDES,
    min_sizes: tuple[int, ...] = YUNET_MIN_SIZES,
    input_w: int = 640,
    input_h: int = 640,
) -> list[tuple[float, float, float, float]]:
    """Deterministic anchor list: (cx, cy, w, h) per level in
    (level, row, col) order. `w == h == min_size * aspect`."""
    anchors: list[tuple[float, float, float, float]] = []
    for stride, min_size in zip(strides, min_sizes, strict=True):
        fw = (input_w + stride - 1) // stride
        fh = (input_h + stride - 1) // stride
        for i in range(fh):
            for j in range(fw):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                side = min_size * YUNET_ANCHOR_ASPECT
                anchors.append((cx, cy, side, side))
    return anchors


def _as_2d(seq, expected_inner: int, name: str):
    """Coerce (N, K) / (K, N) / (1, K, N) inputs to rows (N, K).

    Accepts plain nested lists/tuples and numpy-like arrays (duck-typed
    via ``ndim``/``shape``/``tolist`` — no numpy import here, so this
    module stays dependency-free)."""
    if hasattr(seq, "ndim") and hasattr(seq, "shape"):
        shape = tuple(int(s) for s in seq.shape)
        arr = seq.tolist()
    else:
        probe = seq
        shape = []
        while isinstance(probe, (list, tuple)):
            shape.append(len(probe))
            if probe:
                probe = probe[0]
            else:
                break
        arr = seq
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
        arr = arr[0]
    if len(shape) != 2:
        raise ValueError(
            f"{name}: expected shape (N, {expected_inner}) or "
            f"({expected_inner}, N), got {shape}"
        )
    outer, inner_dim = shape
    if inner_dim == expected_inner:
        return [list(row) for row in arr]
    if outer == expected_inner:
        return [[arr[k][n] for k in range(expected_inner)] for n in range(inner_dim)]
    raise ValueError(
        f"{name}: expected shape (N, {expected_inner}) or "
        f"({expected_inner}, N), got {shape}"
    )


def decode_yunet(
    loc,
    conf,
    iou,  # noqa: ARG001 — kept for API symmetry; YuNet iou is a soft-NMS hint and is unused here
    strides: tuple[int, ...] = YUNET_STRIDES,
    min_sizes: tuple[int, ...] = YUNET_MIN_SIZES,
    input_w: int = 640,
    input_h: int = 640,
    score_threshold: float = 0.6,
):
    """Decode YuNet `loc`/`conf` deltas into score-filtered raw boxes.

    Args:
        loc: box deltas — (N, 4) or (1, 4, N); per-anchor (x, y, w, h).
        conf: class logits — (N, 2) or (1, 2, N); [background, foreground].
        iou: per-anchor iou hints — (N,) / (1, 1, N) / (1, N); UNUSED by the
            hard NMS below (accepted so callers can pass model output as-is).
        strides / min_sizes: YuNet anchor constants (YUNET_STRIDES /
            YUNET_MIN_SIZES by default).
        input_w / input_h: the model INPUT size the deltas refer to
            (the detector's `det_size`), i.e. where output pixels live.
        score_threshold: keep anchors with foreground p >= threshold.

    Returns:
        list of raw boxes ``(x1, y1, x2, y2, score)`` in INPUT-IMAGE pixel
        coordinates (clamped to ``[0, input_w] x [0, input_h]``, x1 < x2,
        y1 < y2), in deterministic anchor order (level, row, col).
    """
    locs = _as_2d(loc, 4, "loc")
    confs = _as_2d(conf, 2, "conf")
    if len(locs) != len(confs):
        # mismatched model output is a bug — fail loudly, do not guess
        raise ValueError(
            f"decode_yunet: loc has {len(locs)} deltas but conf has {len(confs)}"
        )
    n = len(locs)
    anchors = anchor_grid(strides, min_sizes, input_w, input_h)
    if n != len(anchors):
        raise ValueError(
            f"decode_yunet: {n} anchor deltas but the anchor grid for "
            f"input {input_w}x{input_h} with strides {strides} and min "
            f"sizes {min_sizes} has {len(anchors)} anchors"
        )
    boxes: list[tuple[float, float, float, float, float]] = []
    for (cx, cy, w, h), (dx, dy, dw, dh), (bg, fg) in zip(
        anchors[:n], locs[:n], confs[:n], strict=True
    ):
        # numerically stable 2-class softmax for foreground probability
        m = max(bg, fg)
        e_bg, e_fg = math.exp(bg - m), math.exp(fg - m)
        p = e_fg / (e_bg + e_fg)
        if p < score_threshold:
            continue
        var = YUNET_DECODE_VARIANCE
        cx_p = cx + dx * w * var
        cy_p = cy + dy * h * var
        w_p = w * math.exp(dw * var)
        h_p = h * math.exp(dh * var)
        x1 = min(max(cx_p - w_p / 2.0, 0.0), float(input_w))
        y1 = min(max(cy_p - h_p / 2.0, 0.0), float(input_h))
        x2 = min(max(cx_p + w_p / 2.0, 0.0), float(input_w))
        y2 = min(max(cy_p + h_p / 2.0, 0.0), float(input_h))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2, p))
    return boxes


def _iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0.0 else 0.0


def nms(
    boxes: list[tuple],
    scores: list[float],
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> list[int]:
    """Greedy NMS — deterministic.

    Args:
        boxes: (x1, y1, x2, y2) boxes (raw boxes truncated to 4-tuples).
        scores: parallel score list (index-aligned with `boxes`).
        nms_threshold: a box is SUPPRESSED if its IoU with an already-kept
            (higher-scored) box is >= `nms_threshold` (same convention as
            M4a's candidate NMS).
        top_k: cap on the number of kept indices.

    Returns:
        Indices INTO `boxes`, sorted by score DESC with ties broken by
        index ASC — fully deterministic for identical input order.
    """
    if top_k <= 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    keep: list[int] = []
    for i in order:
        if len(keep) >= top_k:
            break
        box_i = tuple(boxes[i][:4])
        suppressed = False
        for j in keep:
            if _iou(box_i, tuple(boxes[j][:4])) >= nms_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return keep


def normalize_box(
    box: tuple, input_w: int, input_h: int
) -> tuple[float, float, float, float]:
    """Map an (x1, y1, x2, y2) box in INPUT-IMAGE pixels to normalized
    (x, y, w, h) in [0, 1] — the `FaceDetection` contract (M6a).

    The detector resizes the ENTIRE frame to (input_w, input_h), so
    normalization is exactly a scale by the input dims."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    x = x1 / input_w
    y = y1 / input_h
    w = max(0.0, min(1.0, (x2 - x1) / input_w))
    h = max(0.0, min(1.0, (y2 - y1) / input_h))
    if w <= 0.0 or h <= 0.0:
        raise ValueError(
            f"normalize_box: degenerate box {box[:4]} (input "
            f"{input_w}x{input_h})"
        )
    x = min(max(x, 0.0), 1.0 - w)
    y = min(max(y, 0.0), 1.0 - h)
    return (x, y, w, h)
