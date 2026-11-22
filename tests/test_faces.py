"""M6b — YuNet ONNX face detector (docs/spec/m6.md).

CPU-only tests: decode/NMS pure helpers, normalized coordinate mapping,
FixtureFaceDetector behavior, loud CUDA-EP failure via a mocked
`onnxruntime`, factory routing (cv2 path only via `cpu=True`), and the
guarantee that importing `clipper.adapters.faces` never pulls in
`onnxruntime` or `cv2`.

The `@pytest.mark.gpu` oracle test compares `YuNetOnnxDetector` against
`cv2.FaceDetectorYN` on the same frame (IoU >= 0.9 for matched faces;
skips when the model file or the CUDA EP is unavailable — it never runs
under `-m "not gpu"`).
"""

import importlib
import os
import sys

import numpy as np
import pytest

from clipper import facedec
from clipper.adapters.faces import (
    Cv2CpuFaceDetector,
    FaceDetectorError,
    FixtureFaceDetector,
    YuNetOnnxDetector,
    get_face_detector,
    parse_cuda_device_id,
)
from clipper.config import Config, FacesConfig
from clipper.models import FaceDetection

DET_SIZE = 640


def _n_anchors(input_w: int = DET_SIZE, input_h: int = DET_SIZE) -> int:
    """(L/8)^2 + (L/16)^2 + (L/32)^2 — 8400 at 640x640."""
    total = 0
    for s in facedec.YUNET_STRIDES:
        fw = (input_w + s - 1) // s
        fh = (input_h + s - 1) // s
        total += fw * fh
    return total


def _synthetic_outputs(n: int, fg_mask: set[int], boxes: list | None = None):
    """Synthetic YuNet outputs for `n` anchors.

    Anchors in `fg_mask` get logit (bg, fg) = (0, 3) -> p ~= 0.95;
    the rest get (3, 0) -> p ~= 0.05. Box deltas default to zero (box ==
    the anchor box); `boxes[i]` optionally overrides anchor i's deltas."""
    loc = np.zeros((n, 4), dtype=np.float32)
    conf = np.full((n, 2), 3.0, dtype=np.float32)
    for i in range(n):
        if i in fg_mask:
            conf[i, 0], conf[i, 1] = 0.0, 3.0
        else:
            conf[i, 0], conf[i, 1] = 3.0, 0.0
        if boxes is not None and i < len(boxes):
            loc[i] = boxes[i]
    iou = np.ones(n, dtype=np.float32)
    return loc, conf, iou


def _cfg(tmp_path) -> Config:
    return Config(
        faces={
            "model_path": str(tmp_path / "yunet.onnx"),
            "score_threshold": 0.6,
            "nms_threshold": 0.3,
            "top_k": 5000,
            "det_size": DET_SIZE,
        }
    )


def _cfg_with_model(tmp_path) -> Config:
    cfg = _cfg(tmp_path)
    (tmp_path / "yunet.onnx").write_bytes(b"dummy onnx blob")
    return cfg


# --- decode_yunet -----------------------------------------------------------


def test_decode_zero_deltas_gives_exact_anchor_box():
    """Anchor 0: level 0 (stride 8, min_size 16), (i=0, j=0): cx=cy=4.
    x1/y1 = -4 clamps to 0 (boxes never leave the input image)."""
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask={0})
    boxes = facedec.decode_yunet(loc, conf, iou)
    x1, y1, x2, y2, score = boxes[0]
    assert (x1, y1) == pytest.approx((0.0, 0.0))  # -4 clamped to image edge
    assert (x2, y2) == pytest.approx((4.0 + 8.0, 4.0 + 8.0))
    # softmax over (0, 3): p = 1 / (1 + e^-3)
    assert score == pytest.approx(1.0 / (1.0 + np.exp(-3.0)))


def test_decode_score_filtering_counts_only_fg():
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask={0, 5, 10})
    boxes = facedec.decode_yunet(loc, conf, iou)
    assert len(boxes) == 3
    assert all(b[4] >= 0.6 for b in boxes)


def test_decode_all_below_threshold_is_empty():
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask=set())
    assert facedec.decode_yunet(loc, conf, iou) == []


def test_decode_box_sanity_and_deterministic_order():
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask={0, 5, 10})
    boxes = facedec.decode_yunet(loc, conf, iou)
    # deterministic order: ascending anchor index (all same score here)
    assert boxes == facedec.decode_yunet(loc, conf, iou)
    for b in boxes:
        x1, y1, x2, y2, s = b
        assert 0.0 <= x1 < x2 <= float(DET_SIZE)
        assert 0.0 <= y1 < y2 <= float(DET_SIZE)
        assert 0.0 < s < 1.0


def test_decode_clamps_to_image_bounds():
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask={0})
    loc[0] = np.array([-0.25, -0.25, 0.0, 0.0], dtype=np.float32)  # partly off-canvas
    boxes = facedec.decode_yunet(loc, conf, iou)
    x1, y1, x2, y2, _s = boxes[0]
    assert x1 == pytest.approx(0.0)  # -8 clamped
    assert (x2, y2) == pytest.approx((0.0 + 8.0, 0.0 + 8.0))


def test_decode_drops_fully_off_canvas_box():
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask={0})
    loc[0] = np.array([-3.0, -3.0, 0.0, 0.0], dtype=np.float32)  # entirely off-canvas
    assert facedec.decode_yunet(loc, conf, iou) == []


def test_decode_rejects_mismatched_loc_conf_lengths():
    loc, conf, iou = _synthetic_outputs(10, fg_mask={0})
    with pytest.raises(ValueError, match="anchor"):
        facedec.decode_yunet(loc, conf, iou)


def test_decode_rejects_bad_shapes():
    with pytest.raises(ValueError):
        facedec.decode_yunet(np.zeros((4, 4)), np.zeros((4, 4)), None)
    with pytest.raises(ValueError):
        facedec.decode_yunet(None, None, None)


def test_decode_accepts_channel_first_shapes():
    """YuNet emits (1, C, N) — decodes without a manual transpose."""
    n = _n_anchors()
    loc, conf, iou = _synthetic_outputs(n, fg_mask={0, 5})
    boxes = facedec.decode_yunet(
        loc.transpose(1, 0)[np.newaxis, ...],
        conf.transpose(1, 0)[np.newaxis, ...],
        iou[np.newaxis, np.newaxis, :],
    )
    assert len(boxes) == 2


# --- nms --------------------------------------------------------------------


def test_nms_suppresses_overlapping_lower_scored():
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (1.0, 1.0, 11.0, 11.0),
        (100.0, 100.0, 110.0, 110.0),
    ]
    scores = [0.9, 0.8, 0.7]
    assert facedec.nms(boxes, scores, nms_threshold=0.3) == [0, 2]


def test_nms_keeps_disjoint_boxes():
    boxes = [(0.0, 0.0, 10.0, 10.0), (12.0, 0.0, 22.0, 10.0)]
    scores = [0.9, 0.8]
    assert facedec.nms(boxes, scores, nms_threshold=0.3) == [0, 1]


def test_nms_suppresses_barely_overlapping():
    # IoU = inter/union = 5/195 ~ 0.0256 < 0.3 -> both kept
    boxes = [(0.0, 0.0, 10.0, 10.0), (9.5, 0.0, 19.5, 10.0)]
    scores = [0.9, 0.8]
    kept = facedec.nms(boxes, scores, nms_threshold=0.3)
    # intersection 0.5x10=5, union 100+100-5=195 -> IoU 0.0256 < 0.3 -> both kept
    assert kept == [0, 1]


def test_nms_suppresses_strong_overlap():
    boxes = [(0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)]  # IoU = 50/150 > 0.3
    scores = [0.9, 0.8]
    assert facedec.nms(boxes, scores, nms_threshold=0.3) == [0]


def test_nms_top_k_cap():
    boxes = [(i * 100.0, 0.0, i * 100.0 + 10.0, 10.0) for i in range(10)]
    scores = [1.0 - 0.01 * i for i in range(10)]  # all disjoint, distinct scores
    kept = facedec.nms(boxes, scores, nms_threshold=0.3, top_k=3)
    assert kept == [0, 1, 2]


def test_nms_deterministic_and_tie_break_by_index():
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (100.0, 0.0, 110.0, 10.0),
        (200.0, 0.0, 210.0, 10.0),
    ]
    scores = [0.8, 0.8, 0.8]  # ties -> index ASC; all disjoint -> all kept
    first = facedec.nms(boxes, scores, nms_threshold=0.3, top_k=5)
    assert first == [0, 1, 2]
    for _ in range(3):
        assert facedec.nms(boxes, scores, nms_threshold=0.3, top_k=5) == first


def test_nms_empty_inputs():
    assert facedec.nms([], [], nms_threshold=0.3) == []
    assert facedec.nms([(0.0, 0.0, 1.0, 1.0)], [0.9], nms_threshold=0.3, top_k=0) == []


# --- normalized mapping -----------------------------------------------------


def test_normalize_box_known_box():
    """64x64 box at (100, 200) in a 640x640 input -> exact fraction of frame."""
    x, y, w, h = facedec.normalize_box((100.0, 200.0, 164.0, 264.0), DET_SIZE, DET_SIZE)
    assert x == pytest.approx(100 / 640)
    assert y == pytest.approx(200 / 640)
    assert w == pytest.approx(64 / 640)
    assert h == pytest.approx(64 / 640)


def test_normalize_box_clamped_into_unit_square():
    x, y, w, h = facedec.normalize_box((-10.0, -10.0, 659.0, 659.0), DET_SIZE, DET_SIZE)
    assert x >= 0.0 and y >= 0.0
    assert x + w <= 1.0
    assert y + h <= 1.0


def test_normalize_box_rejects_degenerate():
    with pytest.raises(ValueError):
        facedec.normalize_box((10.0, 10.0, 10.0, 20.0), DET_SIZE, DET_SIZE)


# --- FixtureFaceDetector ----------------------------------------------------


def test_fixture_returns_configured_detections_and_records_calls():
    det = FixtureFaceDetector(
        {
            1: [FaceDetection(t=1 / 30, x=0.1, y=0.2, w=0.3, h=0.4, score=0.95)],
            31: [FaceDetection(t=31 / 30, x=0.5, y=0.5, w=0.2, h=0.2, score=0.8)],
        }
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert det.detect(frame, t=0.0) == []
    assert det.detect(frame, t=1 / 30) == [
        FaceDetection(t=1 / 30, x=0.1, y=0.2, w=0.3, h=0.4, score=0.95)
    ]
    assert det.calls == [(0, 0.0), (1, 1 / 30)]
    det.close()
    assert det.detect(frame, t=0.0) == []  # deterministic after close


def test_fixture_accepts_dict_rows_and_returns_copies():
    det = FixtureFaceDetector(
        {0: [{"t": 0.0, "x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "score": 0.7}]}
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    dets = det.detect(frame, t=0.0)
    dets[0] = dets[0].model_copy(update={"score": 0.0})  # mutate returned row
    assert det.detect(frame, t=0.0)[0].score == pytest.approx(0.7)
    assert det.name() == "fixture-face-detector"


# --- GPU path: fail loudly, never silently on CPU ---------------------------


class _FakeSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]


class _FakeOrtNoCuda:
    version = "0.0-fake"

    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]

    @staticmethod
    def InferenceSession(*a, **k):
        raise RuntimeError("no cuda in fake")


class _FakeSessionGood:
    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def get_inputs(self):
        class _In:
            name = "input"

        return [_In()]

    def get_outputs(self):
        class _Out:
            def __init__(self, name):
                self.name = name

        return [_Out("loc"), _Out("conf"), _Out("iou")]


class _FakeOrtCuda:
    @staticmethod
    def get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def InferenceSession(path, providers=None, provider_options=None, **k):
        _FakeOrtCuda.last_kwargs = {
            "providers": providers,
            "provider_options": provider_options,
        }
        return _FakeSessionGood()


def test_yunet_detector_raises_when_cuda_ep_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtNoCuda)
    with pytest.raises(FaceDetectorError, match="CUDAExecutionProvider"):
        YuNetOnnxDetector(_cfg_with_model(tmp_path))


def test_yunet_detector_raises_when_onnxruntime_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(FaceDetectorError, match="onnxruntime is not installed"):
        YuNetOnnxDetector(_cfg_with_model(tmp_path))


class _FakeOrtCudaAvailableCpuSession:
    """CUDA EP advertised but the session does NOT actually use it."""

    @staticmethod
    def get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def InferenceSession(*a, **k):
        return _FakeSession()


def test_yunet_detector_raises_when_session_loses_cuda(tmp_path, monkeypatch):
    """CUDA EP 'available' but the session doesn't use it -> still fail."""
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtCudaAvailableCpuSession)
    with pytest.raises(FaceDetectorError, match="did not use CUDAExecutionProvider"):
        YuNetOnnxDetector(_cfg_with_model(tmp_path))


def test_yunet_detector_builds_session_with_cuda_device_id(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtCuda)
    det = YuNetOnnxDetector(_cfg_with_model(tmp_path))
    det.close()
    kwargs = _FakeOrtCuda.last_kwargs
    assert kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert kwargs["provider_options"][0] == {"device_id": 0}
    assert det.name() == "yunet-onnx-cuda"


def test_yunet_detector_missing_model_raises(tmp_path):
    with pytest.raises(FaceDetectorError, match="install_models.sh"):
        YuNetOnnxDetector(_cfg(tmp_path))


def test_parse_cuda_device_id_gpu0_only():
    assert parse_cuda_device_id("cuda:0") == 0
    assert parse_cuda_device_id("cuda:1") == 1
    for bad in ("cpu", "cuda:", "cuda:x", "cuda:-1", "", "cuda"):
        with pytest.raises(FaceDetectorError):
            parse_cuda_device_id(bad)


# --- factory routing --------------------------------------------------------


def test_factory_default_is_gpu_path(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtNoCuda)
    with pytest.raises(FaceDetectorError):  # GPU path refused loudly, not a cv2 swap
        get_face_detector(_cfg(tmp_path))
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtCuda)
    det = get_face_detector(_cfg_with_model(tmp_path))
    det.close()
    assert isinstance(det, YuNetOnnxDetector)


def test_factory_cpu_flag_routes_to_cv2_detector(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(
        cv2,
        "FaceDetectorYN_create",
        staticmethod(lambda *a, **k: object()),
        raising=True,
    )
    det = get_face_detector(_cfg_with_model(tmp_path), cpu=True)
    det.close()
    assert isinstance(det, Cv2CpuFaceDetector)
    assert det.name() == "yunet-cpu"


def test_factory_cpu_without_model_fails_loudly(tmp_path):
    with pytest.raises(FaceDetectorError, match="install_models.sh"):
        get_face_detector(_cfg(tmp_path), cpu=True)


# --- config ----------------------------------------------------------------


def test_config_m6b_defaults():
    f = FacesConfig()
    assert f.model_path == "data/models/yunet.onnx"
    assert f.score_threshold == pytest.approx(0.6)
    assert f.nms_threshold == pytest.approx(0.3)
    assert f.top_k == 5000
    assert f.det_size == 640


def test_config_m6a_defaults_still_present():
    f = FacesConfig()
    assert f.vision_fps_cap == pytest.approx(30.0)
    assert f.vision_width == 960
    assert f.vision_height == 540


# --- import hygiene --------------------------------------------------------


def test_module_import_never_pulls_onnxruntime():
    mods = ["clipper.adapters", "clipper.adapters.faces", "clipper.facedec"]
    for m in mods:
        sys.modules.pop(m, None)
    importlib.import_module("clipper.adapters.faces")
    assert "onnxruntime" not in sys.modules


def test_facedec_source_is_dependency_free():
    with open(os.path.join(os.path.dirname(facedec.__file__), "facedec.py")) as fh:
        src = fh.read()
    for banned in ("import cv2", "import onnxruntime", "import numpy"):
        assert banned not in src


# --- optional oracle (GPU only — excluded by `-m "not gpu"`) ----------------


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter)


@pytest.mark.gpu
def test_gpu_oracle_yunet_onnx_matches_cv2():
    """YuNetOnnxDetector boxes must match cv2.FaceDetectorYN on the same
    frame (IoU >= 0.9 for every GPU face). Skips cleanly when the real
    model or the CUDA EP is unavailable."""
    cfg = Config()  # default: data/models/yunet.onnx (CWD-relative)
    model_path = cfg.faces.model_path
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100_000:
        pytest.skip("YuNet model not present/real (run bash scripts/install_models.sh)")
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDA EP unavailable on this host")

    import cv2

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3)).astype(np.uint8)
    cv2.ellipse(frame, (400, 320), (110, 150), 0, 0, 360, (170, 150, 130), -1)

    gpu = YuNetOnnxDetector(cfg)
    try:
        gpu_boxes = [(d.x, d.y, d.x + d.w, d.y + d.h) for d in gpu.detect(frame, t=0.0)]
    finally:
        gpu.close()
    assert len(gpu_boxes) > 0, "oracle: expected at least one face"

    cpu = Cv2CpuFaceDetector(cfg)
    try:
        cpu_boxes = [(d.x, d.y, d.x + d.w, d.y + d.h) for d in cpu.detect(frame, t=0.0)]
    finally:
        cpu.close()
    assert len(cpu_boxes) > 0, "oracle: cv2 FaceDetectorYN detected no face"

    for g in gpu_boxes:
        best = max(_iou(g, c) for c in cpu_boxes)
        msg = f"GPU box {g} unmatched (best IoU {best:.3f} against {cpu_boxes})"
        assert best >= 0.9, msg
