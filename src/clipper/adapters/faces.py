"""
Face-detector adapters (M6b — docs/spec/m6.md).

GPU0 (RTX 2080 Super, `cuda:0`) is the ONLY production face-detection
path: YuNet ONNX via `onnxruntime` `CUDAExecutionProvider`. If the CUDA
EP is not actually available the detector FAILS LOUDLY — it never
silently runs on CPU (docs/spec/m6.md, hardware rules in the project
skill; the P100s never do video).

CPU face detection (`cv2.FaceDetectorYN`) exists ONLY as an explicit
debug path behind `get_face_detector(cfg, cpu=True)` (docs/spec/m6.md
decision 9 — logged loudly).

`onnxruntime` and `cv2` are imported LAZILY (inside `__init__`), so
importing this module stays light and CPU-only. Pure decode/NMS lives in
`clipper.facedec` (no I/O).
"""

import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

from .. import facedec
from ..config import Config
from ..models import FaceDetection

logger = logging.getLogger("clipper.faces")


class FaceDetectorError(RuntimeError):
    """Clean failure for face-detection setup/detection problems."""


@runtime_checkable
class FaceDetector(Protocol):
    """Protocol for per-frame face detectors (M6b)."""

    def detect(self, frame_bgr, t: float) -> list[FaceDetection]:
        """Detect faces in one BGR frame at segment-local time `t` (s).

        Returns `FaceDetection` rows with x/y/w/h normalized 0..1 to the
        frame (M6a contract)."""
        ...

    def close(self) -> None:
        """Release any underlying session/session resources."""

    def name(self) -> str:
        """Detector id recorded in `ClipLayoutArtifact.detector`."""


def parse_cuda_device_id(vision: str) -> int:
    """`"cuda:0"` -> 0. Anything else is a M6 rule violation (GPU0 only)."""
    parts = (vision or "").strip().lower().split(":")
    if len(parts) != 2 or parts[0] != "cuda":
        raise FaceDetectorError(
            f"M6 face detection must run on GPU0 (docs/spec/m6.md): "
            f"devices.vision={vision!r} is not 'cuda:<index>'"
        )
    try:
        index = int(parts[1])
    except ValueError as e:
        raise FaceDetectorError(
            f"devices.vision={vision!r}: bad CUDA index"
        ) from e
    if index < 0:
        raise FaceDetectorError(f"devices.vision={vision!r}: negative CUDA index")
    return index


def _validate_frame(frame_bgr) -> None:
    if not isinstance(frame_bgr, np.ndarray):
        raise FaceDetectorError(
            "detect() expects a BGR numpy frame (uint8, HxWx3)"
        )
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise FaceDetectorError(
            f"detect() expects an HxWx3 frame, got shape {frame_bgr.shape}"
        )
    if frame_bgr.dtype != np.uint8:
        raise FaceDetectorError(
            f"detect() expects a uint8 BGR frame, got dtype {frame_bgr.dtype}"
        )


class YuNetOnnxDetector:
    """YuNet ONNX face detector on GPU0 via onnxruntime CUDA EP.

    - `onnxruntime` is imported LAZILY in `__init__` (module import stays
      CPU-only and dependency-free).
    - `InferenceSession` is built with
      `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` and
      `provider_options=[{"device_id": <index from cfg.devices.vision>}, {}]`;
      the CUDA index comes from `cfg.devices.vision` (default `cuda:0`).
    - FAILS LOUDLY (`FaceDetectorError`) if `CUDAExecutionProvider` is not
      available in onnxruntime or not actually selected by the session —
      never silently on CPU.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._det_size = int(cfg.faces.det_size)
        model_path = cfg.faces.model_path
        if not os.path.exists(model_path):
            raise FaceDetectorError(
                f"YuNet model not found at {model_path!r}; download it "
                f"manually with `bash scripts/install_models.sh`"
            )
        self._device_id = parse_cuda_device_id(cfg.devices.vision)
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise FaceDetectorError(
                "onnxruntime is not installed; install `onnxruntime-gpu` "
                "to enable M6 GPU face detection"
            ) from e

        available = list(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise FaceDetectorError(
                "CUDAExecutionProvider is NOT available in onnxruntime "
                f"(available: {available}). M6 face detection must run on "
                "GPU0 — it will NEVER silently fall back to CPU "
                "(docs/spec/m6.md). Check the onnxruntime-gpu install and "
                "the CUDA runtime libraries."
            )
        try:
            session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"device_id": self._device_id}, {}],
            )
        except Exception as e:
            raise FaceDetectorError(
                f"onnxruntime InferenceSession failed for {model_path!r}: {e}"
            ) from e
        if "CUDAExecutionProvider" not in list(session.get_providers()):
            raise FaceDetectorError(
                "InferenceSession did not use CUDAExecutionProvider "
                f"(session providers: {list(session.get_providers())}); "
                "refusing to run face detection on CPU"
            )
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name.lower() for o in session.get_outputs()]

    def name(self) -> str:
        return "yunet-onnx-cuda"

    def close(self) -> None:
        self._session = None
        self._input_name = None

    def _require_open(self) -> None:
        if self._session is None:
            raise FaceDetectorError("YuNetOnnxDetector is closed")

    def _order_outputs(self, outputs: list):
        """Model outputs -> (loc, conf, iou) by name; positional (0, 1, 2)
        fallback only when the names are unrecognizable."""
        by_name = {}
        for i, nm in enumerate(self._output_names):
            key = nm.replace("_", "")
            if key in ("loc", "conf", "iou") and key not in by_name:
                by_name[key] = i
        if {"loc", "conf", "iou"} <= by_name.keys():
            return (
                outputs[by_name["loc"]],
                outputs[by_name["conf"]],
                outputs[by_name["iou"]],
            )
        if len(outputs) >= 3:
            return (outputs[0], outputs[1], outputs[2])
        raise FaceDetectorError(
            f"YuNet model returned {len(outputs)} outputs "
            f"{self._output_names}; expected loc/conf/iou"
        )

    def detect(self, frame_bgr, t: float) -> list[FaceDetection]:
        """Detect faces in one BGR frame; boxes normalized 0..1, `t`/score
        attached. Deterministic (score DESC, then x, y)."""
        self._require_open()
        _validate_frame(frame_bgr)
        cfg = self._cfg
        # NCHW blob of the whole frame resized to the model input size.
        frame = np.ascontiguousarray(frame_bgr)
        resized = np.ascontiguousarray(
            np.asarray(_resize_to_size(frame, self._det_size))
        )
        blob = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        outputs = self._session.run(None, {self._input_name: blob})
        loc, conf, iou = self._order_outputs(outputs)
        raw = facedec.decode_yunet(
            loc,
            conf,
            iou,
            input_w=self._det_size,
            input_h=self._det_size,
            score_threshold=float(cfg.faces.score_threshold),
        )
        boxes = [r[:4] for r in raw]
        scores = [r[4] for r in raw]
        kept = facedec.nms(
            boxes,
            scores,
            nms_threshold=float(cfg.faces.nms_threshold),
            top_k=int(cfg.faces.top_k),
        )
        detections: list[FaceDetection] = []
        for idx in kept:
            x, y, w, h = facedec.normalize_box(
                raw[idx], self._det_size, self._det_size
            )
            detections.append(
                FaceDetection(
                    t=float(t), x=x, y=y, w=w, h=h, score=float(scores[idx])
                )
            )
        detections.sort(key=lambda d: (-d.score, d.x, d.y, d.t))
        return detections


def _resize_to_size(frame: np.ndarray, det_size: int) -> np.ndarray:
    """Resize an HxWx3 BGR frame to det_size x det_size via cv2 (lazy)."""
    try:
        import cv2
    except ImportError as e:
        raise FaceDetectorError(
            "opencv-python-headless is unavailable; it is required to "
            "resize frames for YuNet"
        ) from e
    return cv2.resize(frame, (int(det_size), int(det_size)))


class Cv2CpuFaceDetector:
    """`cv2.FaceDetectorYN` CPU detector — EXPLICIT debug path ONLY
    (docs/spec/m6.md decision 9). Never returned by the factory unless
    `cpu=True`."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._det_size = int(cfg.faces.det_size)
        model_path = cfg.faces.model_path
        if not os.path.exists(model_path):
            raise FaceDetectorError(
                f"YuNet model not found at {model_path!r}; download it "
                f"manually with `bash scripts/install_models.sh`"
            )
        try:
            import cv2
        except ImportError as e:
            raise FaceDetectorError(
                "opencv-python-headless is unavailable; it is required for "
                "the CPU (cv2.FaceDetectorYN) face-detection debug path"
            ) from e
        try:
            self._detector = cv2.FaceDetectorYN_create(
                model_path,
                "",
                (self._det_size, self._det_size),
                float(cfg.faces.score_threshold),
                float(cfg.faces.nms_threshold),
                int(cfg.faces.top_k),
            )
        except Exception as e:
            raise FaceDetectorError(
                f"cv2.FaceDetectorYN_create failed for {model_path!r}: {e}"
            ) from e

    def name(self) -> str:
        return "yunet-cpu"

    def close(self) -> None:
        self._detector = None

    def detect(self, frame_bgr, t: float) -> list[FaceDetection]:
        if self._detector is None:
            raise FaceDetectorError("Cv2CpuFaceDetector is closed")
        _validate_frame(frame_bgr)
        rectangles, scores = self._detector.detect(frame_bgr)
        det_size = self._det_size
        detections = [
            FaceDetection(
                t=float(t),
                x=float(rect[0]) / det_size,
                y=float(rect[1]) / det_size,
                w=float(rect[2]) / det_size,
                h=float(rect[3]) / det_size,
                score=float(scores[i]),
            )
            for i, rect in enumerate(rectangles)
        ]
        detections.sort(key=lambda d: (-d.score, d.x, d.y, d.t))
        return detections


class FixtureFaceDetector:
    """CPU-only fixture: canned detections per FRAME INDEX, deterministic,
    records calls. No model, no GPU, no network."""

    def __init__(
        self,
        detections: dict[int, list] | None = None,
        fps: float = 30.0,
    ):
        self.fps = float(fps)
        self._detections: dict[int, list[FaceDetection]] = {}
        for key, rows in (detections or {}).items():
            self._detections[int(key)] = [
                d if isinstance(d, FaceDetection) else FaceDetection(**d)
                for d in rows
            ]
        # (frame_index, t) in call order — for test assertions.
        self.calls: list[tuple[int, float]] = []

    def name(self) -> str:
        return "fixture-face-detector"

    def close(self) -> None:
        return None

    def detect(self, frame_bgr, t: float) -> list[FaceDetection]:
        """Frame index = round(t * fps). Returns a COPY of the canned
        detections for that index (empty list when none are configured)."""
        index = int(round(float(t) * self.fps))
        self.calls.append((index, float(t)))
        rows = self._detections.get(index, [])
        return [FaceDetection(**d.model_dump()) for d in rows]


def get_face_detector(cfg: Config, cpu: bool = False) -> FaceDetector:
    """Factory (M6b):

    - default (`cpu=False`): `YuNetOnnxDetector` — GPU0 onnxruntime CUDA
      EP. Raises `FaceDetectorError` loudly if the GPU path is unavailable.
    - `cpu=True`: `Cv2CpuFaceDetector` — the EXPLICIT debug path only,
      logged loudly (docs/spec/m6.md decision 9).

    A fresh detector is returned per call (cheap to construct; the GPU
    session is the only expensive part and it is deterministic).
    """
    if cpu:
        logger.warning(
            "CPU face detection (cv2.FaceDetectorYN) ENABLED — explicit "
            "debug path only (docs/spec/m6.md decision 9); production "
            "must use GPU0 / yunet-onnx-cuda"
        )
        return Cv2CpuFaceDetector(cfg)
    return YuNetOnnxDetector(cfg)
