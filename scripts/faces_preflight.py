#!/usr/bin/env python3
"""M6b face-detector preflight (GPU0, onnxruntime CUDA EP).

Prints the onnxruntime version and available execution providers, asserts
CUDAExecutionProvider is present, loads the configured YuNet model, runs
ONE dummy 640x640 frame through `YuNetOnnxDetector.detect`, and prints
the detection count + the device actually used.

Exit 0 = success, exit 1 = clean error.

Usage:
    python scripts/faces_preflight.py [optional BGR image path]

(Without an image argument a deterministic synthetic frame is used — it
may legitimately yield 0 faces; the device actually used is the point.)
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clipper.adapters.faces import (  # noqa: E402
    FaceDetectorError,
    YuNetOnnxDetector,
)
from clipper.config import Config  # noqa: E402


def _dummy_frame(det_size: int):
    import numpy as np

    rng = np.random.default_rng(0)  # deterministic
    h = w = int(det_size)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    gx = (np.linspace(0, 255, w) * np.ones((h, 1))).astype(np.uint8)
    frame[..., 0] = gx
    frame[..., 1] = 128
    frame[..., 2] = 40
    # a face-ish ellipse in the middle
    import cv2

    cv2.ellipse(
        frame,
        (w // 2, h // 2),
        (w // 8, h // 6),
        0,
        0,
        360,
        (160, 140, 120),
        -1,
    )
    frame[rng.integers(0, h, size=(h // 20, 3)), :] = rng.integers(
        0, 255, size=(h // 20, 1, 3)
    ).astype(np.uint8)
    return frame


def main() -> int:
    cfg = Config.load_config()
    model_path = cfg.faces.model_path
    if not os.path.exists(model_path):
        print(f"FAIL: YuNet model not found at {model_path!r}")
        print("      run `bash scripts/install_models.sh` first")
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("FAIL: onnxruntime is not installed (install onnxruntime-gpu)")
        return 1

    print(f"onnxruntime version: {ort.__version__}")
    providers = list(ort.get_available_providers())
    print(f"available providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print(
            "FAIL: CUDAExecutionProvider is NOT available — M6 face "
            "detection must run on GPU0 and never silently on CPU"
        )
        return 1

    frame_path = sys.argv[1] if len(sys.argv) > 1 else None
    if frame_path is not None:
        import cv2

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"FAIL: could not read image: {frame_path!r}")
            return 1
        if frame.ndim != 3 or frame.shape[2] != 3:
            print(f"FAIL: not an HxWx3 BGR image: {frame.shape}")
            return 1
        print(f"frame: {frame_path} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        det_size = int(cfg.faces.det_size)
        frame = _dummy_frame(det_size)
        print(f"frame: synthetic dummy ({frame.shape[1]}x{frame.shape[0]})")

    detector = YuNetOnnxDetector(cfg)  # raises FaceDetectorError if no CUDA
    try:
        dets = detector.detect(frame, t=0.0)
    finally:
        detector.close()

    print(f"detector: {detector.name()}")
    print(f"device used: CUDAExecutionProvider (devices.vision={cfg.devices.vision})")
    print(f"detections: {len(dets)}")
    for d in dets[:5]:
        print(
            f"  face: x={d.x:.4f} y={d.y:.4f} w={d.w:.4f} h={d.h:.4f} "
            f"score={d.score:.4f} t={d.t}"
        )
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FaceDetectorError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
