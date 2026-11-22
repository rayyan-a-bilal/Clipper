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

import glob
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _cuda_libs_resolvable() -> bool:
    """True if the CUDA 13 / cuDNN 9 runtime deps of onnxruntime's CUDA EP
    are already on the dynamic loader path."""
    import ctypes

    for soname in ("libcublasLt.so.13", "libcudnn_ops.so.9", "libcudart.so.13"):
        try:
            ctypes.CDLL(soname)
        except OSError:
            return False
    return True


def _ensure_cuda_libs() -> None:
    """Find CUDA 13 + cuDNN 9 libraries and re-exec once with them on
    LD_LIBRARY_PATH (guarded).

    onnxruntime's CUDAExecutionProvider dlopens `libcublasLt.so.13` +
    cuDNN 9 at session creation. glibc's dlopen search paths are frozen at
    exec time, so a mid-process env change is futile — we re-exec the
    script exactly once with a corrected LD_LIBRARY_PATH (Ollama ships a
    cuda_v13 bundle; cuDNN 9 lives in nvidia pip wheels)."""
    if os.environ.get("_CLIPPER_CUDA_LIBS_FIX") == "1":
        return  # already re-exec'd once; let the real error surface
    if _cuda_libs_resolvable():
        return
    patterns = (
        "/usr/local/lib/ollama/cuda_v*",
        "/usr/local/cuda-*/lib64",
        "/usr/lib/x86_64-linux-gnu",
        os.path.expanduser("~/.local/share") + "/**/nvidia/cu13/lib",
        os.path.expanduser("~/.local/share") + "/**/nvidia/cublas/lib",
        os.path.expanduser("~/.local/share") + "/**/nvidia/cudnn/lib",
        "/usr/local/cuda/lib64",
    )
    dirs: list[str] = []
    for pat in patterns:
        for d in sorted(glob.glob(pat, recursive=True)):
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    need = ("libcublasLt.so.13", "libcudnn_ops.so.9", "libcudart.so.13")
    covered = True
    for soname in need:
        if not any(os.path.exists(os.path.join(d, soname)) for d in dirs):
            covered = False
            break
    if not covered:
        return  # let the normal failure message report the problem
    old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + ((":" + old) if old else "")
    print("[preflight] CUDA 13/cuDNN 9 libs not on loader path; re-executing with")
    print(f"[preflight] LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
    os.environ["_CLIPPER_CUDA_LIBS_FIX"] = "1"
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])

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
    rows = rng.integers(0, h, size=(h // 20,))
    frame[rows, :] = rng.integers(0, 255, size=(h // 20, 1, 3)).astype(np.uint8)
    return frame


def main() -> int:
    _ensure_cuda_libs()
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
