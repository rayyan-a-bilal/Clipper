#!/usr/bin/env bash
# clipper model installer — MANUAL RUN ONLY (never invoked by tests).
#
# Downloads model files into <repo>/data/models/. Every model is SKIPped
# if already present (idempotent). Prints the resolved target path of
# each model.
#
# Usage:  bash scripts/install_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/data/models}"
mkdir -p "${MODELS_DIR}"

# --- M6b: YuNet face detector (OpenCV Zoo) -------------------------------
# Source: OpenCV Zoo — models/face_detection_yunet (face detection model
# used by cv2.FaceDetectorYN and by our onnxruntime adapter, adapters/faces.py).
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_TARGET="${MODELS_DIR}/yunet.onnx"

install_one() {
  local url="$1" target="$2"
  if [[ -f "${target}" && -s "${target}" ]]; then
    echo "[install_models] skip (already present): ${target}"
  else
    echo "[install_models] downloading ${url}"
    local tmp
    tmp="$(mktemp "${target}.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -f '${tmp}'" RETURN
    curl -fSL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"
    mv "${tmp}" "${target}"
  fi
  echo "[install_models] target: ${target}"
}

install_one "${YUNET_URL}" "${YUNET_TARGET}"

echo "[install_models] done."
