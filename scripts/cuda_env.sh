# scripts/cuda_env.sh — source this before GPU (onnxruntime-gpu) runs:
#   source scripts/cuda_env.sh && python scripts/layout_job.py <job_id>
#
# Why: the system has no CUDA toolkit/cuDNN installed (no sudo).
# onnxruntime-gpu 1.32 needs cuBLAS 13 + cuDNN 9, which exist as
# user-local pip packages in the comfy-cli venv.
CUDA_LIB_BASE="${CUDA_LIB_BASE:-/home/rbilal/.local/share/comfy-cli/lib/python3.10/site-packages/nvidia}"
export LD_LIBRARY_PATH="$CUDA_LIB_BASE/cu13/lib:$CUDA_LIB_BASE/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# GPU0 = RTX 2080 Super (cc 7.5). P100s (cc 6.0) are NOT supported by
# onnxruntime 1.32's CUDA EP — pin device 0 (override to test other GPUs).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
