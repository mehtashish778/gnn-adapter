#!/usr/bin/env bash
# Activate vLLM runtime env (CUDA libs + sampler flags) and run a Python command.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${REPO}/.venv-bench"
CUDA_PKG="${VENV}/lib/python3.10/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_PKG}"
export PATH="${CUDA_PKG}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PKG}/lib:${VENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "${REPO}"
exec "${VENV}/bin/python" "$@"
