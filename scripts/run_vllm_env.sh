#!/usr/bin/env bash
# Activate the vLLM runtime env. Can be sourced to export variables into the
# current shell, or executed to run a Python command inside the configured env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO}/.venv-bench"
CUDA_PKG="${VENV}/lib/python3.10/site-packages/nvidia/cu13"

export CUDA_HOME="${CUDA_PKG}"
export PATH="${CUDA_PKG}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PKG}/lib:${VENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
# TP/multiproc workers can hit torch inductor JIT on 2nd+ request; nvcc in WSL
# may be non-executable on /mnt/d paths, so disable dynamo compile entirely.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  cd "${REPO}"
else
  cd "${REPO}"
  exec "${VENV}/bin/python" "$@"
fi
