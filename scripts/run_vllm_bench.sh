#!/usr/bin/env bash
# Back-compat alias: run benchmark via vLLM env wrapper.
exec "$(dirname "$0")/run_vllm_env.sh" scripts/benchmark_vllm_vs_hf_qwen35.py "$@"
