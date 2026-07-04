#!/usr/bin/env python3
"""Launch 2-GPU parallel frozen Qwen3.5 vLLM scoring (half the dataset per GPU)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_ENV = _REPO / "scripts" / "run_vllm_env.sh"
_SCORE = _REPO / "scripts" / "04_score_frozen_qwen35_vllm_batch.py"


def _worker_cmd(
    *,
    gpu_id: int,
    shard_idx: int,
    variant: str,
    max_new_tokens: int,
    request_batch_size: int,
    gpu_memory_utilization: float,
    max_samples: int,
    extra: list[str],
) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "bash",
        str(_ENV),
        str(_SCORE),
        "--variant",
        variant,
        "--num_shards",
        "2",
        "--shard_idx",
        str(shard_idx),
        "--gpu_id",
        "0",
        "--max_new_tokens",
        str(max_new_tokens),
        "--request_batch_size",
        str(request_batch_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]
    if max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])
    cmd.extend(extra)
    return env, cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["2b", "4b"], default="2b")
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--request_batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_samples", type=int, default=0)
    args, extra = parser.parse_known_args()

    env0, cmd0 = _worker_cmd(
        gpu_id=args.gpu0,
        shard_idx=0,
        variant=args.variant,
        max_new_tokens=args.max_new_tokens,
        request_batch_size=args.request_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_samples=args.max_samples,
        extra=extra,
    )
    env1, cmd1 = _worker_cmd(
        gpu_id=args.gpu1,
        shard_idx=1,
        variant=args.variant,
        max_new_tokens=args.max_new_tokens,
        request_batch_size=args.request_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_samples=args.max_samples,
        extra=extra,
    )

    print({"launch_gpu0": cmd0, "cuda_visible_devices": args.gpu0})
    print({"launch_gpu1": cmd1, "cuda_visible_devices": args.gpu1})

    p0 = subprocess.Popen(cmd0, cwd=str(_REPO), env=env0)
    p1 = subprocess.Popen(cmd1, cwd=str(_REPO), env=env1)
    rc0 = p0.wait()
    rc1 = p1.wait()
    if rc0 != 0 or rc1 != 0:
        raise SystemExit(f"Worker exit codes: gpu0={rc0}, gpu1={rc1}")
    print({"status": "both_workers_finished"})


if __name__ == "__main__":
    main()
