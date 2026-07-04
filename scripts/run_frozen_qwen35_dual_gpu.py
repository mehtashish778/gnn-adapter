#!/usr/bin/env python3
"""Launch 2-GPU parallel frozen Qwen3.5 scoring (half the dataset per GPU)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCORE = _REPO / "scripts" / "04_score_frozen_qwen35_batch.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["2b", "4b"], default="2b")
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    args, extra = parser.parse_known_args()

    base = [
        sys.executable,
        str(_SCORE),
        "--variant",
        args.variant,
        "--num_shards",
        "2",
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    if args.max_samples > 0:
        base.extend(["--max_samples", str(args.max_samples)])
    if args.use_flash_attn:
        base.append("--use_flash_attn")
    base.extend(extra)

    cmd0 = base + ["--gpu_id", str(args.gpu0), "--shard_idx", "0"]
    cmd1 = base + ["--gpu_id", str(args.gpu1), "--shard_idx", "1"]

    print({"launch_gpu0": cmd0})
    print({"launch_gpu1": cmd1})

    p0 = subprocess.Popen(cmd0, cwd=str(_REPO))
    p1 = subprocess.Popen(cmd1, cwd=str(_REPO))
    rc0 = p0.wait()
    rc1 = p1.wait()
    if rc0 != 0 or rc1 != 0:
        raise SystemExit(f"Worker exit codes: gpu0={rc0}, gpu1={rc1}")
    print({"status": "both_workers_finished"})


if __name__ == "__main__":
    main()
