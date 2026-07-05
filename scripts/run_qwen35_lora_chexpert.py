#!/usr/bin/env python3
"""
Qwen3.5 LoRA CheXpert training (cls head or JSON SFT) on Qwen2-matched splits.

Uses HuggingFace + peft (not vLLM). Same split JSONs as CCA/CBM qwen35_2b_qwen2subset runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_PY = sys.executable


def _run(cmd: list[str]) -> None:
    print({"run": " ".join(str(c) for c in cmd)})
    subprocess.run(cmd, cwd=str(_REPO), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cls", "sft"], default="cls")
    parser.add_argument("--variant", choices=["2b", "4b"], default="2b")
    parser.add_argument(
        "--reuse_qwen2_splits",
        action="store_true",
        help="Use Qwen2 split paths (data/processed/splits) for fair comparison.",
    )
    parser.add_argument(
        "--splits_dir",
        default="",
        help="Override split directory (default: qwen35_2b_qwen2subset or data/processed/splits).",
    )
    parser.add_argument(
        "--protocol",
        default="qwen35_2b_qwen2subset",
    )
    parser.add_argument("--run_id", default="")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Cap train/val/test at 500 rows.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke else ""
    if args.splits_dir:
        splits = Path(args.splits_dir)
    elif args.reuse_qwen2_splits:
        splits = Path("data/processed/splits")
        if args.protocol == "qwen35_2b_qwen2subset":
            args.protocol = "default"
        if not args.run_id:
            args.run_id = f"qwen35_{args.variant}_lora_r16_qwen2_splits"
    else:
        splits = Path("data/processed/splits/qwen35_2b_qwen2subset")
    if args.smoke and not args.reuse_qwen2_splits and str(splits).endswith("_qwen2subset"):
        splits = Path(str(args.splits_dir) + "_smoke")
        if not (splits / "train_rows.json").is_file():
            splits = Path(args.splits_dir)

    train_json = splits / "train_rows.json"
    val_json = splits / "val_rows.json"
    test_json = splits / "test_rows.json"
    if not train_json.is_file():
        raise FileNotFoundError(f"Missing {train_json}; run CCA align/splits first.")

    model_id = f"qwen35_{args.variant}_lora_r16" if args.mode == "cls" else f"qwen35_{args.variant}_lora_r16_sft"
    run_id = args.run_id or f"qwen35_{args.variant}_lora_r16{suffix}_qwen2subset"
    protocol = f"{args.protocol}_smoke" if args.smoke and args.protocol == "qwen35_2b_qwen2subset" else args.protocol

    script = "train_qwen35_lora_cls.py" if args.mode == "cls" else "train_qwen35_lora_sft.py"
    cmd = [
        _PY,
        str(_SCRIPTS / script),
        "--variant",
        args.variant,
        "--model_id",
        model_id,
        "--protocol",
        protocol,
        "--run_id",
        run_id,
        "--train_rows_json",
        str(train_json),
        "--val_rows_json",
        str(val_json),
        "--test_rows_json",
        str(test_json),
        "--image_root",
        args.image_root,
        "--gpu_id",
        str(args.gpu_id),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--lora_rank",
        str(args.lora_rank),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--no_download",
    ]
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.smoke:
        cmd.extend(["--max_train_samples", "500", "--max_val_samples", "500", "--max_test_samples", "500"])

    _run(cmd)
    print({"status": "done", "model_id": model_id, "protocol": protocol, "run_id": run_id})


if __name__ == "__main__":
    main()
