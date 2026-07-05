#!/usr/bin/env python3
"""
CheXpert pipeline: Qwen3.5-2B frozen VLM scoring (vLLM) → align → splits → CCA train.

Uses vLLM dual-GPU scoring (not HuggingFace transformers). Outputs are kept separate
from legacy Qwen2-VL artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_PY = sys.executable
_VLLM_LAUNCHER = _SCRIPTS / "run_frozen_qwen35_vllm_dual_gpu.py"


def _run(cmd: list[str]) -> None:
    print({"run": " ".join(str(c) for c in cmd)})
    subprocess.run(cmd, cwd=str(_REPO), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical_json",
        default="data/processed/multilabel/canonical_labels.json",
    )
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument(
        "--vlm_dir",
        default="data/outputs_vlm_qwen35_2b_chexpert",
        help="vLLM JSONL shard output directory.",
    )
    parser.add_argument(
        "--aligned_json",
        default="data/processed/multilabel/aligned_vlm_targets_qwen35_chexpert.json",
    )
    parser.add_argument(
        "--splits_dir",
        default="data/processed/splits/qwen35_chexpert",
    )
    parser.add_argument("--variant", choices=["2b", "4b"], default="2b")
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    parser.add_argument("--request_batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = full CheXpert train set.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick subset run (~500 samples; sets --max_samples if unset).",
    )
    parser.add_argument("--skip_vlm", action="store_true", help="Skip vLLM scoring if shards exist.")
    parser.add_argument("--skip_align", action="store_true")
    parser.add_argument("--skip_splits", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="After splits, evaluate frozen VLM baseline (05_run_baseline_frozen_vlm.py).",
    )
    parser.add_argument(
        "--baseline_run_id",
        default="",
        help="Run id for frozen baseline metrics (default: vlm_zeroshot_qwen35_{variant}_qwen2_splits).",
    )
    parser.add_argument("--model_id", default="cca")
    parser.add_argument(
        "--protocol",
        default="default",
        help="CCA protocol / CLIP patch-cache namespace. Use e.g. qwen35_smoke for subset runs.",
    )
    parser.add_argument("--run_id", default="cca_qwen35_vllm_2b")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU for CCA training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--same_as_qwen2",
        action="store_true",
        help="Score/train on the same ~62k CheXpert subset as Qwen2-VL CCA (from aligned_vlm_targets.json).",
    )
    parser.add_argument(
        "--reuse_qwen2_splits",
        action="store_true",
        help="Keep Qwen2 train/val/test paths; inject Qwen3.5 VLM scores (fair comparison).",
    )
    parser.add_argument(
        "--reference_splits_dir",
        default="data/processed/splits",
        help="Reference splits when --reuse_qwen2_splits is set.",
    )
    args, extra = parser.parse_known_args()

    if args.smoke:
        args.max_samples = 500 if args.max_samples == 0 else args.max_samples

    if args.same_as_qwen2:
        subset_canonical = _REPO / "data/processed/multilabel/canonical_labels_qwen2_subset.json"
        _run(
            [
                _PY,
                str(_SCRIPTS / "build_canonical_qwen2_subset.py"),
                "--out_json",
                str(subset_canonical),
            ]
        )
        args.canonical_json = str(subset_canonical)
        suffix = "_smoke" if args.smoke else ""
        if args.vlm_dir == "data/outputs_vlm_qwen35_2b_chexpert":
            args.vlm_dir = f"data/outputs_vlm_qwen35_2b_chexpert_qwen2subset{suffix}"
        if args.aligned_json == "data/processed/multilabel/aligned_vlm_targets_qwen35_chexpert.json":
            args.aligned_json = (
                f"data/processed/multilabel/aligned_vlm_targets_qwen35_2b_qwen2subset{suffix}.json"
            )
        if args.splits_dir == "data/processed/splits/qwen35_chexpert":
            args.splits_dir = f"data/processed/splits/qwen35_2b_qwen2subset{suffix}"
        if args.run_id == "cca_qwen35_vllm_2b":
            args.run_id = f"cca_qwen35_vllm_2b_qwen2subset{suffix}"
        if args.smoke:
            args.protocol = "qwen35_2b_qwen2subset_smoke"
        elif args.protocol == "default":
            # Separate CLIP cache namespace: Qwen3.5 align may drop a few rows vs Qwen2 splits.
            args.protocol = "qwen35_2b_qwen2subset"

    if args.reuse_qwen2_splits:
        suffix = "_smoke" if args.smoke else ""
        if args.splits_dir == "data/processed/splits/qwen35_chexpert":
            args.splits_dir = f"data/processed/splits/qwen35_qwen2_splits{suffix}"
        if args.run_id == "cca_qwen35_vllm_2b":
            args.run_id = f"cca_qwen35_vllm_2b_qwen2_splits{suffix}"
        if args.smoke:
            args.protocol = "qwen35_qwen2_splits_smoke"
        elif args.protocol in ("default", "qwen35_2b_qwen2subset"):
            # Same paths as Qwen2 default splits → reuse CLIP patch cache namespace.
            args.protocol = "default"

    vlm_dir = Path(args.vlm_dir)
    aligned = Path(args.aligned_json)
    splits = Path(args.splits_dir)
    train_rows = splits / "train_rows.json"
    val_rows = splits / "val_rows.json"
    test_rows = splits / "test_rows.json"

    if not args.skip_vlm:
        has_shards = vlm_dir.is_dir() and any(vlm_dir.glob("*.jsonl"))
        if has_shards and args.max_samples == 0:
            print({"skip_vlm": "shards already present; use --skip_vlm to silence"})
        else:
            cmd = [
                _PY,
                str(_VLLM_LAUNCHER),
                "--variant",
                args.variant,
                "--request_batch_size",
                str(args.request_batch_size),
                "--gpu_memory_utilization",
                str(args.gpu_memory_utilization),
                "--gpu0",
                str(args.gpu0),
                "--gpu1",
                str(args.gpu1),
                "--canonical_json",
                args.canonical_json,
                "--image_root",
                args.image_root,
                "--out_dir",
                str(vlm_dir),
            ]
            if args.max_samples > 0:
                cmd.extend(["--max_samples", str(args.max_samples)])
            cmd.extend(extra)
            _run(cmd)

    if not args.skip_align:
        _run(
            [
                _PY,
                str(_SCRIPTS / "02_align_vlm_outputs.py"),
                "--canonical_json",
                args.canonical_json,
                "--vlm_dir",
                str(vlm_dir),
                "--out_json",
                str(aligned),
            ]
        )

    if not args.skip_splits:
        splits.mkdir(parents=True, exist_ok=True)
        if args.reuse_qwen2_splits:
            _run(
                [
                    _PY,
                    str(_SCRIPTS / "remap_splits_vlm_scores.py"),
                    "--reference_splits_dir",
                    args.reference_splits_dir,
                    "--aligned_json",
                    str(aligned),
                    "--out_dir",
                    str(splits),
                ]
            )
        else:
            _run(
                [
                    _PY,
                    str(_SCRIPTS / "03_make_multilabel_splits.py"),
                    "--aligned_json",
                    str(aligned),
                    "--out_dir",
                    str(splits),
                    "--seed",
                    str(args.seed),
                ]
            )

    if not args.skip_train:
        if not train_rows.is_file():
            raise FileNotFoundError(f"Missing {train_rows}; run align + splits first.")
        _run(
            [
                _PY,
                str(_SCRIPTS / "14_train_cca.py"),
                "--model_id",
                args.model_id,
                "--protocol",
                args.protocol,
                "--run_id",
                args.run_id,
                "--train_rows_json",
                str(train_rows),
                "--val_rows_json",
                str(val_rows),
                "--test_rows_json",
                str(test_rows),
                "--image_root",
                args.image_root,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--gpu_id",
                str(args.gpu_id),
                "--seed",
                str(args.seed),
            ]
        )

    if args.run_baseline:
        if not test_rows.is_file():
            raise FileNotFoundError(f"Missing {test_rows}; run align + splits first.")
        baseline_run_id = args.baseline_run_id or (
            f"vlm_zeroshot_qwen35_{args.variant}_qwen2_splits{'_smoke' if args.smoke else ''}"
        )
        _run(
            [
                _PY,
                str(_SCRIPTS / "05_run_baseline_frozen_vlm.py"),
                "--val_rows_json",
                str(val_rows),
                "--test_rows_json",
                str(test_rows),
                "--model_id",
                "vlm_zeroshot",
                "--protocol",
                "default",
                "--run_id",
                baseline_run_id,
            ]
        )

    print(
        {
            "status": "done",
            "vlm_dir": str(vlm_dir),
            "aligned_json": str(aligned),
            "splits_dir": str(splits),
            "cca_run_id": args.run_id,
        }
    )


if __name__ == "__main__":
    main()
