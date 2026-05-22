#!/usr/bin/env python3
"""Patch metrics.json trainable_params for cls runs saved after eval reload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import require_cuda_device, write_json
from qwen2vl_lora_common import (
    DEFAULT_MODEL_ROOT,
    count_cls_trainable_params,
    ensure_model_snapshot,
    load_base_qwen_model,
)
from train_qwen2vl_lora_cls import Qwen2VLClassifier


def patch_run_dir(run_dir: Path, model_dir: Path, device: str) -> int:
    from peft import PeftModel

    adapter_dir = run_dir / "adapter"
    head_path = run_dir / "classifier_head.pt"
    metrics_path = run_dir / "metrics.json"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter: {adapter_dir}")
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing head: {head_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    model_dir = ensure_model_snapshot(model_dir)
    base = load_base_qwen_model(model_dir, device, gradient_checkpointing=False)
    backbone = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    hidden_size = backbone.config.hidden_size
    hparams = json.loads(metrics_path.read_text(encoding="utf-8")).get("hparams", {})
    num_labels = len(hparams.get("labels", [])) or 7
    model = Qwen2VLClassifier(backbone, hidden_size, num_labels).to(device)
    model.head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    trainable = count_cls_trainable_params(model)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    old = metrics.get("trainable_params")
    metrics["trainable_params"] = trainable
    write_json(metrics_path, metrics)
    print({"run_dir": str(run_dir), "old_trainable_params": old, "trainable_params": trainable})
    return trainable


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path, help="Experiment run dirs with adapter/ and metrics.json")
    p.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument("--gpu_id", type=int, default=0, help="CUDA device index (ignored if --device is set)")
    p.add_argument("--device", default=None, help="Override device, e.g. cuda:1")
    args = p.parse_args()
    device = args.device or require_cuda_device(args.gpu_id)
    for run_dir in args.run_dirs:
        patch_run_dir(run_dir.resolve(), args.model_dir, device)


if __name__ == "__main__":
    main()
