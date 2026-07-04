#!/usr/bin/env python3
"""
Score a trained Qwen3.5 LoRA run (classification head or generative JSON) on val/test splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    load_per_class_thresholds,
    load_rows,
    masked_macro_f1,
    masked_subset_accuracy,
    probabilistic_metrics,
    require_cuda_device,
    set_seed,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from qwen35_common import (
    CLS_PROMPT,
    DEFAULT_VARIANT,
    build_user_messages,
    default_lora_cls_model_id,
    ensure_qwen35_snapshot,
    generate_probs_from_rows,
    load_lora_model,
    load_processor,
    model_root_for_variant,
    normalize_variant,
    open_image,
    pool_last_token_hidden,
    prepare_inputs,
    qwen_hidden_size,
)


class Qwen35Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, num_labels, dtype=torch.float32)

    def forward(self, inputs: dict) -> torch.Tensor:
        pooled = pool_last_token_hidden(self.backbone, inputs)
        return self.head(pooled)


def score_cls(model, head, processor, rows, image_root, device, batch_size):
    model.eval()
    head.eval()
    probs_chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            images = [open_image(image_root, r) for r in batch]
            if len(batch) == 1:
                msgs = build_user_messages(images[0], CLS_PROMPT)
                inputs = prepare_inputs(processor, msgs, images, device)
            else:
                texts = [
                    processor.apply_chat_template(
                        build_user_messages(img, CLS_PROMPT),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for img in images
                ]
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            logits = head(pool_last_token_hidden(model, inputs))
            probs = torch.nan_to_num(torch.sigmoid(logits.float()), nan=0.5).clamp(0.0, 1.0)
            probs_chunks.append(probs.cpu())
    return torch.cat(probs_chunks, dim=0)


def score_gen(model, processor, rows, image_root, device, batch_size):
    with torch.no_grad():
        return generate_probs_from_rows(model, processor, rows, image_root, device, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="", help="Experiment run directory with adapter/")
    parser.add_argument("--ckpt_dir", default="", help="Alias for --run_dir (checkpoint directory).")
    parser.add_argument(
        "--variant",
        choices=["2b", "4b"],
        default=DEFAULT_VARIANT,
        help="Qwen3.5 model size: 2b or 4b.",
    )
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--mode", choices=["cls", "gen"], default="cls")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Fail if local HF cache is incomplete (do not fetch from Hub).",
    )
    parser.add_argument(
        "--checkpoint_run_dir",
        default="",
        help="CheXpert-trained run with adapter/ (cross-site: load weights from here).",
    )
    parser.add_argument(
        "--out_run_id",
        default="",
        help="Run id for writing metrics (defaults to --run_id or crosssite_eval).",
    )
    parser.add_argument("--skip_val", action="store_true", help="Only score test_rows (cross-site).")
    args = parser.parse_args()

    variant = normalize_variant(args.variant)
    model_id = args.model_id or default_lora_cls_model_id(variant)
    model_path = Path(args.model_path) if args.model_path else model_root_for_variant(variant)

    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    ckpt_dir = None
    if args.checkpoint_run_dir:
        ckpt_dir = Path(args.checkpoint_run_dir)
    elif args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
    elif args.run_dir:
        ckpt_dir = Path(args.run_dir)
    if ckpt_dir is None:
        ckpt_dir = resolve_experiment_dir(
            out_dir=None,
            model_id=model_id,
            protocol="default",
            run_id=args.run_id or None,
            default_legacy_out_dir=f"data/processed/experiments/qwen35_{variant}_lora_r16",
        )
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint run dir not found: {ckpt_dir}")

    out_run_id = args.out_run_id or args.run_id or "crosssite_eval"
    run_dir = resolve_experiment_dir(
        out_dir=None,
        model_id=model_id,
        protocol=args.protocol,
        run_id=out_run_id,
        default_legacy_out_dir=f"data/processed/experiments/qwen35_{variant}_lora_r16",
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ckpt_dir / "run_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        args.mode = meta.get("mode", args.mode)
        if meta.get("variant"):
            variant = normalize_variant(meta["variant"])

    adapter_dir = ckpt_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter at {adapter_dir}")

    val_rows = []
    if not args.skip_val:
        val_rows = load_rows(Path(args.val_rows_json))
    test_rows = load_rows(Path(args.test_rows_json))
    image_root = Path(args.image_root)
    model_dir = ensure_qwen35_snapshot(
        model_path,
        variant=variant,
        allow_download=not args.no_download,
    )
    processor = load_processor(model_dir, local_files_only=True)

    parse_failures = 0
    val_f1 = None
    if args.mode == "cls":
        backbone = load_lora_model(
            model_dir,
            adapter_dir,
            device,
            causal_lm=False,
            variant=variant,
            allow_download=not args.no_download,
        )
        head_path = ckpt_dir / "classifier_head.pt"
        if not head_path.exists():
            raise FileNotFoundError(f"Missing {head_path} for cls mode")
        hidden_size = qwen_hidden_size(backbone)
        num_labels = len(test_rows[0]["y_true"])
        head = nn.Linear(hidden_size, num_labels, dtype=torch.float32).to(device=device)
        head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
        val_prob = None
        if val_rows:
            val_prob = score_cls(backbone, head, processor, val_rows, image_root, device, args.batch_size)
        test_prob = score_cls(backbone, head, processor, test_rows, image_root, device, args.batch_size)
    else:
        model = load_lora_model(
            model_dir,
            adapter_dir,
            device,
            causal_lm=True,
            variant=variant,
            allow_download=not args.no_download,
        )
        val_prob, pf_v = None, 0
        if val_rows:
            val_prob, pf_v = score_gen(model, processor, val_rows, image_root, device, args.batch_size)
        test_prob, pf_t = score_gen(model, processor, test_rows, image_root, device, args.batch_size)
        parse_failures = pf_v + pf_t

    te_y = torch.tensor([r["y_true"] for r in test_rows], dtype=torch.float32)
    te_m = torch.tensor([r["y_mask"] for r in test_rows], dtype=torch.float32)
    test_f1 = masked_macro_f1(test_prob, te_y, te_m, threshold=0.5)
    test_pm = probabilistic_metrics(test_prob, te_y, te_m)

    thr_path = Path("data/processed/experiments/thresholds/per_class_thresholds.json")
    thr_list = load_per_class_thresholds(thr_path)

    metrics: dict = {
        "test_macro_f1@0.5": test_f1,
        "test_subset_accuracy@0.5": masked_subset_accuracy(test_prob, te_y, te_m, threshold=0.5),
        "test_macro_auroc": test_pm["macro_auroc"],
        "test_macro_auprc": test_pm["macro_auprc"],
        "test_macro_ece": test_pm["macro_ece"],
        "test_macro_brier": test_pm["macro_brier"],
        "scored_mode": args.mode,
        "parse_failures": parse_failures,
        "qwen35_variant": variant,
        "cross_site": True,
        "chexpert_run_dir": str(ckpt_dir),
    }
    if thr_list and len(thr_list) == te_y.shape[1]:
        metrics["test_macro_f1@per_class_thr"] = masked_macro_f1(
            test_prob, te_y, te_m, threshold=thr_list
        )
    ckpt_metrics_path = ckpt_dir / "metrics.json"
    if ckpt_metrics_path.exists():
        with ckpt_metrics_path.open("r", encoding="utf-8") as f:
            ckpt_m = json.load(f)
        if ckpt_m.get("trainable_params") is not None:
            metrics["trainable_params"] = ckpt_m["trainable_params"]
    if val_rows and val_prob is not None:
        va_y = torch.tensor([r["y_true"] for r in val_rows], dtype=torch.float32)
        va_m = torch.tensor([r["y_mask"] for r in val_rows], dtype=torch.float32)
        val_f1 = masked_macro_f1(val_prob, va_y, va_m, threshold=0.5)
        val_pm = probabilistic_metrics(val_prob, va_y, va_m)
        metrics.update(
            {
                "val_macro_f1@0.5": val_f1,
                "val_subset_accuracy@0.5": masked_subset_accuracy(val_prob, va_y, va_m, threshold=0.5),
                "val_macro_auroc": val_pm["macro_auroc"],
                "val_macro_auprc": val_pm["macro_auprc"],
                "val_macro_ece": val_pm["macro_ece"],
                "val_macro_brier": val_pm["macro_brier"],
            }
        )
        write_json(
            run_dir / "val_predictions.json",
            {"probs": val_prob.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
        )
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )

    if args.model_id and args.protocol:
        reg_metrics = {
            "test_macro_f1@0.5": test_f1,
            "test_macro_auroc": test_pm["macro_auroc"],
        }
        if val_f1 is not None:
            reg_metrics["val_macro_f1@0.5"] = val_f1
        update_run_registry(
            model_id=model_id,
            protocol=args.protocol,
            run_dir=run_dir,
            metrics=reg_metrics,
            hparams={"mode": args.mode, "variant": variant},
        )

    print({"run_dir": str(run_dir), "mode": args.mode, "test_macro_f1@0.5": test_f1, "variant": variant})


if __name__ == "__main__":
    main()
