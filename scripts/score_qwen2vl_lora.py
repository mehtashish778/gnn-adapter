#!/usr/bin/env python3
"""
Score a trained Qwen2-VL LoRA run (classification head or generative JSON) on val/test splits.
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
    load_rows,
    masked_macro_f1,
    masked_subset_accuracy,
    probabilistic_metrics,
    require_cuda_device,
    set_seed,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from qwen2vl_lora_common import (
    CLS_PROMPT,
    DEFAULT_MODEL_ROOT,
    JSON_PROMPT,
    VLM_LABELS,
    build_user_messages,
    extract_json_dict,
    load_lora_model,
    load_processor,
    open_image,
    pool_last_token_hidden,
    prepare_inputs,
    ensure_model_snapshot,
)


class Qwen2VLClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, num_labels)

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
            pooled = pool_last_token_hidden(model, inputs).to(dtype=head.weight.dtype)
            logits = head(pooled)
            probs_chunks.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(probs_chunks, dim=0)


def score_gen(model, processor, rows, image_root, device, batch_size):
    model.eval()
    probs_list = []
    parse_failures = 0
    with torch.no_grad():
        for row in rows:
            img = open_image(image_root, row)
            msgs = build_user_messages(img, JSON_PROMPT)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            try:
                d = extract_json_dict(decoded)
                probs_list.append([d[lbl] for lbl in VLM_LABELS])
            except (ValueError, json.JSONDecodeError):
                parse_failures += 1
                probs_list.append([0.5] * len(VLM_LABELS))
    return torch.tensor(probs_list, dtype=torch.float32), parse_failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="", help="Experiment run directory with adapter/")
    parser.add_argument("--model_id", default="qwen2vl_lora_r16")
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--mode", choices=["cls", "gen"], default="cls")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT))
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
    args = parser.parse_args()

    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else resolve_experiment_dir(
        out_dir=None,
        model_id=args.model_id,
        protocol=args.protocol,
        run_id=args.run_id or None,
        default_legacy_out_dir=f"data/processed/experiments/{args.model_id}",
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        args.mode = meta.get("mode", args.mode)

    adapter_dir = run_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter at {adapter_dir}")

    val_rows = load_rows(Path(args.val_rows_json))
    test_rows = load_rows(Path(args.test_rows_json))
    image_root = Path(args.image_root)
    model_dir = ensure_model_snapshot(
        Path(args.model_path),
        allow_download=not args.no_download,
    )
    processor = load_processor(model_dir, local_files_only=True)

    parse_failures = 0
    if args.mode == "cls":
        backbone = load_lora_model(model_dir, adapter_dir, device, causal_lm=False)
        head_path = run_dir / "classifier_head.pt"
        if not head_path.exists():
            raise FileNotFoundError(f"Missing {head_path} for cls mode")
        hidden_size = backbone.config.hidden_size
        num_labels = len(val_rows[0]["y_true"])
        head_dtype = next(backbone.parameters()).dtype
        head = nn.Linear(hidden_size, num_labels).to(device=device, dtype=head_dtype)
        head.load_state_dict(torch.load(head_path, map_location=device))
        val_prob = score_cls(backbone, head, processor, val_rows, image_root, device, args.batch_size)
        test_prob = score_cls(backbone, head, processor, test_rows, image_root, device, args.batch_size)
    else:
        model = load_lora_model(model_dir, adapter_dir, device, causal_lm=True)
        val_prob, pf_v = score_gen(model, processor, val_rows, image_root, device, args.batch_size)
        test_prob, pf_t = score_gen(model, processor, test_rows, image_root, device, args.batch_size)
        parse_failures = pf_v + pf_t

    va_y = torch.tensor([r["y_true"] for r in val_rows], dtype=torch.float32)
    va_m = torch.tensor([r["y_mask"] for r in val_rows], dtype=torch.float32)
    te_y = torch.tensor([r["y_true"] for r in test_rows], dtype=torch.float32)
    te_m = torch.tensor([r["y_mask"] for r in test_rows], dtype=torch.float32)

    val_f1 = masked_macro_f1(val_prob, va_y, va_m, threshold=0.5)
    test_f1 = masked_macro_f1(test_prob, te_y, te_m, threshold=0.5)
    val_pm = probabilistic_metrics(val_prob, va_y, va_m)
    test_pm = probabilistic_metrics(test_prob, te_y, te_m)

    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    metrics.update(
        {
            "val_macro_f1@0.5": val_f1,
            "test_macro_f1@0.5": test_f1,
            "val_subset_accuracy@0.5": masked_subset_accuracy(val_prob, va_y, va_m, threshold=0.5),
            "test_subset_accuracy@0.5": masked_subset_accuracy(test_prob, te_y, te_m, threshold=0.5),
            "val_macro_auroc": val_pm["macro_auroc"],
            "val_macro_auprc": val_pm["macro_auprc"],
            "val_macro_ece": val_pm["macro_ece"],
            "val_macro_brier": val_pm["macro_brier"],
            "test_macro_auroc": test_pm["macro_auroc"],
            "test_macro_auprc": test_pm["macro_auprc"],
            "test_macro_ece": test_pm["macro_ece"],
            "test_macro_brier": test_pm["macro_brier"],
            "scored_mode": args.mode,
            "parse_failures": parse_failures,
        }
    )
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "val_predictions.json",
        {"probs": val_prob.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
    )
    write_json(
        run_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )

    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id,
            protocol=args.protocol,
            run_dir=run_dir,
            metrics={
                "val_macro_f1@0.5": val_f1,
                "test_macro_f1@0.5": test_f1,
                "test_macro_auroc": test_pm["macro_auroc"],
            },
            hparams={"mode": args.mode},
        )

    print({"run_dir": str(run_dir), "mode": args.mode, "test_macro_f1@0.5": test_f1})


if __name__ == "__main__":
    main()
