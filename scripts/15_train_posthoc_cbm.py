#!/usr/bin/env python3
"""Train post-hoc Concept Bottleneck Model on frozen VLM [logits; probs]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    build_standard_argparser,
    load_rows,
    require_cuda_device,
    set_seed,
    to_label_tensors,
    to_vlm_training_batch,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from models.architectures.posthoc_cbm import PostHocCBM


def main():
    parser = build_standard_argparser("Train post-hoc CBM baseline.")
    parser.add_argument("--num_concepts", type=int, default=30)
    args = parser.parse_args()
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    tr = load_rows(Path(args.train_rows_json))
    va = load_rows(Path(args.val_rows_json))
    te = load_rows(Path(args.test_rows_json))
    xtr, ytr, mtr = to_vlm_training_batch(tr)
    xva, yva, mva = to_vlm_training_batch(va)
    xte, yte, mte = to_vlm_training_batch(te)

    c = ytr.shape[1]
    model = PostHocCBM(xtr.shape[1], args.num_concepts, c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos = (ytr * mtr).sum(0)
    neg = ((1 - ytr) * mtr).sum(0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).to(device)

    loader = DataLoader(TensorDataset(xtr, ytr, mtr), batch_size=64, shuffle=True)
    for _ in range(args.epochs):
        model.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight, reduction="none"
            )
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        va_out, _ = model(xva.to(device))
        te_out, _ = model(xte.to(device))
        from common_multilabel import masked_macro_f1, probabilistic_metrics

        va_prob = torch.sigmoid(va_out)
        te_prob = torch.sigmoid(te_out)
        va_f1 = masked_macro_f1(va_prob, yva.to(device), mva.to(device))
        te_f1 = masked_macro_f1(te_prob, yte.to(device), mte.to(device))
        va_pm = probabilistic_metrics(va_prob, yva, mva)
        te_pm = probabilistic_metrics(te_prob, yte, mte)

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or "cbm_posthoc",
        protocol=args.protocol or "default",
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/cbm_posthoc",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "variant": "cbm_posthoc",
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "val_macro_f1@0.5": float(va_f1),
        "test_macro_f1@0.5": float(te_f1),
        "val_macro_auroc": va_pm["macro_auroc"],
        "test_macro_auroc": te_pm["macro_auroc"],
        "val_macro_auprc": va_pm["macro_auprc"],
        "test_macro_auprc": te_pm["macro_auprc"],
        "val_macro_ece": va_pm["macro_ece"],
        "test_macro_ece": te_pm["macro_ece"],
        "val_macro_brier": va_pm["macro_brier"],
        "test_macro_brier": te_pm["macro_brier"],
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "test_predictions.json",
        {"probs": te_prob.cpu().tolist(), "y_true": yte.tolist(), "y_mask": mte.tolist()},
    )
    write_json(
        out_dir / "val_predictions.json",
        {"probs": va_prob.cpu().tolist(), "y_true": yva.tolist(), "y_mask": mva.tolist()},
    )
    torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")
    update_run_registry(
        model_id=args.model_id or "cbm_posthoc",
        protocol=args.protocol or "default",
        run_dir=out_dir,
        metrics={"val_macro_f1@0.5": float(va_f1), "test_macro_f1@0.5": float(te_f1)},
        hparams={"epochs": args.epochs, "lr": args.lr, "num_concepts": args.num_concepts},
    )
    print(metrics)


if __name__ == "__main__":
    main()
