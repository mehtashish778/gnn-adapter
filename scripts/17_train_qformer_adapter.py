#!/usr/bin/env python3
"""Train Q-Former-style adapter on cached ViT patch tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cca_train_core import PatchRowDataset, build_argparser, load_cca_data
from common_multilabel import masked_bce_with_logits, masked_macro_f1, require_cuda_device, set_seed, write_json
from model_registry import resolve_experiment_dir, update_run_registry
from models.architectures.qformer_adapter import QFormerAdapter


def main():
    base = build_argparser()
    parser = argparse.ArgumentParser(parents=[base], conflict_handler="resolve", add_help=True)
    parser.add_argument("--num_queries", type=int, default=32)
    args = parser.parse_args()
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)
    data = load_cca_data(args, device)

    model = QFormerAdapter(
        patch_dim=data.patch_dim,
        query_dim=args.query_dim,
        num_queries=args.num_queries,
        num_labels=data.c,
        n_heads=args.n_heads,
        n_layers=max(1, args.n_cross_attn_layers),
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = (data.tr_y * data.tr_m).sum(0)
    neg = ((1 - data.tr_y) * data.tr_m).sum(0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=args.pos_weight_max).to(device)
    loader = DataLoader(
        PatchRowDataset(data.tr_patch, data.tr_logits, data.tr_probs, data.tr_y, data.tr_m),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for patches, _, _, yt, ym in loader:
            patches, yt, ym = patches.to(device), yt.to(device), ym.to(device)
            opt.zero_grad()
            out = model(patches)
            loss = masked_bce_with_logits(out, yt, ym, pos_weight)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = data.va_patch[:2048].to(device)
            out = model(vp)
            f1 = masked_macro_f1(torch.sigmoid(out), data.va_y[:2048].to(device), data.va_m[:2048].to(device))
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if args.early_stop_patience and epoch > args.early_stop_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    from common_multilabel import probabilistic_metrics

    with torch.no_grad():
        va_parts = []
        for start in range(0, data.va_patch.shape[0], args.batch_size):
            va_parts.append(model(data.va_patch[start : start + args.batch_size].to(device)))
        va_out = torch.cat(va_parts, dim=0)
        te_parts = []
        for start in range(0, data.te_patch.shape[0], args.batch_size):
            te_parts.append(model(data.te_patch[start : start + args.batch_size].to(device)))
        te_out = torch.cat(te_parts, dim=0)
        va_prob = torch.sigmoid(va_out)
        te_prob = torch.sigmoid(te_out)
        va_f1 = masked_macro_f1(va_prob, data.va_y.to(device), data.va_m.to(device))
        te_f1 = masked_macro_f1(te_prob, data.te_y.to(device), data.te_m.to(device))
        va_pm = probabilistic_metrics(va_prob, data.va_y, data.va_m)
        te_pm = probabilistic_metrics(te_prob, data.te_y, data.te_m)

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or "qformer_adapter",
        protocol=args.protocol or "default",
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/qformer_adapter",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "variant": "qformer_adapter",
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
        {"probs": te_prob.cpu().tolist(), "y_true": data.te_y.tolist(), "y_mask": data.te_m.tolist()},
    )
    write_json(
        out_dir / "val_predictions.json",
        {"probs": va_prob.cpu().tolist(), "y_true": data.va_y.tolist(), "y_mask": data.va_m.tolist()},
    )
    torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")
    update_run_registry(
        model_id=args.model_id or "qformer_adapter",
        protocol=args.protocol or "default",
        run_dir=out_dir,
        metrics={"val_macro_f1@0.5": float(best_f1), "test_macro_f1@0.5": float(te_f1)},
        hparams={"epochs": args.epochs, "lr": args.lr, "num_queries": args.num_queries},
    )
    print(metrics)


if __name__ == "__main__":
    main()
