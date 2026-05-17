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
    with torch.no_grad():
        te_parts = []
        for start in range(0, data.te_patch.shape[0], args.batch_size):
            te_parts.append(model(data.te_patch[start : start + args.batch_size].to(device)))
        te_out = torch.cat(te_parts, dim=0)
        te_f1 = masked_macro_f1(
            torch.sigmoid(te_out), data.te_y.to(device), data.te_m.to(device)
        )

    out_dir = resolve_experiment_dir(
        model_id=args.model_id or "qformer_adapter",
        protocol=args.protocol,
        run_id=args.run_id,
        default_legacy_out_dir="data/processed/experiments/qformer_adapter",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"val_macro_f1@0.5": float(best_f1), "test_macro_f1@0.5": float(te_f1)}
    write_json(out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")
    update_run_registry(args.model_id or "qformer_adapter", args.protocol, out_dir, metrics, {})
    print(metrics)


if __name__ == "__main__":
    main()
