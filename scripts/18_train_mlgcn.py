#!/usr/bin/env python3
"""Train simplified ML-GCN on label co-occurrence graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    build_standard_argparser,
    load_rows,
    masked_bce_with_logits,
    masked_macro_f1,
    require_cuda_device,
    set_seed,
    to_label_tensors,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from models.architectures.mlgcn import MLGCN


def build_label_adj(rows: list, num_labels: int) -> torch.Tensor:
    counts = np.zeros((num_labels, num_labels), dtype=np.float64)
    for row in rows:
        y = np.array(row["y_true"])
        m = np.array(row["y_mask"])
        active = [i for i in range(num_labels) if m[i] > 0.5 and y[i] > 0.5]
        for i in active:
            for j in active:
                counts[i, j] += 1.0
    adj = counts + counts.T
    np.fill_diagonal(adj, counts.diagonal())
    adj = adj / adj.sum(axis=1, keepdims=True).clip(min=1.0)
    return torch.tensor(adj, dtype=torch.float32)


def main():
    parser = build_standard_argparser("Train ML-GCN baseline.")
    args = parser.parse_args()
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    tr = load_rows(Path(args.train_rows_json))
    va = load_rows(Path(args.val_rows_json))
    te = load_rows(Path(args.test_rows_json))
    tr_logits, tr_probs, tr_y, tr_m = to_label_tensors(tr)
    va_logits, va_probs, va_y, va_m = to_label_tensors(va)
    te_logits, te_probs, te_y, te_m = to_label_tensors(te)

    c = tr_y.shape[1]
    model = MLGCN(c).to(device)
    model.set_adjacency(build_label_adj(tr, c).to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos = (tr_y * tr_m).sum(0)
    neg = ((1 - tr_y) * tr_m).sum(0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).to(device)
    loader = DataLoader(TensorDataset(tr_logits, tr_probs, tr_y, tr_m), batch_size=128, shuffle=True)

    for _ in range(args.epochs):
        model.train()
        for ll, pp, yt, ym in loader:
            ll, pp, yt, ym = ll.to(device), pp.to(device), yt.to(device), ym.to(device)
            opt.zero_grad()
            out = model(ll, pp)
            loss = masked_bce_with_logits(out, yt, ym, pos_weight)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        va_out = model(va_logits.to(device), va_probs.to(device))
        te_out = model(te_logits.to(device), te_probs.to(device))
        va_f1 = masked_macro_f1(torch.sigmoid(va_out), va_y.to(device), va_m.to(device))
        te_f1 = masked_macro_f1(torch.sigmoid(te_out), te_y.to(device), te_m.to(device))

    out_dir = resolve_experiment_dir(
        model_id=args.model_id or "mlgcn",
        protocol=args.protocol,
        run_id=args.run_id,
        default_legacy_out_dir="data/processed/experiments/mlgcn",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"val_macro_f1@0.5": float(va_f1), "test_macro_f1@0.5": float(te_f1)}
    write_json(out_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")
    update_run_registry(args.model_id or "mlgcn", args.protocol, out_dir, metrics, {})
    print(metrics)


if __name__ == "__main__":
    main()
