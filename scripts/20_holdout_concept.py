#!/usr/bin/env python3
"""Held-out primitive ablation: force gate column off and measure F1 drop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cca_train_core import build_argparser, load_cca_data
from common_multilabel import masked_macro_f1, require_cuda_device, set_seed, write_json
from models.architectures.cca import CCAModel


@torch.no_grad()
def eval_with_holdout(model, data, device, args, holdout_p: int):
    model.eval()
    parts = []
    n = min(2048, data.va_patch.shape[0])
    for start in range(0, n, args.batch_size):
        patches = data.va_patch[start : start + args.batch_size].to(device)
        ll = data.va_logits[start : start + args.batch_size].to(device)
        pp = data.va_probs[start : start + args.batch_size].to(device)
        if holdout_p < 0:
            out, _, _ = model(patches, ll, pp, gumbel_tau=args.gumbel_tau_min)
        else:
            _, comp, _ = model._encode(patches)
            if model.gate is not None:
                m = model.gate.hard_gate().clone()
                if 0 <= holdout_p < m.shape[1]:
                    m[:, holdout_p] = 0.0
                out, _ = model.forward_from_comp_feats(comp, ll, pp, gate_M=m)
            else:
                out, _ = model.forward_from_comp_feats(comp, ll, pp, gate_M=None)
        parts.append(out)
    logits = torch.cat(parts, dim=0)
    prob = torch.sigmoid(logits)
    return masked_macro_f1(
        prob, data.va_y[:n].to(device), data.va_m[:n].to(device), threshold=0.5
    )


def main():
    base = build_argparser()
    parser = argparse.ArgumentParser(parents=[base], conflict_handler="resolve")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout_primitive", type=int, default=0)
    args = parser.parse_args()
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)
    data = load_cca_data(args, device)

    model = CCAModel(
        patch_dim=data.patch_dim,
        query_dim=args.query_dim,
        num_primitives=args.num_primitives,
        num_findings=data.c,
        n_heads=args.n_heads,
        n_cross_attn_layers=args.n_cross_attn_layers,
        n_self_attn_layers=args.n_self_attn_layers,
        alpha=args.alpha,
        use_gate_M=args.use_gate_M,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("adapter_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    f1_full = eval_with_holdout(model, data, device, args, holdout_p=-1)
    f1_hold = eval_with_holdout(model, data, device, args, holdout_p=args.holdout_primitive)
    out = {
        "holdout_primitive": args.holdout_primitive,
        "val_macro_f1_full": float(f1_full),
        "val_macro_f1_holdout": float(f1_hold),
        "val_f1_drop": float(f1_full - f1_hold),
    }
    print(out)
    if args.out_dir:
        write_json(Path(args.out_dir) / "holdout_concept.json", out)


if __name__ == "__main__":
    main()
