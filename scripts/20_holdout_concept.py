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
from common_multilabel import masked_macro_f1, probabilistic_metrics, require_cuda_device, set_seed, write_json
from models.architectures.cca import CCAModel


@torch.no_grad()
def eval_with_holdout(model, data, device, args, holdout_p: int):
    model.eval()
    parts = []
    n = min(4096, data.va_patch.shape[0])
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
    y = data.va_y[:n].to(device)
    m = data.va_m[:n].to(device)
    f1 = masked_macro_f1(prob, y, m, threshold=0.5)
    pm = probabilistic_metrics(prob, y, m)
    return {
        "val_macro_f1@0.5": float(f1),
        "val_macro_auroc": pm["macro_auroc"],
        "val_macro_auprc": pm["macro_auprc"],
        "val_macro_brier": pm["macro_brier"],
    }


def main():
    base = build_argparser()
    parser = argparse.ArgumentParser(parents=[base], conflict_handler="resolve")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout_primitive", type=int, default=-1,
                        help="If >=0 hold out a single primitive; if -1 sweep all primitives.")
    parser.add_argument("--summary_json", default="",
                        help="Optional explicit JSON path to write the summary report.")
    args = parser.parse_args()
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)
    data = load_cca_data(args, device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    hparams = ckpt.get("adapter_hparams", {})
    num_primitives = int(hparams.get("num_primitives", args.num_primitives))
    query_dim = int(hparams.get("query_dim", args.query_dim))
    n_heads = int(hparams.get("n_heads", args.n_heads))
    n_cross = int(hparams.get("n_cross_attn_layers", args.n_cross_attn_layers))
    n_self = int(hparams.get("n_self_attn_layers", args.n_self_attn_layers))
    alpha = float(hparams.get("alpha", args.alpha))
    use_gate_M = bool(hparams.get("use_gate_M", args.use_gate_M))

    model = CCAModel(
        patch_dim=data.patch_dim,
        query_dim=query_dim,
        num_primitives=num_primitives,
        num_findings=data.c,
        n_heads=n_heads,
        n_cross_attn_layers=n_cross,
        n_self_attn_layers=n_self,
        alpha=alpha,
        use_gate_M=use_gate_M,
    ).to(device)
    state = ckpt.get("adapter_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    base = eval_with_holdout(model, data, device, args, holdout_p=-1)

    def _drops(base_metrics, holdout_metrics):
        return {k.replace("val_", "drop_"): base_metrics[k] - holdout_metrics[k] for k in base_metrics}

    if args.holdout_primitive >= 0:
        hold = eval_with_holdout(model, data, device, args, holdout_p=args.holdout_primitive)
        report = {
            "checkpoint": args.checkpoint,
            "use_gate_M": use_gate_M,
            "num_primitives": num_primitives,
            "holdout_primitive": args.holdout_primitive,
            "full": base,
            "holdout": hold,
            "drops": _drops(base, hold),
        }
    else:
        per_p = []
        for p in range(num_primitives):
            hold = eval_with_holdout(model, data, device, args, holdout_p=p)
            per_p.append({"p": p, "holdout": hold, "drops": _drops(base, hold)})
        per_p.sort(key=lambda r: -r["drops"]["drop_macro_auroc"])
        top5 = per_p[:5]
        report = {
            "checkpoint": args.checkpoint,
            "use_gate_M": use_gate_M,
            "num_primitives": num_primitives,
            "full": base,
            "max_drop_auroc": per_p[0]["drops"]["drop_macro_auroc"] if per_p else 0.0,
            "mean_drop_auroc": float(sum(r["drops"]["drop_macro_auroc"] for r in per_p) / max(1, len(per_p))),
            "max_drop_f1": max(r["drops"]["drop_macro_f1@0.5"] for r in per_p) if per_p else 0.0,
            "p_with_max_drop_auroc": per_p[0]["p"] if per_p else None,
            "top5_by_auroc_drop": [{"p": r["p"], "drops": r["drops"]} for r in top5],
            "per_primitive": per_p,
        }
    print({k: v for k, v in report.items() if k not in {"per_primitive", "top5_by_auroc_drop"}})
    out_path = Path(args.summary_json) if args.summary_json else (
        Path(args.out_dir) / "holdout_concept.json" if args.out_dir else None
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, report)


if __name__ == "__main__":
    main()
