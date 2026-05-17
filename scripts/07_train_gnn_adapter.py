#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import (
    build_adj,
    load_per_class_thresholds,
    load_rows,
    masked_bce_with_logits,
    masked_macro_f1,
    masked_subset_accuracy,
    require_cuda_device,
    set_seed,
    to_label_tensors,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from models.architectures.gnn07_label_residual import ResidualLabelGNNModel


def main():
    parser = argparse.ArgumentParser(description="Train residual label-graph adapter (torch required).")
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--calib_rows_json", default=None, help="Optional calibration rows JSON (threshold tuning).")
    parser.add_argument("--edge_index_json", default="data/processed/graph/edge_index.json")
    parser.add_argument("--edge_weight_json", default="data/processed/graph/edge_weight.json")
    parser.add_argument("--per_class_thresholds_json", default="data/processed/experiments/thresholds/per_class_thresholds.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Floor LR when using cosine schedule.")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--pos_weight_max", type=float, default=100.0)
    parser.add_argument(
        "--lr_scheduler",
        choices=("none", "cosine", "plateau"),
        default="cosine",
        help="cosine: CosineAnnealingLR; plateau: ReduceLROnPlateau on val_bce.",
    )
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=6)
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear LR warmup (cosine mode only).")
    parser.add_argument(
        "--best_metric",
        choices=("val_bce", "val_macro_f1_thr", "val_macro_f1_05"),
        default="val_bce",
        help="Checkpoint selection: val_bce=min; val_macro_f1_thr=max (needs per_class_thresholds); val_macro_f1_05=max.",
    )
    parser.add_argument("--early_stop_patience", type=int, default=18, help="Stop if best_metric does not improve for N epochs (0=disabled).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--resume_from", default="", help="Optional checkpoint path to initialize model weights.")
    parser.add_argument("--gpu_id", type=int, default=1, help="Single GPU index to use.")
    args = parser.parse_args()

    try:
        import math

        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("This script requires PyTorch.") from exc

    device = require_cuda_device(args.gpu_id)

    set_seed(args.seed)

    train_rows = load_rows(Path(args.train_rows_json))
    val_rows = load_rows(Path(args.val_rows_json))
    test_rows = load_rows(Path(args.test_rows_json))
    calib_rows = load_rows(Path(args.calib_rows_json)) if args.calib_rows_json else None
    n_train, n_val, n_test = len(train_rows), len(val_rows), len(test_rows)
    n_calib = len(calib_rows) if calib_rows is not None else 0
    print({"dataset_sizes": {"train": n_train, "val": n_val, "calib": n_calib, "test": n_test}})

    with Path(args.edge_index_json).open("r", encoding="utf-8") as f:
        edge_index = json.load(f)
    with Path(args.edge_weight_json).open("r", encoding="utf-8") as f:
        edge_weight = json.load(f)

    c = len(train_rows[0]["x_probs"])
    adj = build_adj(c, edge_index, edge_weight).to(device)
    model = ResidualLabelGNNModel(hidden_dim=args.hidden_dim, alpha=args.alpha)
    model = model.to(device)
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    plateau_sched = None
    if args.lr_scheduler == "plateau":
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    tr_logits, tr_probs, tr_y, tr_m = to_label_tensors(train_rows)
    va_logits, va_probs, va_y, va_m = to_label_tensors(val_rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(test_rows)
    if calib_rows is not None:
        ca_logits, ca_probs, ca_y, ca_m = to_label_tensors(calib_rows)
    tr_logits = tr_logits.to(device)
    tr_probs = tr_probs.to(device)
    tr_y = tr_y.to(device)
    tr_m = tr_m.to(device)
    va_logits = va_logits.to(device)
    va_probs = va_probs.to(device)
    va_y = va_y.to(device)
    va_m = va_m.to(device)
    te_logits = te_logits.to(device)
    te_probs = te_probs.to(device)
    te_y = te_y.to(device)
    te_m = te_m.to(device)
    if calib_rows is not None:
        ca_logits = ca_logits.to(device)
        ca_probs = ca_probs.to(device)
        ca_y = ca_y.to(device)
        ca_m = ca_m.to(device)

    pos = (tr_y * tr_m).sum(dim=0)
    neg = ((1 - tr_y) * tr_m).sum(dim=0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=args.pos_weight_max)

    thr_path = Path(args.per_class_thresholds_json)
    thr_list = load_per_class_thresholds(thr_path)
    if thr_list is not None and len(thr_list) != c:
        thr_list = None

    best_metric = args.best_metric
    if best_metric == "val_macro_f1_thr" and thr_list is None:
        print("Warning: --best_metric val_macro_f1_thr but no valid per-class thresholds; using val_bce.")
        best_metric = "val_bce"

    best = {"score": None, "state_dict": None}
    history = []
    epochs_no_improve = 0
    base_lr = args.lr

    def lr_at_epoch(epoch: int) -> float:
        if args.lr_scheduler != "cosine":
            return base_lr
        w = max(0, args.warmup_epochs)
        if w > 0 and epoch <= w:
            return base_lr * (epoch / w)
        t = (epoch - w) / max(1, args.epochs - w)
        return args.min_lr + (base_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    def is_better(metric_name, val_bce, f1_05, f1_thr):
        if metric_name == "val_bce":
            cur = val_bce
            if best["score"] is None:
                return True
            return cur < best["score"]
        if metric_name == "val_macro_f1_05":
            cur = f1_05
            if best["score"] is None:
                return True
            return cur > best["score"]
        cur = f1_thr
        if best["score"] is None:
            return True
        return cur > best["score"]

    for epoch in range(1, args.epochs + 1):
        if args.lr_scheduler == "cosine":
            lr_now = lr_at_epoch(epoch)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

        model.train()
        out = model(tr_logits, tr_probs, adj)
        raw_loss = F.binary_cross_entropy_with_logits(out, tr_y, pos_weight=pos_weight, reduction="none")
        loss = (raw_loss * tr_m).sum() / tr_m.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(va_logits, va_probs, adj)
            val_prob = torch.sigmoid(val_out)
            val_bce = float(masked_bce_with_logits(val_out, va_y, va_m, pos_weight).item())
            val_f1_05 = masked_macro_f1(val_prob, va_y, va_m, threshold=0.5)
            val_f1_thr = (
                masked_macro_f1(val_prob, va_y, va_m, threshold=thr_list) if thr_list is not None else val_f1_05
            )

        row = {
            "epoch": epoch,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": float(loss.item()),
            "val_bce": val_bce,
            "val_macro_f1@0.5": val_f1_05,
            "val_macro_f1@thr": val_f1_thr,
        }
        history.append(row)

        if plateau_sched is not None:
            plateau_sched.step(val_bce)

        cur_f1 = val_f1_thr if best_metric == "val_macro_f1_thr" else val_f1_05
        if is_better(best_metric, val_bce, val_f1_05, val_f1_thr):
            if best_metric == "val_bce":
                best["score"] = val_bce
            elif best_metric == "val_macro_f1_05":
                best["score"] = val_f1_05
            else:
                best["score"] = val_f1_thr
            best["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
            print({"early_stop": True, "epoch": epoch, "epochs_no_improve": epochs_no_improve})
            break

    if best["state_dict"] is None:
        raise RuntimeError("No checkpoint was saved; training may have failed.")

    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(va_logits, va_probs, adj))
        test_prob = torch.sigmoid(model(te_logits, te_probs, adj))
        if calib_rows is not None:
            calib_prob = torch.sigmoid(model(ca_logits, ca_probs, adj))
    val_f1 = masked_macro_f1(val_prob, va_y, va_m, threshold=0.5)
    test_f1 = masked_macro_f1(test_prob, te_y, te_m, threshold=0.5)
    if calib_rows is not None:
        calib_f1 = masked_macro_f1(calib_prob, ca_y, ca_m, threshold=0.5)
    val_f1_thr_eval = (
        masked_macro_f1(val_prob, va_y, va_m, threshold=thr_list) if thr_list is not None else val_f1
    )
    test_f1_thr_eval = (
        masked_macro_f1(test_prob, te_y, te_m, threshold=thr_list) if thr_list is not None else test_f1
    )
    if calib_rows is not None:
        calib_f1_thr_eval = (
            masked_macro_f1(calib_prob, ca_y, ca_m, threshold=thr_list) if thr_list is not None else calib_f1
        )

    val_sub = masked_subset_accuracy(val_prob, va_y, va_m, threshold=0.5)
    test_sub = masked_subset_accuracy(test_prob, te_y, te_m, threshold=0.5)
    calib_sub = None
    if calib_rows is not None:
        calib_sub = masked_subset_accuracy(calib_prob, ca_y, ca_m, threshold=0.5)
    val_sub_thr = masked_subset_accuracy(val_prob, va_y, va_m, threshold=thr_list) if thr_list is not None else val_sub
    test_sub_thr = masked_subset_accuracy(test_prob, te_y, te_m, threshold=thr_list) if thr_list is not None else test_sub
    calib_sub_thr = None
    if calib_rows is not None:
        calib_sub_thr = masked_subset_accuracy(calib_prob, ca_y, ca_m, threshold=thr_list) if thr_list is not None else calib_sub

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or None,
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/gnn_adapter",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state_dict"], out_dir / "best_checkpoint.pt")
    metrics_payload = {
        "dataset_sizes": {"train": n_train, "val": n_val, "calib": n_calib, "test": n_test},
        "best_metric": best_metric,
        "best_score": best["score"],
        "seed": args.seed,
        "hparams": {
            "epochs_ran": len(history),
            "lr": args.lr,
            "min_lr": args.min_lr,
            "hidden_dim": args.hidden_dim,
            "alpha": args.alpha,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "lr_scheduler": args.lr_scheduler,
            "pos_weight_max": args.pos_weight_max,
            "early_stop_patience": args.early_stop_patience,
        },
        "val_macro_f1@0.5": val_f1,
        "test_macro_f1@0.5": test_f1,
        "val_macro_f1@per_class_thr": val_f1_thr_eval,
        "test_macro_f1@per_class_thr": test_f1_thr_eval,
        "val_subset_accuracy@0.5": val_sub,
        "test_subset_accuracy@0.5": test_sub,
        "val_subset_accuracy@per_class_thr": val_sub_thr,
        "test_subset_accuracy@per_class_thr": test_sub_thr,
    }
    if calib_rows is not None:
        metrics_payload["calib_macro_f1@0.5"] = calib_f1
        metrics_payload["calib_macro_f1@per_class_thr"] = calib_f1_thr_eval
        metrics_payload["calib_subset_accuracy@0.5"] = calib_sub
        metrics_payload["calib_subset_accuracy@per_class_thr"] = calib_sub_thr
    write_json(out_dir / "metrics.json", metrics_payload)
    write_json(out_dir / "history.json", history)
    write_json(
        out_dir / "val_predictions.json",
        {"probs": val_prob.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
    )
    if calib_rows is not None:
        write_json(
            out_dir / "calib_predictions.json",
            {"probs": calib_prob.tolist(), "y_true": ca_y.tolist(), "y_mask": ca_m.tolist()},
        )
    write_json(
        out_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )
    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id,
            protocol=args.protocol,
            run_dir=out_dir,
            metrics={
                "val_macro_f1@0.5": val_f1,
                "test_macro_f1@0.5": test_f1,
                "val_macro_f1@per_class_thr": val_f1_thr_eval,
                "test_macro_f1@per_class_thr": test_f1_thr_eval,
                "val_subset_accuracy@0.5": val_sub,
                "test_subset_accuracy@0.5": test_sub,
                "val_subset_accuracy@per_class_thr": val_sub_thr,
                "test_subset_accuracy@per_class_thr": test_sub_thr,
            },
            hparams={
                "epochs": len(history),
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "alpha": args.alpha,
                "best_metric": best_metric,
            },
        )
    print(
        {
            "best_metric": best_metric,
            "best_score": best["score"],
            "val_macro_f1@0.5": val_f1,
            "test_macro_f1@0.5": test_f1,
            "test_subset_accuracy@0.5": test_sub,
            "test_subset_accuracy@per_class_thr": test_sub_thr,
            "val_macro_f1@per_class_thr": val_f1_thr_eval,
            "test_macro_f1@per_class_thr": test_f1_thr_eval,
        }
    )


if __name__ == "__main__":
    main()
