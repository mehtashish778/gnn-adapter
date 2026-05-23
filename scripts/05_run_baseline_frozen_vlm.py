#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from common_multilabel import (
    f1_from_counts,
    load_per_class_thresholds,
    masked_macro_f1,
    probabilistic_metrics,
    subset_accuracy_masked_lists,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry


def evaluate(rows, thresholds):
    c = len(thresholds)
    tp = [0] * c
    fp = [0] * c
    fn = [0] * c
    all_probs = []
    all_y = []
    all_m = []
    for row in rows:
        probs = row["x_probs"]
        y = row["y_true"]
        m = row["y_mask"]
        all_probs.append(probs)
        all_y.append(y)
        all_m.append(m)
        for i in range(c):
            if m[i] == 0:
                continue
            pred = 1 if probs[i] >= thresholds[i] else 0
            if pred == 1 and y[i] == 1:
                tp[i] += 1
            elif pred == 1 and y[i] == 0:
                fp[i] += 1
            elif pred == 0 and y[i] == 1:
                fn[i] += 1
    per_class_f1 = [f1_from_counts(tp[i], fp[i], fn[i]) for i in range(c)]
    macro_f1 = sum(per_class_f1) / c
    subset_acc, subset_n = subset_accuracy_masked_lists(all_probs, all_y, all_m, thresholds)
    return {"macro_f1": macro_f1, "subset_accuracy": subset_acc, "subset_n_examples": subset_n, "per_class_f1": per_class_f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen VLM multi-label baseline.")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--skip_val", action="store_true", help="Only evaluate test_rows (cross-site).")
    args = parser.parse_args()

    val_rows = []
    if not args.skip_val:
        with Path(args.val_rows_json).open("r", encoding="utf-8") as f:
            val_rows = json.load(f)["rows"]
    with Path(args.test_rows_json).open("r", encoding="utf-8") as f:
        test_rows = json.load(f)["rows"]

    c = len(test_rows[0]["x_probs"])
    thresholds = [args.threshold] * c
    val_metrics = evaluate(val_rows, thresholds) if val_rows else None
    test_metrics = evaluate(test_rows, thresholds)

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or "vlm_zeroshot",
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/vlm_zeroshot",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    te_probs = torch.tensor([r["x_probs"] for r in test_rows], dtype=torch.float32)
    te_y = torch.tensor([r["y_true"] for r in test_rows], dtype=torch.float32)
    te_m = torch.tensor([r["y_mask"] for r in test_rows], dtype=torch.float32)
    test_pm = probabilistic_metrics(te_probs, te_y, te_m)
    thr_list = load_per_class_thresholds(
        Path("data/processed/experiments/thresholds/per_class_thresholds.json")
    )

    metrics = {
        "variant": "vlm_zeroshot",
        "thresholds": thresholds,
        "test_macro_f1@0.5": test_metrics["macro_f1"],
        "test_macro_auroc": test_pm["macro_auroc"],
        "test_macro_auprc": test_pm["macro_auprc"],
        "test_macro_ece": test_pm["macro_ece"],
        "test_macro_brier": test_pm["macro_brier"],
        "trainable_params": 0,
    }
    if args.protocol:
        metrics["cross_site"] = True
        metrics["protocol"] = args.protocol
    if thr_list and len(thr_list) == c:
        metrics["test_macro_f1@per_class_thr"] = masked_macro_f1(
            te_probs, te_y, te_m, threshold=thr_list
        )
    if val_metrics is not None:
        metrics["val"] = val_metrics
        metrics["val_macro_f1@0.5"] = val_metrics["macro_f1"]
    metrics["test"] = test_metrics

    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "test_predictions.json",
        {"probs": te_probs.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )
    if val_rows:
        va_probs = torch.tensor([r["x_probs"] for r in val_rows], dtype=torch.float32)
        va_y = torch.tensor([r["y_true"] for r in val_rows], dtype=torch.float32)
        va_m = torch.tensor([r["y_mask"] for r in val_rows], dtype=torch.float32)
        write_json(
            out_dir / "val_predictions.json",
            {"probs": va_probs.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
        )
    if args.model_id and args.protocol:
        reg_metrics = {
            "test_macro_f1@0.5": test_metrics["macro_f1"],
            "test_macro_auroc": test_pm["macro_auroc"],
        }
        if val_metrics is not None:
            reg_metrics["val_macro_f1@0.5"] = val_metrics["macro_f1"]
        update_run_registry(
            model_id=args.model_id,
            protocol=args.protocol,
            run_dir=out_dir,
            metrics=reg_metrics,
            hparams={"threshold": args.threshold},
        )
    out = {"test_macro_f1@0.5": test_metrics["macro_f1"], "test_macro_auroc": test_pm["macro_auroc"]}
    if val_metrics is not None:
        out["val_macro_f1@0.5"] = val_metrics["macro_f1"]
    print(out)


if __name__ == "__main__":
    main()
