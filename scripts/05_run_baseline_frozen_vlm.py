#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import f1_from_counts, write_json
from model_registry import resolve_experiment_dir, update_run_registry


def evaluate(rows, thresholds):
    c = len(thresholds)
    tp = [0] * c
    fp = [0] * c
    fn = [0] * c
    for row in rows:
        probs = row["x_probs"]
        y = row["y_true"]
        m = row["y_mask"]
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
    return {"macro_f1": macro_f1, "per_class_f1": per_class_f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen VLM multi-label baseline.")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--run_id", default="")
    args = parser.parse_args()

    with Path(args.val_rows_json).open("r", encoding="utf-8") as f:
        val_rows = json.load(f)["rows"]
    with Path(args.test_rows_json).open("r", encoding="utf-8") as f:
        test_rows = json.load(f)["rows"]

    c = len(val_rows[0]["x_probs"])
    thresholds = [args.threshold] * c
    val_metrics = evaluate(val_rows, thresholds)
    test_metrics = evaluate(test_rows, thresholds)

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or None,
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/baseline_frozen_vlm",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", {"val": val_metrics, "test": test_metrics, "thresholds": thresholds})
    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id,
            protocol=args.protocol,
            run_dir=out_dir,
            metrics={
                "val_macro_f1": val_metrics["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            hparams={"threshold": args.threshold},
        )
    print({"val_macro_f1": val_metrics["macro_f1"], "test_macro_f1": test_metrics["macro_f1"]})


if __name__ == "__main__":
    main()
