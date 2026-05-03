#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import f1_from_counts, subset_accuracy_masked_lists, write_json


def eval_with_thresholds(data, thresholds):
    probs = data["probs"]
    y_true = data["y_true"]
    y_mask = data["y_mask"]
    c = len(thresholds)
    per_class = []
    for i in range(c):
        tp = fp = fn = tn = 0
        for y, m, p in zip(y_true, y_mask, probs):
            if m[i] == 0:
                continue
            pred = 1 if p[i] >= thresholds[i] else 0
            gt = int(y[i])
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 1:
                fn += 1
            else:
                tn += 1
        f1 = f1_from_counts(tp, fp, fn)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        per_class.append({"f1": f1, "recall": rec, "precision": prec, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    macro_f1 = sum(x["f1"] for x in per_class) / c
    subset_acc, subset_n = subset_accuracy_masked_lists(probs, y_true, y_mask, thresholds)
    return {
        "macro_f1": macro_f1,
        "subset_accuracy": subset_acc,
        "subset_n_examples": subset_n,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate test predictions with frozen per-class thresholds.")
    parser.add_argument(
        "--test_predictions_json",
        default="data/processed/experiments/gnn07_label_residual/default/repro_full_20260503/test_predictions.json",
    )
    parser.add_argument("--thresholds_json", default="data/processed/experiments/thresholds/per_class_thresholds.json")
    parser.add_argument("--out_json", default="data/processed/experiments/final_eval/test_metrics.json")
    args = parser.parse_args()

    with Path(args.test_predictions_json).open("r", encoding="utf-8") as f:
        test_data = json.load(f)
    with Path(args.thresholds_json).open("r", encoding="utf-8") as f:
        thresholds = json.load(f)["thresholds"]

    out = eval_with_thresholds(test_data, thresholds)
    write_json(Path(args.out_json), out)
    print({"macro_f1": out["macro_f1"], "subset_accuracy": out["subset_accuracy"], "subset_n_examples": out["subset_n_examples"]})


if __name__ == "__main__":
    main()
