#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import f1_from_counts, write_json


def class_f1(y_true, y_mask, probs, t, idx):
    tp = fp = fn = 0
    for y, m, p in zip(y_true, y_mask, probs):
        if m[idx] == 0:
            continue
        pred = 1 if p[idx] >= t else 0
        if pred == 1 and y[idx] == 1:
            tp += 1
        elif pred == 1 and y[idx] == 0:
            fp += 1
        elif pred == 0 and y[idx] == 1:
            fn += 1
    return f1_from_counts(tp, fp, fn)


def main():
    parser = argparse.ArgumentParser(description="Tune per-class thresholds on validation predictions.")
    parser.add_argument("--val_predictions_json", default="data/processed/experiments/gnn_adapter/val_predictions.json")
    parser.add_argument("--out_json", default="data/processed/experiments/thresholds/per_class_thresholds.json")
    args = parser.parse_args()

    with Path(args.val_predictions_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    probs = data["probs"]
    y_true = data["y_true"]
    y_mask = data["y_mask"]
    c = len(probs[0])
    thresholds = []
    best_f1 = []
    for i in range(c):
        best_t = 0.5
        best = -1.0
        for k in range(5, 96, 5):
            t = k / 100.0
            f1 = class_f1(y_true, y_mask, probs, t, i)
            if f1 > best:
                best = f1
                best_t = t
        thresholds.append(best_t)
        best_f1.append(best)

    write_json(Path(args.out_json), {"thresholds": thresholds, "val_class_f1": best_f1})
    print({"thresholds": thresholds})


if __name__ == "__main__":
    main()
