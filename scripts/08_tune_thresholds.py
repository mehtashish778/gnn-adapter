#!/usr/bin/env python3
"""CLI: tune per-class thresholds (delegates to calibration module)."""

import argparse
from pathlib import Path

from calibration import tune_thresholds_from_file


def main():
    parser = argparse.ArgumentParser(description="Tune per-class thresholds on validation predictions.")
    parser.add_argument(
        "--val_predictions_json",
        default="data/processed/experiments/gnn07_label_residual/default/repro_full_20260503/val_predictions.json",
    )
    parser.add_argument(
        "--out_json",
        default="data/processed/experiments/thresholds/per_class_thresholds.json",
    )
    parser.add_argument(
        "--method",
        choices=("per_class_grid", "isotonic", "temperature"),
        default="per_class_grid",
    )
    args = parser.parse_args()

    payload = tune_thresholds_from_file(
        args.val_predictions_json,
        args.out_json,
        method=args.method,
    )
    print({"thresholds": payload["thresholds"]})


if __name__ == "__main__":
    main()
