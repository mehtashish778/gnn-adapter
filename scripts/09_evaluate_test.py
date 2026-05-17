#!/usr/bin/env python3
"""CLI: evaluate test predictions with frozen thresholds (delegates to calibration module)."""

import argparse
from pathlib import Path

from calibration import calibrated_eval_from_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate test predictions with frozen per-class thresholds.")
    parser.add_argument(
        "--test_predictions_json",
        default="data/processed/experiments/gnn07_label_residual/default/repro_full_20260503/test_predictions.json",
    )
    parser.add_argument(
        "--thresholds_json",
        default="data/processed/experiments/thresholds/per_class_thresholds.json",
    )
    parser.add_argument(
        "--out_json",
        default="data/processed/experiments/final_eval/test_metrics.json",
    )
    args = parser.parse_args()

    out = calibrated_eval_from_files(
        args.test_predictions_json,
        args.thresholds_json,
        out_json=args.out_json,
    )
    print(
        {
            "macro_f1": out["macro_f1"],
            "subset_accuracy": out["subset_accuracy"],
            "subset_n_examples": out["subset_n_examples"],
        }
    )


if __name__ == "__main__":
    main()
