#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import CSV_TO_VLM, normalize_path, parse_uncertain, read_csv_rows, write_json


def main():
    parser = argparse.ArgumentParser(description="Build canonical multi-label targets from CheXpert CSV.")
    parser.add_argument("--train_csv", default="data/raw/train.csv")
    parser.add_argument("--out_json", default="data/processed/multilabel/canonical_labels.json")
    parser.add_argument(
        "--uncertain_policy",
        choices=["u_ones", "u_zeros", "ignore"],
        default="u_zeros",
    )
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.train_csv))
    out_rows = []
    prevalence = {v: 0 for v in CSV_TO_VLM.values()}
    valid_counts = {v: 0 for v in CSV_TO_VLM.values()}
    missing_paths = 0

    for row in rows:
        path = normalize_path(row.get("Path", ""))
        if not path:
            missing_paths += 1
            continue
        y = {}
        m = {}
        for csv_col, vlm_label in CSV_TO_VLM.items():
            label_value, mask_value = parse_uncertain(row.get(csv_col), args.uncertain_policy)
            y[vlm_label] = label_value
            m[vlm_label] = mask_value
            prevalence[vlm_label] += label_value
            valid_counts[vlm_label] += mask_value
        out_rows.append(
            {
                "path": path,
                "patient_id": path.split("/")[2] if len(path.split("/")) > 2 else "unknown",
                "labels": y,
                "mask": m,
            }
        )

    payload = {
        "meta": {
            "source": args.train_csv,
            "uncertain_policy": args.uncertain_policy,
            "num_rows": len(out_rows),
            "missing_paths": missing_paths,
            "label_order": list(CSV_TO_VLM.values()),
        },
        "rows": out_rows,
    }
    write_json(Path(args.out_json), payload)

    report = {
        "num_rows": len(out_rows),
        "missing_paths": missing_paths,
        "prevalence_sum": prevalence,
        "valid_label_counts": valid_counts,
    }
    write_json(Path("data/processed/multilabel/canonical_labels_report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
