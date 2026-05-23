#!/usr/bin/env python3
"""Merge NIH canonical labels with aligned VLM scores into test_rows.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common_multilabel import VLM_LABELS, normalize_path, write_json


def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical_json",
        default="data/processed/multilabel/nih/canonical_labels.json",
    )
    parser.add_argument(
        "--aligned_json",
        default="data/processed/multilabel/nih/aligned_vlm_targets.json",
    )
    parser.add_argument(
        "--out_json",
        default="data/processed/splits/nih/test_rows.json",
    )
    args = parser.parse_args()

    with Path(args.canonical_json).open("r", encoding="utf-8") as f:
        canonical = json.load(f)
    label_map = {normalize_path(r["path"]): r for r in canonical["rows"]}

    with Path(args.aligned_json).open("r", encoding="utf-8") as f:
        aligned = json.load(f)
    aligned_by_path = {normalize_path(r["path"]): r for r in aligned["rows"]}

    merged = []
    missing_vlm = 0
    for path, target in label_map.items():
        if path not in aligned_by_path:
            missing_vlm += 1
            continue
        ar = aligned_by_path[path]
        merged.append(
            {
                "path": path,
                "image_id": ar.get("image_id", target.get("image_id")),
                "patient_id": target["patient_id"],
                "x_probs": ar["x_probs"],
                "x_logits": ar["x_logits"],
                "y_true": [int(target["labels"][lbl]) for lbl in VLM_LABELS],
                "y_mask": [int(target["mask"][lbl]) for lbl in VLM_LABELS],
            }
        )

    out = {
        "meta": {
            "canonical": args.canonical_json,
            "aligned": args.aligned_json,
            "num_rows": len(merged),
            "missing_vlm": missing_vlm,
        },
        "rows": merged,
    }
    write_json(Path(args.out_json), out)
    print(out["meta"])


if __name__ == "__main__":
    main()
