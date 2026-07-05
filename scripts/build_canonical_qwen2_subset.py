#!/usr/bin/env python3
"""Build canonical JSON limited to paths present in an aligned VLM targets file (Qwen2 subset)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common_multilabel import normalize_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned_json",
        default="data/processed/multilabel/aligned_vlm_targets.json",
        help="Aligned rows defining the image subset (same pool as Qwen2 CCA/LoRA).",
    )
    parser.add_argument(
        "--canonical_json",
        default="data/processed/multilabel/canonical_labels.json",
    )
    parser.add_argument(
        "--out_json",
        default="data/processed/multilabel/canonical_labels_qwen2_subset.json",
    )
    args = parser.parse_args()

    aligned = json.loads(Path(args.aligned_json).read_text(encoding="utf-8"))
    canonical = json.loads(Path(args.canonical_json).read_text(encoding="utf-8"))
    want = {normalize_path(r["path"]) for r in aligned["rows"]}
    rows = [r for r in canonical["rows"] if normalize_path(r["path"]) in want]
    missing = len(want) - len(rows)
    payload = {
        "meta": {
            **canonical.get("meta", {}),
            "subset_of": args.canonical_json,
            "subset_from_aligned": args.aligned_json,
            "num_rows": len(rows),
            "missing_from_canonical": missing,
        },
        "rows": rows,
    }
    write_json(Path(args.out_json), payload)
    print(
        {
            "aligned_paths": len(want),
            "canonical_rows_written": len(rows),
            "missing_from_canonical": missing,
            "out_json": args.out_json,
        }
    )


if __name__ == "__main__":
    main()
