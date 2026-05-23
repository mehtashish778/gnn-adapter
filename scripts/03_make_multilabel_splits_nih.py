#!/usr/bin/env python3
"""Create NIH test-only split manifests (cross-site; no NIH training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common_multilabel import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical_json",
        default="data/processed/multilabel/nih/canonical_labels.json",
    )
    parser.add_argument("--out_dir", default="data/processed/splits/nih")
    parser.add_argument("--val_shim_size", type=int, default=10, help="Tiny val shard for scripts that require val.")
    args = parser.parse_args()

    with Path(args.canonical_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "test_rows.json", {"rows": rows})
    val_shim = rows[: min(args.val_shim_size, len(rows))]
    write_json(out_dir / "val_rows.json", {"rows": val_shim})

    report = {
        "protocol": "nih",
        "test": len(rows),
        "val_shim": len(val_shim),
    }
    write_json(out_dir / "split_manifest_nih.json", report)
    print(report)


if __name__ == "__main__":
    main()
