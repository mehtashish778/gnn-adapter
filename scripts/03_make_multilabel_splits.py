#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import train_val_test_split, write_json


def main():
    parser = argparse.ArgumentParser(description="Create reproducible patient-level split manifests.")
    parser.add_argument("--aligned_json", default="data/processed/multilabel/aligned_vlm_targets.json")
    parser.add_argument("--out_dir", default="data/processed/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.aligned_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload["rows"]
    train, val, test = train_val_test_split(rows, key="patient_id", seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "train_rows.json", {"rows": train})
    write_json(out_dir / "val_rows.json", {"rows": val})
    write_json(out_dir / "test_rows.json", {"rows": test})

    report = {
        "seed": args.seed,
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    write_json(out_dir / "split_manifest_v1.json", report)
    print(report)


if __name__ == "__main__":
    main()
