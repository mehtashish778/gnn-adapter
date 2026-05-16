#!/usr/bin/env python3
"""Dump rows JSON ({rows: [...]}) to val_predictions-compatible format using x_probs (frozen VLM)."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows_json", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    payload = json.loads(Path(args.rows_json).read_text(encoding="utf-8"))
    rows = payload["rows"]
    out = {
        "probs": [r["x_probs"] for r in rows],
        "y_true": [r["y_true"] for r in rows],
        "y_mask": [r["y_mask"] for r in rows],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print({"rows": len(rows), "out": args.out_json})


if __name__ == "__main__":
    main()
