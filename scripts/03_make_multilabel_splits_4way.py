#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict
from pathlib import Path

from common_multilabel import write_json


def main():
    parser = argparse.ArgumentParser(description="Create reproducible 4-way patient-level splits.")
    parser.add_argument(
        "--aligned_json",
        default="data/processed/multilabel/aligned_vlm_targets.json",
        help="JSON produced by scripts/02_align_vlm_outputs.py.",
    )
    parser.add_argument("--out_dir", default="data/processed/splits_4way")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key", default="patient_id", help="Group rows by this field before splitting.")

    parser.add_argument("--train_fit_ratio", type=float, default=0.70)
    parser.add_argument("--calib_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    args = parser.parse_args()

    import json

    if abs((args.train_fit_ratio + args.calib_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            "Ratios must sum to 1.0; got "
            f"{args.train_fit_ratio}+{args.calib_ratio}+{args.val_ratio}+{args.test_ratio}="
            f"{args.train_fit_ratio + args.calib_ratio + args.val_ratio + args.test_ratio}"
        )

    with Path(args.aligned_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"]

    groups = defaultdict(list)
    for row in rows:
        groups[row[args.key]].append(row)

    ids = list(groups.keys())
    rnd = random.Random(args.seed)
    rnd.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train_fit_ratio)
    n_calib = int(n * args.calib_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_calib - n_val

    # Ensure non-empty splits if possible (mainly guards against tiny datasets).
    if n_test <= 0 and n > 3:
        n_test = 1
        n_train = max(0, n_train - 1)
    if n_calib <= 0 and n > 3:
        n_calib = 1
        n_train = max(0, n_train - 1)
    if n_val <= 0 and n > 3:
        n_val = 1
        n_train = max(0, n_train - 1)

    train_ids = set(ids[:n_train])
    calib_ids = set(ids[n_train : n_train + n_calib])
    val_ids = set(ids[n_train + n_calib : n_train + n_calib + n_val])
    test_ids = set(ids[n_train + n_calib + n_val :])

    train_fit, calib, val, test = [], [], [], []
    for k, group_rows in groups.items():
        if k in train_ids:
            train_fit.extend(group_rows)
        elif k in calib_ids:
            calib.extend(group_rows)
        elif k in val_ids:
            val.extend(group_rows)
        else:
            test.extend(group_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "train_fit_rows.json", {"rows": train_fit})
    write_json(out_dir / "calib_rows.json", {"rows": calib})
    write_json(out_dir / "val_rows.json", {"rows": val})
    write_json(out_dir / "test_rows.json", {"rows": test})

    report = {
        "seed": args.seed,
        "key": args.key,
        "total_rows": len(rows),
        "total_groups": n,
        "ratios": {
            "train_fit": args.train_fit_ratio,
            "calib": args.calib_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "sizes": {
            "train_fit": len(train_fit),
            "calib": len(calib),
            "val": len(val),
            "test": len(test),
        },
    }
    write_json(out_dir / "split_manifest_v1.json", report)
    print(report)


if __name__ == "__main__":
    main()

