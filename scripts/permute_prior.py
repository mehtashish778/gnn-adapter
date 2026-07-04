#!/usr/bin/env python3
"""Permute concept prior rows for ablation control (destroys semantic structure)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common_multilabel import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with Path(args.in_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    mat = np.array(data["matrix"], dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(mat.shape[0])
    mat_p = mat[perm][:, perm]
    out = dict(data)
    out["matrix"] = mat_p.tolist()
    out["permutation_seed"] = args.seed
    out["source"] = str(data.get("source", "unknown")) + "_permuted"
    write_json(Path(args.out_json), out)
    print({"wrote": args.out_json, "seed": args.seed})


if __name__ == "__main__":
    main()
