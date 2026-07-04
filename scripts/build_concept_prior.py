#!/usr/bin/env python3
"""Build P×P concept prior matrix from train-label co-occurrence (RadGraph fallback)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common_multilabel import load_rows, write_json
from models.architectures.cca import DEFAULT_CONCEPT_PHRASES


def build_cooccurrence_prior(train_rows: list, num_primitives: int) -> np.ndarray:
    """Map first P CheXpert labels to primitives; symmetrize co-occurrence counts."""
    c = len(train_rows[0]["y_true"])
    p_use = min(num_primitives, c)
    counts = np.zeros((p_use, p_use), dtype=np.float64)
    for row in train_rows:
        y = np.array(row["y_true"], dtype=np.float32)
        m = np.array(row["y_mask"], dtype=np.float32)
        active = [i for i in range(p_use) if m[i] > 0.5 and y[i] > 0.5]
        for i in active:
            for j in active:
                counts[i, j] += 1.0
    mat = counts + counts.T
    np.fill_diagonal(mat, counts.diagonal())
    row_sum = mat.sum(axis=1, keepdims=True).clip(min=1.0)
    mat = mat / row_sum
    if p_use < num_primitives:
        full = np.eye(num_primitives, dtype=np.float32) * 0.1
        full[:p_use, :p_use] = mat.astype(np.float32)
        mat = full
    return mat.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--num_primitives", type=int, default=30)
    parser.add_argument("--out_json", default="data/processed/graph/radgraph_prior_P30.json")
    parser.add_argument("--radgraph_json", default="", help="Optional RadGraph matrix JSON to load instead.")
    args = parser.parse_args()

    if args.radgraph_json and Path(args.radgraph_json).exists():
        with Path(args.radgraph_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        mat = np.array(data.get("matrix") or data.get("prior"), dtype=np.float32)
        if mat.shape[0] != args.num_primitives:
            raise ValueError(f"RadGraph shape {mat.shape} != P={args.num_primitives}")
    else:
        rows = load_rows(Path(args.train_rows_json))
        mat = build_cooccurrence_prior(rows, args.num_primitives)
        print(f"Built co-occurrence prior from {len(rows)} rows (RadGraph file not provided).")

    out = {
        "matrix": mat.tolist(),
        "num_primitives": args.num_primitives,
        "concept_phrases": DEFAULT_CONCEPT_PHRASES[: args.num_primitives],
        "source": "cooccurrence_train_labels",
    }
    write_json(Path(args.out_json), out)
    print({"wrote": args.out_json, "shape": list(mat.shape)})


if __name__ == "__main__":
    main()
