#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import VLM_LABELS, write_json


def main():
    parser = argparse.ArgumentParser(description="Build co-error label graph from train split.")
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--out_dir", default="data/processed/graph")
    args = parser.parse_args()

    with Path(args.train_rows_json).open("r", encoding="utf-8") as f:
        rows = json.load(f)["rows"]

    c = len(VLM_LABELS)
    label_to_idx = {l: i for i, l in enumerate(VLM_LABELS)}
    m = [[0.0 for _ in range(c)] for _ in range(c)]

    for row in rows:
        probs = row["x_probs"]
        y = row["y_true"]
        mask = row["y_mask"]
        present = [i for i, v in enumerate(y) if v == 1 and mask[i] == 1]
        absent = [i for i, v in enumerate(y) if v == 0 and mask[i] == 1]
        if not present:
            continue
        absent_sorted = sorted(absent, key=lambda i: probs[i], reverse=True)[: args.top_k]
        present_sorted = sorted(present, key=lambda i: probs[i])[: args.top_k]
        for i in present_sorted:
            for j in absent_sorted:
                if i != j:
                    m[i][j] += 1.0

    w = []
    for i in range(c):
        row_sum = sum(m[i])
        if row_sum == 0:
            w.append([0.0] * c)
            continue
        w.append([v / row_sum for v in m[i]])

    edge_index = [[], []]
    edge_weight = []
    for i in range(c):
        candidates = [(j, w[i][j]) for j in range(c) if j != i and w[i][j] >= args.tau]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for j, wij in candidates[: args.top_k]:
            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_weight.append(wij)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "label_to_idx.json", label_to_idx)
    write_json(out_dir / "coerror_matrix.json", m)
    write_json(out_dir / "coerror_matrix_normalized.json", w)
    write_json(out_dir / "edge_index.json", edge_index)
    write_json(out_dir / "edge_weight.json", edge_weight)
    write_json(
        out_dir / "graph_build_report.json",
        {
            "num_nodes": c,
            "num_edges": len(edge_weight),
            "top_k": args.top_k,
            "tau": args.tau,
        },
    )
    print({"num_edges": len(edge_weight), "num_nodes": c})


if __name__ == "__main__":
    main()
