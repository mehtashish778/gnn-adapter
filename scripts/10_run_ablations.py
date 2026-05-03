#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import write_json


def main():
    parser = argparse.ArgumentParser(description="Collect ablation metrics into one table.")
    parser.add_argument("--out_csv", default="data/processed/experiments/ablations/ablation_table.csv")
    args = parser.parse_args()

    rows = []
    metric_files = [
        ("baseline_frozen_vlm", Path("data/processed/experiments/baseline_frozen_vlm/metrics.json")),
        (
            "baseline_mlp",
            Path("data/processed/experiments/vlm_mlp/default/repro_full_20260503/metrics.json"),
        ),
        (
            "gnn_adapter",
            Path("data/processed/experiments/gnn07_label_residual/default/repro_full_20260503/metrics.json"),
        ),
        ("final_eval", Path("data/processed/experiments/final_eval/test_metrics.json")),
    ]
    for name, p in metric_files:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append({"experiment": name, "metrics": payload})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("experiment,macro_f1_hint\n")
        for r in rows:
            m = r["metrics"]
            macro = m.get("macro_f1", m.get("test_macro_f1@0.5", ""))
            f.write(f"{r['experiment']},{macro}\n")
    write_json(Path("data/processed/experiments/ablations/ablation_notes.json"), {"rows": rows})
    print({"num_rows": len(rows)})


if __name__ == "__main__":
    main()
