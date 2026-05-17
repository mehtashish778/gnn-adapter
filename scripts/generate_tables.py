#!/usr/bin/env python3
"""
Aggregate multi-seed parquet summaries into markdown and LaTeX table shells.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_METRIC_COLS = [
    "test_metrics_calibrated.macro_f1",
    "val_metrics_calibrated.macro_f1",
    "metrics.test_macro_f1@0.5",
    "metrics.val_macro_f1@0.5",
]


def find_metric_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mean_std(series: pd.Series) -> str:
    if series.empty:
        return "—"
    m = series.mean()
    s = series.std(ddof=0) if len(series) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"


def load_seeds_summary(repo: Path, model_id: str, protocol: str) -> Optional[pd.DataFrame]:
    base = repo / "data/processed/experiments" / model_id / protocol
    parquet_path = base / "seeds_summary.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    json_path = base / "seeds_summary.json"
    if json_path.exists():
        import json

        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return pd.DataFrame(payload.get("rows", []))
    return None


def render_markdown_table(rows: List[dict], metric_col: str) -> str:
    lines = [
        "| Model | " + metric_col + " (mean ± std) | n seeds |",
        "|-------|------------------------------|---------|",
    ]
    for r in rows:
        lines.append(f"| {r['model_id']} | {r['value']} | {r['n_seeds']} |")
    return "\n".join(lines) + "\n"


def render_latex_table(rows: List[dict], metric_col: str) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Baseline comparison (" + metric_col + ", mean $\\pm$ std over seeds).}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & " + metric_col + " & n \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r['model_id']} & {r['value']} & {r['n_seeds']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables from seeds_summary.parquet.")
    parser.add_argument(
        "--model_ids",
        default="gnn13_clip_bipartite,vlm_mlp,vlm_zeroshot,gnn12_clip_vlm_homo,gnn07_label_residual",
    )
    parser.add_argument("--protocol", default="calibrated4way")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--metric", default="", help="Metric column; auto-detect if empty.")
    parser.add_argument("--out_md", default="reports/comparison/table2_baselines.md")
    parser.add_argument("--out_tex", default="reports/comparison/table2_baselines.tex")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]

    summary_rows = []
    metric_col_used = args.metric or None

    for model_id in model_ids:
        df = load_seeds_summary(repo, model_id, args.protocol)
        if df is None or df.empty:
            summary_rows.append({"model_id": model_id, "value": "—", "n_seeds": 0})
            continue
        col = metric_col_used or find_metric_column(df, DEFAULT_METRIC_COLS)
        if col is None:
            summary_rows.append({"model_id": model_id, "value": "—", "n_seeds": len(df)})
            continue
        metric_col_used = col
        summary_rows.append(
            {
                "model_id": model_id,
                "value": mean_std(df[col].dropna()),
                "n_seeds": int(len(df)),
            }
        )

    metric_label = metric_col_used or "macro_f1"
    md = render_markdown_table(summary_rows, metric_label)
    tex = render_latex_table(summary_rows, metric_label)

    out_md = repo / args.out_md
    out_tex = repo / args.out_tex
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_tex.write_text(tex, encoding="utf-8")
    print({"wrote_md": str(out_md), "wrote_tex": str(out_tex), "metric": metric_label})


if __name__ == "__main__":
    main()
