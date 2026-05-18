#!/usr/bin/env python3
"""
Train CCA variants at three lambda_sparse levels (faithfulness Pareto sweep) and write comparison report.

Reuses existing runs when metrics.json is present (e.g. cca_faithful @ lambda_sparse=0.01).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent

# (run_id, lambda_sparse, optional existing alias dir under cca/default)
PARETO_VARIANTS: List[Dict[str, Any]] = [
    {"run_id": "faith_pareto_ls1e3", "lambda_sparse": 1e-3, "alias": None},
    {"run_id": "faith_pareto_ls1e2", "lambda_sparse": 1e-2, "alias": "cca_faithful"},
    {"run_id": "faith_pareto_ls1e1", "lambda_sparse": 1e-1, "alias": None},
]

TRAIN_BASE = [
    sys.executable,
    str(_SCRIPT_DIR / "14_train_cca.py"),
    "--model_id",
    "cca",
    "--protocol",
    "default",
    "--gpu_id",
    "0",
    "--num_workers",
    "0",
    "--epochs",
    "60",
    "--early_stop_patience",
    "16",
    "--best_metric",
    "val_macro_f1_05",
    "--batch_size",
    "16",
    "--query_dim",
    "128",
    "--n_cross_attn_layers",
    "2",
    "--n_self_attn_layers",
    "2",
    "--n_heads",
    "2",
    "--alpha",
    "1.0",
    "--dropout",
    "0.1",
    "--lr",
    "0.0003",
    "--weight_decay",
    "0.0001",
    "--lambda_faithful",
    "0.1",
    "--sparsity_target",
    "0.1",
    "--use_gate_M",
    "--init_queries_from_text",
    "--no-save_attention_maps",
]


def load_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_dir_for(run_id: str) -> Path:
    return _REPO / "data/processed/experiments/cca/default" / run_id


def train_variant(v: Dict[str, Any], gpu_id: int, dry_run: bool) -> None:
    out = run_dir_for(v["run_id"])
    if out.joinpath("metrics.json").exists():
        print({"skip_existing": v["run_id"]})
        return
    alias = v.get("alias")
    if alias:
        alias_dir = run_dir_for(alias)
        if alias_dir.joinpath("metrics.json").exists():
            print({"reuse_alias": alias, "as": v["run_id"]})
            return
    cmd = list(TRAIN_BASE)
    idx = cmd.index("--gpu_id")
    cmd[idx + 1] = str(gpu_id)
    cmd.extend(["--run_id", v["run_id"], "--lambda_sparse", str(v["lambda_sparse"])])
    print({"training": v["run_id"], "lambda_sparse": v["lambda_sparse"]})
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=_REPO)


def collect_point(run_id: str, alias: str | None, lambda_sparse: float) -> dict:
    d = run_dir_for(run_id)
    if not d.joinpath("metrics.json").exists() and alias:
        d = run_dir_for(alias)
        run_id = alias
    m = load_metrics(d)
    if not m:
        return {"run_id": run_id, "lambda_sparse": lambda_sparse, "missing": True}
    h = m.get("hparams") or {}
    return {
        "run_id": run_id,
        "lambda_sparse": float(h.get("lambda_sparse", lambda_sparse)),
        "test_macro_f1@0.5": m.get("test_macro_f1@0.5"),
        "test_macro_auroc": m.get("test_macro_auroc"),
        "val_macro_f1@0.5": m.get("val_macro_f1@0.5"),
        "gate_density_eval": m.get("gate_density_eval"),
        "intervention_consistency": m.get("intervention_consistency")
        or m.get("faithfulness_intervention_consistency"),
        "necessity_drop": m.get("faithfulness_necessity_drop"),
        "sufficiency_f1": m.get("faithfulness_sufficiency_f1"),
    }


def _fmt(x, ndigits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{ndigits}f}"
    return str(x)


def write_report(points: List[dict], out_md: Path) -> None:
    lines = [
        "# Faithfulness–utility Pareto (λ_sparse sweep, frozen patches, gate on)",
        "",
        "Config: default CCA (~435K params), `lambda_faithful=0.1`, `use_gate_M=true`, frozen CLIP patches.",
        "",
        "| run_id | λ_sparse | gate density | test F1 | test AUROC | intervention consistency | necessity drop |",
        "|--------|----------|--------------|---------|------------|--------------------------|----------------|",
    ]
    for p in points:
        if p.get("missing"):
            lines.append(
                f"| {p['run_id']} | {p['lambda_sparse']:.0e} | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {p['run_id']} | {p['lambda_sparse']:.0e} | "
            f"{_fmt(p.get('gate_density_eval'), 3)} | "
            f"{_fmt(p.get('test_macro_f1@0.5'))} | "
            f"{_fmt(p.get('test_macro_auroc'))} | "
            f"{_fmt(p.get('intervention_consistency'))} | "
            f"{_fmt(p.get('necessity_drop'))} |"
        )
    lines.extend(
        [
            "",
            "Utility axis: test macro-F1 @0.5. Faithfulness axis: intervention consistency (higher = more faithful).",
            "",
            "See also: [`docs/cca_experiment_results.md`](../docs/cca_experiment_results.md).",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print({"wrote": str(out_md), "n_points": len(points)})


def main():
    parser = argparse.ArgumentParser(description="Faithfulness Pareto λ_sparse sweep.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--compare_only", action="store_true")
    parser.add_argument("--out_md", default="reports/comparison/cca_faithfulness_pareto.md")
    args = parser.parse_args()

    if not args.compare_only:
        for v in PARETO_VARIANTS:
            train_variant(v, args.gpu_id, args.dry_run)

    points = [collect_point(v["run_id"], v.get("alias"), v["lambda_sparse"]) for v in PARETO_VARIANTS]
    write_report(points, Path(args.out_md))


if __name__ == "__main__":
    main()
