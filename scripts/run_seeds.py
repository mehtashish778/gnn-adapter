#!/usr/bin/env python3
"""
Multi-seed training launcher: runs model entry points per seed and aggregates metrics to parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common_multilabel import write_json


MODEL_ENTRY_SCRIPTS = {
    "vlm_zeroshot": "scripts/models/vlm_zeroshot/run_default.py",
    "vlm_mlp": "scripts/models/vlm_mlp/train.py",
    "gnn07_label_residual": "scripts/models/gnn07_label_residual/train.py",
    "gnn12_clip_vlm_homo": "scripts/models/gnn12_clip_vlm_homo/train.py",
    "gnn13_clip_bipartite": "scripts/models/gnn13_clip_bipartite/train.py",
    "cca": "scripts/models/cca/train.py",
}

NUMBERED_SCRIPTS = {
    "vlm_zeroshot": "scripts/05_run_baseline_frozen_vlm.py",
    "vlm_mlp": "scripts/06_run_baseline_mlp.py",
    "gnn07_label_residual": "scripts/07_train_gnn_adapter.py",
    "gnn12_clip_vlm_homo": "scripts/12_train_clip_vlm_gnn_adapter.py",
    "gnn13_clip_bipartite": "scripts/13_train_bipartite_gnn_adapter.py",
    "cca": "scripts/14_train_cca.py",
}


def git_hash(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_seeds(seeds_arg: str) -> List[int]:
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_run_metrics(run_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run_dir": str(run_dir)}
    for name in (
        "metrics.json",
        "val_metrics_calibrated.json",
        "test_metrics_calibrated.json",
    ):
        data = read_json(run_dir / name)
        if not data:
            continue
        prefix = name.replace(".json", "")
        for k, v in data.items():
            if isinstance(v, (int, float)):
                row[f"{prefix}.{k}"] = v
            elif k in ("macro_f1", "subset_accuracy"):
                row[f"{prefix}.{k}"] = v
    return row


def main():
    parser = argparse.ArgumentParser(description="Run training across multiple seeds and summarize to parquet.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--run_id_prefix", default="seeds")
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument(
        "--use_numbered_script",
        action="store_true",
        help="Call numbered scripts directly instead of models/*/train.py wrappers.",
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to training script.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    seeds = parse_seeds(args.seeds)
    ghash = git_hash(repo)

    if args.use_numbered_script:
        script_rel = NUMBERED_SCRIPTS.get(args.model_id)
    else:
        script_rel = MODEL_ENTRY_SCRIPTS.get(args.model_id)
    if not script_rel:
        raise KeyError(f"Unknown model_id {args.model_id!r}")

    script_path = repo / script_rel
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{args.run_id_prefix}_s{seed}"
        cmd = [
            sys.executable,
            str(script_path),
            "--model_id",
            args.model_id,
            "--protocol",
            args.protocol,
            "--run_id",
            run_id,
            "--seed",
            str(seed),
        ]
        if args.extra_args:
            extra = list(args.extra_args)
            if extra and extra[0] == "--":
                extra = extra[1:]
            cmd.extend(extra)

        env = os.environ.copy()
        scripts_path = str(repo / "scripts")
        env["PYTHONPATH"] = scripts_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        print({"running": cmd})
        subprocess.check_call(cmd, cwd=repo, env=env)

        run_dir = repo / "data/processed/experiments" / args.model_id / args.protocol / run_id
        metrics_row = collect_run_metrics(run_dir)
        metrics_row.update(
            {
                "seed": seed,
                "run_id": run_id,
                "git_hash": ghash,
                "model_id": args.model_id,
                "protocol": args.protocol,
            }
        )
        rows.append(metrics_row)

    import pandas as pd

    df = pd.DataFrame(rows)
    out_dir = repo / "data/processed/experiments" / args.model_id / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "seeds_summary.parquet"
    df.to_parquet(parquet_path, index=False)
    # also write JSON for environments without parquet readers
    json_path = out_dir / "seeds_summary.json"
    write_json(json_path, {"git_hash": ghash, "rows": rows})
    print({"wrote": str(parquet_path), "n_seeds": len(rows)})


if __name__ == "__main__":
    main()
