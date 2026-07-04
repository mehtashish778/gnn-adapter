#!/usr/bin/env python3
"""
Build concept priors (none / co-occurrence / co-error / RadGraph / permuted) and train CCA variants.

Prior JSON format: {"matrix": P×P float, "source": str, ...}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_concept_prior import build_cooccurrence_prior
from common_multilabel import load_rows, write_json
from models.architectures.cca import DEFAULT_CONCEPT_PHRASES


def prior_from_coerror(coerror_json: Path, num_primitives: int) -> np.ndarray:
    with coerror_json.open("r", encoding="utf-8") as f:
        w = np.array(json.load(f), dtype=np.float32)
    p_use = min(num_primitives, w.shape[0])
    mat = np.eye(num_primitives, dtype=np.float32) * 0.05
    mat[:p_use, :p_use] = w[:p_use, :p_use].astype(np.float32)
    return mat


def write_prior(path: Path, mat: np.ndarray, source: str, num_primitives: int) -> None:
    write_json(
        path,
        {
            "matrix": mat.tolist(),
            "num_primitives": num_primitives,
            "concept_phrases": DEFAULT_CONCEPT_PHRASES[:num_primitives],
            "source": source,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="CCA concept-prior ablation driver.")
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--coerror_json", default="data/processed/graph/coerror_matrix_normalized.json")
    parser.add_argument("--num_primitives", type=int, default=30)
    parser.add_argument("--out_dir", default="data/processed/graph/prior_ablation")
    parser.add_argument("--radgraph_json", default="", help="Optional external RadGraph prior.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_id_prefix", default="prior_ablation")
    parser.add_argument("--dry_run", action="store_true", help="Only build prior JSON files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.train_rows_json))

    co_occur_path = out_dir / "co_occur_P30.json"
    coerror_path = out_dir / "coerror_P30.json"
    radgraph_path = out_dir / "radgraph_P30.json"
    permuted_path = out_dir / "radgraph_P30_permuted.json"

    write_prior(
        co_occur_path,
        build_cooccurrence_prior(rows, args.num_primitives),
        "cooccurrence_train_labels",
        args.num_primitives,
    )
    if Path(args.coerror_json).exists():
        write_prior(
            coerror_path,
            prior_from_coerror(Path(args.coerror_json), args.num_primitives),
            "coerror_matrix_normalized",
            args.num_primitives,
        )
    else:
        print(f"Warning: co-error matrix missing: {args.coerror_json}")

    if args.radgraph_json and Path(args.radgraph_json).exists():
        with Path(args.radgraph_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        mat = np.array(data.get("matrix") or data.get("prior"), dtype=np.float32)
        write_prior(radgraph_path, mat, "radgraph_external", args.num_primitives)
    else:
        import shutil

        shutil.copy2(co_occur_path, radgraph_path)
        with radgraph_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        d["source"] = "radgraph_placeholder_cooccurrence"
        write_json(radgraph_path, d)

    subprocess.check_call(
        [
            sys.executable,
            str(_SCRIPT_DIR / "permute_prior.py"),
            "--in_json",
            str(radgraph_path),
            "--out_json",
            str(permuted_path),
            "--seed",
            "0",
        ],
        cwd=_SCRIPT_DIR.parent,
    )

    variants = [
        ("none", ""),
        ("co_occur", str(co_occur_path)),
        ("coerror", str(coerror_path) if coerror_path.exists() else ""),
        ("radgraph", str(radgraph_path)),
        ("permuted", str(permuted_path)),
    ]

    print({"built_priors": [v[0] for v in variants]})
    if args.dry_run:
        return

    train_script = _SCRIPT_DIR / "14_train_cca.py"
    for name, prior_json in variants:
        run_id = f"{args.run_id_prefix}_{name}"
        cmd = [
            sys.executable,
            str(train_script),
            "--model_id",
            "cca",
            "--protocol",
            "default",
            "--gpu_id",
            str(args.gpu_id),
            "--num_workers",
            "0",
            "--epochs",
            str(args.epochs),
            "--run_id",
            run_id,
            "--no-save_attention_maps",
        ]
        if prior_json:
            cmd.extend(["--radgraph_prior_json", prior_json])
        print({"training": name, "cmd": cmd})
        subprocess.check_call(cmd, cwd=_SCRIPT_DIR.parent)


if __name__ == "__main__":
    main()
