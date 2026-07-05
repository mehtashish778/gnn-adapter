#!/usr/bin/env python3
"""NIH cross-site eval for Qwen3.5 fair CheXpert models (same 6k NIH set as Qwen2)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_PY = sys.executable

# Same NIH subset as Qwen2 cross-site report (test_rows_n6000.json paths).
NIH_TEST_QWEN35 = _REPO / "data/processed/splits/nih/test_rows_qwen35_2b_n6000.json"
NIH_TEST_QWEN2 = _REPO / "data/processed/splits/nih/test_rows_n6000.json"

FAIR_RUNS = {
    "cca": _REPO / "data/processed/experiments/cca/qwen35_qwen2_splits/cca_qwen35_vllm_2b_qwen2_splits",
    "cbm_posthoc": _REPO / "data/processed/experiments/cbm_posthoc/default/cbm_posthoc_qwen35_qwen2_splits",
    "cbm_labelfree": _REPO / "data/processed/experiments/cbm_labelfree/default/cbm_labelfree_qwen35_qwen2_splits",
}


def _run(cmd: list[str]) -> None:
    print({"run": " ".join(str(c) for c in cmd)})
    subprocess.run(cmd, cwd=str(_REPO), check=True)


def _n_rows(path: Path) -> int:
    return len(json.loads(path.read_text(encoding="utf-8"))["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--run_id", default="crosssite_eval_qwen35_2b")
    parser.add_argument("--protocol", default="nih")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--skip_patches", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vlm_zeroshot", "cca", "cbm_posthoc", "cbm_labelfree"],
    )
    args = parser.parse_args()

    if not NIH_TEST_QWEN35.is_file():
        raise FileNotFoundError(
            f"Missing {NIH_TEST_QWEN35}. Run Qwen3.5 NIH vLLM scoring + align first."
        )
    if NIH_TEST_QWEN2.is_file():
        n2, n35 = _n_rows(NIH_TEST_QWEN2), _n_rows(NIH_TEST_QWEN35)
        if n2 != n35:
            raise ValueError(f"NIH row count mismatch: Qwen2={n2} Qwen3.5={n35}")

    test_rows = NIH_TEST_QWEN35

    if not args.skip_patches:
        _run(
            [
                _PY,
                str(_SCRIPTS / "precompute_patch_cache.py"),
                "--rows_json",
                str(test_rows),
                "--image_root",
                args.image_root,
                "--protocol",
                args.protocol,
                "--split_name",
                "test",
                "--gpu_id",
                str(args.gpu_id),
            ]
        )

    if args.skip_eval:
        return

    if "vlm_zeroshot" in args.models:
        _run(
            [
                _PY,
                str(_SCRIPTS / "05_run_baseline_frozen_vlm.py"),
                "--model_id",
                "vlm_zeroshot",
                "--protocol",
                args.protocol,
                "--run_id",
                args.run_id,
                "--test_rows_json",
                str(test_rows),
                "--skip_val",
            ]
        )

    for model_id in args.models:
        if model_id == "vlm_zeroshot":
            continue
        ckpt = FAIR_RUNS.get(model_id)
        if ckpt is None or not ckpt.is_dir():
            raise FileNotFoundError(f"No fair CheXpert checkpoint for {model_id}: {ckpt}")
        _run(
            [
                _PY,
                str(_SCRIPTS / "eval_crosssite.py"),
                "--model_id",
                model_id,
                "--chexpert_run_dir",
                str(ckpt),
                "--test_rows_json",
                str(test_rows),
                "--image_root",
                args.image_root,
                "--protocol",
                args.protocol,
                "--run_id",
                args.run_id,
                "--gpu_id",
                str(args.gpu_id),
            ]
        )

    print({"status": "done", "nih_test_rows": str(test_rows), "run_id": args.run_id})


if __name__ == "__main__":
    main()
