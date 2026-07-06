#!/usr/bin/env python3
"""Refresh qwen2_vs_qwen35_nih_metrics.json from experiment outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/comparison/qwen2_vs_qwen35_nih_metrics.json"

RUNS = {
    "Frozen VLM": {
        "Qwen2": REPO / "data/processed/experiments/vlm_zeroshot/nih/crosssite_eval",
        "Qwen3.5-2B": REPO / "data/processed/experiments/vlm_zeroshot/nih/crosssite_eval_qwen35_2b",
    },
    "CBM post-hoc": {
        "Qwen2": REPO / "data/processed/experiments/cbm_posthoc/nih/crosssite_eval",
        "Qwen3.5-2B": REPO / "data/processed/experiments/cbm_posthoc/nih/crosssite_eval_qwen35_2b",
    },
    "CBM label-free": {
        "Qwen2": REPO / "data/processed/experiments/cbm_labelfree/nih/crosssite_eval",
        "Qwen3.5-2B": REPO / "data/processed/experiments/cbm_labelfree/nih/crosssite_eval_qwen35_2b",
    },
    "CCA": {
        "Qwen2": REPO / "data/processed/experiments/cca/nih/crosssite_eval",
        "Qwen3.5-2B": REPO / "data/processed/experiments/cca/nih/crosssite_eval_qwen35_2b",
    },
    "LoRA r16": {
        "Qwen2": REPO / "data/processed/experiments/qwen2vl_lora_r16/nih/crosssite_eval",
        "Qwen3.5-2B": REPO / "data/processed/experiments/qwen35_2b_lora_r16/nih/crosssite_eval_qwen35_2b",
    },
}

TEST_ROWS = {
    "Qwen2": REPO / "data/processed/splits/nih/test_rows_n6000.json",
    "Qwen3.5-2B": REPO / "data/processed/splits/nih/test_rows_qwen35_2b_n6000.json",
}
LORA_TEST_ROWS = REPO / "data/processed/splits/nih/test_rows_n6000.json"


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    auc = sum(float((pv > neg).sum()) + 0.5 * float((pv == neg).sum()) for pv in pos)
    return auc / (len(pos) * len(neg))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y = y[order]
    n_pos = float(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = 0.0
    precs = []
    for i, yi in enumerate(y, start=1):
        if yi:
            tp += 1
            precs.append(tp / i)
    return float(np.mean(precs)) if precs else 0.0


def _macro(probs, y_true, y_mask, fn):
    vals = []
    for i in range(probs.shape[1]):
        m = y_mask[:, i] > 0.5
        if m.sum() < 2:
            continue
        p = np.clip(np.nan_to_num(probs[m, i], nan=0.5), 0.0, 1.0)
        y = y_true[m, i]
        if y.sum() > 0 and y.sum() < len(y):
            vals.append(fn(y, p))
    return float(np.mean(vals)) if vals else float("nan")


def _load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pred = run_dir / "test_predictions.json"
    if not pred.exists():
        return None
    data = json.loads(pred.read_text(encoding="utf-8"))
    return (
        np.array(data["probs"], dtype=np.float64),
        np.array(data["y_true"], dtype=np.float64),
        np.array(data["y_mask"], dtype=np.float64),
    )


def _val(metrics: dict, key: str):
    v = metrics.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def collect() -> dict:
    out: dict = {
        "_meta": {
            "protocol": "nih",
            "n_test": 6000,
            "description": "CheXpert-trained models evaluated on NIH ChestX-ray14 6k subset",
            "test_rows_qwen2": str(TEST_ROWS["Qwen2"].relative_to(REPO)).replace("\\", "/"),
            "test_rows_qwen35_vlm": str(TEST_ROWS["Qwen3.5-2B"].relative_to(REPO)).replace("\\", "/"),
            "lora_test_rows": str(LORA_TEST_ROWS.relative_to(REPO)).replace("\\", "/"),
            "note": "LoRA uses test_rows_n6000.json (same 6000 image paths as Qwen2; image-based scoring).",
        }
    }
    for method, backends in RUNS.items():
        out[method] = {}
        for backend, run_dir in backends.items():
            if not run_dir.is_dir():
                raise FileNotFoundError(run_dir)
            m = _load_metrics(run_dir)
            row = {
                "n_test": 6000,
                "status": "complete",
                "cross_site": m.get("cross_site", True),
                "test_macro_f1@0.5": _val(m, "test_macro_f1@0.5"),
                "test_macro_f1@per_class_thr": _val(m, "test_macro_f1@per_class_thr"),
                "test_subset_accuracy@0.5": _val(m, "test_subset_accuracy@0.5"),
                "test_macro_auroc": _val(m, "test_macro_auroc"),
                "test_macro_auprc": _val(m, "test_macro_auprc"),
                "test_macro_ece": _val(m, "test_macro_ece"),
                "test_macro_brier": _val(m, "test_macro_brier"),
                "trainable_params": m.get("trainable_params"),
                "run_dir": str(run_dir.relative_to(REPO)).replace("\\", "/"),
            }
            if m.get("chexpert_run_dir"):
                row["chexpert_run_dir"] = str(m["chexpert_run_dir"]).replace("\\", "/")
            if method == "LoRA r16":
                row["test_rows_json"] = str(LORA_TEST_ROWS.relative_to(REPO)).replace("\\", "/")
                row["scored_mode"] = m.get("scored_mode", "cls")
            elif backend == "Qwen3.5-2B" and method != "LoRA r16":
                row["test_rows_json"] = str(TEST_ROWS["Qwen3.5-2B"].relative_to(REPO)).replace("\\", "/")
            else:
                row["test_rows_json"] = str(TEST_ROWS["Qwen2"].relative_to(REPO)).replace("\\", "/")

            arrays = _load_arrays(run_dir)
            if arrays is not None:
                probs, y_true, y_mask = arrays
                row["n_predictions"] = int(len(y_true))
                if row["test_macro_auroc"] is None:
                    row["test_macro_auroc"] = _macro(probs, y_true, y_mask, _roc_auc)
                    row["test_macro_auroc_source"] = "computed_from_predictions"
                if row["test_macro_auprc"] is None:
                    row["test_macro_auprc"] = _macro(probs, y_true, y_mask, _auprc)
                    row["test_macro_auprc_source"] = "computed_from_predictions"
            out[method][backend] = row
    return out


def main() -> None:
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
