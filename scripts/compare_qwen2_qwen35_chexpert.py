#!/usr/bin/env python3
"""Head-to-head CheXpert metrics: Qwen2-VL vs Qwen3.5-2B (fair splits, n=9,197 test)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

RUNS = {
    "Frozen VLM": {
        "Qwen2": REPO / "data/processed/splits/test_rows.json",
        "Qwen3.5-2B": REPO / "data/processed/splits/qwen35_qwen2_splits/test_rows.json",
    },
    "CBM post-hoc": {
        "Qwen2": REPO / "data/processed/experiments/cbm_posthoc/default/cbm_posthoc_default",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cbm_posthoc/default/cbm_posthoc_qwen35_qwen2_splits",
    },
    "CBM label-free": {
        "Qwen2": REPO / "data/processed/experiments/cbm_labelfree/default/cbm_labelfree_default",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cbm_labelfree/default/cbm_labelfree_qwen35_qwen2_splits",
    },
    "CCA": {
        "Qwen2": REPO / "data/processed/experiments/cca/default/cca_faithful",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cca/qwen35_qwen2_splits/cca_qwen35_vllm_2b_qwen2_splits",
    },
    "LoRA r16": {
        "Qwen2": REPO / "data/processed/experiments/qwen2vl_lora_r16/default/qwen2vl_lora_r16_v2",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/qwen35_2b_lora_r16/default/qwen35_2b_lora_r16_qwen2_splits",
    },
}

METRIC_KEYS = [
    "val_macro_f1@0.5",
    "test_macro_f1@0.5",
    "test_macro_f1@per_class_thr",
    "test_subset_accuracy@0.5",
    "test_macro_auroc",
    "test_macro_auprc",
    "test_macro_ece",
    "test_macro_brier",
    "trainable_params",
]


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _arrays_from_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _load_json(path)["rows"]
    probs = np.array([r["x_probs"] for r in rows], dtype=np.float64)
    y_true = np.array([r.get("y_true", r.get("y")) for r in rows], dtype=np.float64)
    n = probs.shape[1]
    y_mask = np.array([r.get("y_mask", [1] * n) for r in rows], dtype=np.float64)
    return probs, y_true, y_mask


def _arrays_from_preds(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_json(path)
    return (
        np.array(data["probs"], dtype=np.float64),
        np.array(data["y_true"], dtype=np.float64),
        np.array(data["y_mask"], dtype=np.float64),
    )


def _f1(probs, y_true, y_mask, thr=0.5):
    if isinstance(thr, (list, np.ndarray)):
        thr_arr = np.asarray(thr, dtype=np.float64)
        pred = (probs >= thr_arr.reshape(1, -1)).astype(np.float64)
    else:
        pred = (probs >= thr).astype(np.float64)
    f1s = []
    for i in range(probs.shape[1]):
        m = y_mask[:, i] > 0
        if m.sum() == 0:
            continue
        p, t = pred[m, i], y_true[m, i]
        tp = ((p == 1) & (t == 1)).sum()
        fp = ((p == 1) & (t == 0)).sum()
        fn = ((p == 0) & (t == 1)).sum()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    auc = 0.0
    for pv in pos:
        auc += float((pv > neg).sum()) + 0.5 * float((pv == neg).sum())
    return auc / (len(pos) * len(neg))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y = y[order]
    n_pos = float(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = 0.0
    precisions = []
    for i, yi in enumerate(y, start=1):
        if yi:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else 0.0


def _macro(probs, y_true, y_mask, fn):
    vals = []
    for i in range(probs.shape[1]):
        m = y_mask[:, i] > 0.5
        if m.sum() < 2:
            continue
        p = np.clip(np.nan_to_num(probs[m, i], nan=0.5), 0.0, 1.0)
        y = y_true[m, i]
        n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
        if n_pos > 0 and n_neg > 0:
            vals.append(fn(y, p))
    return float(np.mean(vals)) if vals else float("nan")


def _ece(probs, y_true, y_mask, n_bins=15):
    eces = []
    for i in range(probs.shape[1]):
        m = y_mask[:, i] > 0.5
        if m.sum() < 2:
            continue
        p = np.clip(probs[m, i], 0.0, 1.0)
        y = y_true[m, i]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            sel = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
            if sel.sum() == 0:
                continue
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
        eces.append(ece)
    return float(np.mean(eces)) if eces else float("nan")


def _brier(probs, y_true, y_mask):
    vals = []
    for i in range(probs.shape[1]):
        m = y_mask[:, i] > 0.5
        if m.sum() == 0:
            continue
        p = np.clip(probs[m, i], 0.0, 1.0)
        y = y_true[m, i]
        vals.append(float(np.mean((p - y) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


def _subset_acc(probs, y_true, y_mask, thr=0.5):
    pred = (probs >= thr).astype(np.float64)
    correct = []
    for j in range(len(y_true)):
        m = y_mask[j] > 0
        if m.sum() == 0:
            continue
        correct.append(float(np.all(pred[j, m] == y_true[j, m])))
    return float(np.mean(correct)) if correct else 0.0


def _resolve_arrays(source: Path):
    if source.name.endswith("rows.json"):
        return _arrays_from_rows(source)
    if (source / "test_predictions.json").exists():
        return _arrays_from_preds(source / "test_predictions.json")
    raise FileNotFoundError(source)


def _get_metric(source: Path, key: str, metrics: dict, probs, y_true, y_mask) -> float | int | None:
    if key == "trainable_params":
        return metrics.get("trainable_params", 0 if "rows.json" in source.name else None)
    val = metrics.get(key)
    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        return val
    if key == "test_macro_f1@0.5":
        return _f1(probs, y_true, y_mask)
    if key == "test_macro_f1@per_class_thr":
        thr_path = REPO / "data/processed/experiments/thresholds/per_class_thresholds.json"
        if thr_path.exists():
            thr = _load_json(thr_path)
            if isinstance(thr, list):
                return _f1(probs, y_true, y_mask, thr=thr)
        return None
    if key == "test_subset_accuracy@0.5":
        v = metrics.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
        return _subset_acc(probs, y_true, y_mask)
    if key == "test_macro_auroc":
        return _macro(probs, y_true, y_mask, _roc_auc)
    if key == "test_macro_auprc":
        return _macro(probs, y_true, y_mask, _auprc)
    if key == "test_macro_ece":
        v = metrics.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
        return _ece(probs, y_true, y_mask)
    if key == "test_macro_brier":
        v = metrics.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
        return _brier(probs, y_true, y_mask)
    if key == "val_macro_f1@0.5":
        return metrics.get(key)
    return None


def collect() -> dict:
    out = {}
    for method, backends in RUNS.items():
        out[method] = {}
        for backend, source in backends.items():
            metrics = {}
            if source.is_dir() and (source / "metrics.json").exists():
                metrics = _load_json(source / "metrics.json")
            elif source.name == "metrics.json":
                metrics = _load_json(source)
            probs, y_true, y_mask = _resolve_arrays(source)
            row = {"n_test": int(len(y_true))}
            for key in METRIC_KEYS:
                row[key] = _get_metric(source, key, metrics, probs, y_true, y_mask)
            out[method][backend] = row
    return out


def main() -> None:
    data = collect()
    out_path = REPO / "reports/comparison/qwen2_vs_qwen35_chexpert_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
