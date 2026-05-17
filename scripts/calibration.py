"""
Post-hoc threshold tuning and calibrated evaluation (four-way protocol).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from common_multilabel import (
    f1_from_counts,
    subset_accuracy_masked_lists,
    write_json,
)


def class_f1(y_true, y_mask, probs, threshold: float, class_idx: int) -> float:
    tp = fp = fn = 0
    for y, m, p in zip(y_true, y_mask, probs):
        if m[class_idx] == 0:
            continue
        pred = 1 if p[class_idx] >= threshold else 0
        if pred == 1 and y[class_idx] == 1:
            tp += 1
        elif pred == 1 and y[class_idx] == 0:
            fp += 1
        elif pred == 0 and y[class_idx] == 1:
            fn += 1
    return f1_from_counts(tp, fp, fn)


def tune_thresholds(
    calib_predictions: Dict[str, Any],
    method: Literal["per_class_grid", "isotonic", "temperature"] = "per_class_grid",
    grid_start: int = 5,
    grid_stop: int = 96,
    grid_step: int = 5,
) -> List[float]:
    """
    Tune per-class thresholds on calibration predictions.

    Currently only ``per_class_grid`` is implemented (same grid search as 08_tune_thresholds).
    """
    if method != "per_class_grid":
        raise NotImplementedError(f"Threshold method {method!r} is not implemented yet.")

    probs = calib_predictions["probs"]
    y_true = calib_predictions["y_true"]
    y_mask = calib_predictions["y_mask"]
    c = len(probs[0])
    thresholds = []
    best_f1 = []
    for i in range(c):
        best_t = 0.5
        best = -1.0
        for k in range(grid_start, grid_stop, grid_step):
            t = k / 100.0
            f1 = class_f1(y_true, y_mask, probs, t, i)
            if f1 > best:
                best = f1
                best_t = t
        thresholds.append(best_t)
        best_f1.append(best)
    return thresholds


def tune_thresholds_from_file(
    val_predictions_json: Union[str, Path],
    out_json: Union[str, Path],
    method: Literal["per_class_grid", "isotonic", "temperature"] = "per_class_grid",
) -> Dict[str, Any]:
    """Load predictions JSON, tune thresholds, write output JSON."""
    with Path(val_predictions_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    thresholds = tune_thresholds(data, method=method)
    val_class_f1 = [
        class_f1(data["y_true"], data["y_mask"], data["probs"], thresholds[i], i)
        for i in range(len(thresholds))
    ]
    payload = {"thresholds": thresholds, "val_class_f1": val_class_f1}
    write_json(Path(out_json), payload)
    return payload


def calibrated_eval(
    predictions: Dict[str, Any],
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate predictions with frozen per-class thresholds."""
    probs = predictions["probs"]
    y_true = predictions["y_true"]
    y_mask = predictions["y_mask"]
    c = len(thresholds)
    per_class = []
    for i in range(c):
        tp = fp = fn = tn = 0
        for y, m, p in zip(y_true, y_mask, probs):
            if m[i] == 0:
                continue
            pred = 1 if p[i] >= thresholds[i] else 0
            gt = int(y[i])
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 1:
                fn += 1
            else:
                tn += 1
        f1 = f1_from_counts(tp, fp, fn)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        per_class.append(
            {"f1": f1, "recall": rec, "precision": prec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        )
    macro_f1 = sum(x["f1"] for x in per_class) / c if c else 0.0
    subset_acc, subset_n = subset_accuracy_masked_lists(probs, y_true, y_mask, thresholds)
    return {
        "macro_f1": macro_f1,
        "subset_accuracy": subset_acc,
        "subset_n_examples": subset_n,
        "per_class": per_class,
    }


def calibrated_eval_from_files(
    test_predictions_json: Union[str, Path],
    thresholds_json: Union[str, Path],
    out_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load test predictions and thresholds JSON, evaluate, optionally write metrics."""
    with Path(test_predictions_json).open("r", encoding="utf-8") as f:
        test_data = json.load(f)
    with Path(thresholds_json).open("r", encoding="utf-8") as f:
        thresholds = json.load(f)["thresholds"]
    out = calibrated_eval(test_data, thresholds)
    if out_json is not None:
        write_json(Path(out_json), out)
    return out


def run_calibrated_pipeline(
    run_dir: Path,
    *,
    calib_predictions_name: str = "calib_predictions.json",
    val_predictions_name: str = "val_predictions.json",
    test_predictions_name: str = "test_predictions.json",
    thresholds_name: str = "per_class_thresholds.json",
) -> Dict[str, Path]:
    """
    Tune thresholds on calib split, evaluate val and test (four-way protocol).
    Returns paths to written metric files.
    """
    run_dir = Path(run_dir)
    calib_path = run_dir / calib_predictions_name
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing {calib_path}")

    thr_path = run_dir / thresholds_name
    tune_thresholds_from_file(calib_path, thr_path)

    with thr_path.open("r", encoding="utf-8") as f:
        thresholds = json.load(f)["thresholds"]

    written = {"thresholds": thr_path}
    for split_name, pred_name, metric_name in (
        ("val", val_predictions_name, "val_metrics_calibrated.json"),
        ("test", test_predictions_name, "test_metrics_calibrated.json"),
    ):
        pred_path = run_dir / pred_name
        if pred_path.exists():
            metric_path = run_dir / metric_name
            with pred_path.open("r", encoding="utf-8") as f:
                preds = json.load(f)
            metrics = calibrated_eval(preds, thresholds)
            write_json(metric_path, metrics)
            written[split_name] = metric_path
    return written
