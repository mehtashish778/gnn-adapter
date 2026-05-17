#!/usr/bin/env python3
"""
Aggregate multi-seed metrics, bootstrap CIs, DeLong AUROC tests, Benjamini-Hochberg correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_predictions(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with (run_dir / "test_predictions.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    probs = np.array(data["probs"], dtype=np.float64)
    y = np.array(data["y_true"], dtype=np.float64)
    m = np.array(data["y_mask"], dtype=np.float64)
    return probs, y, m


def macro_f1(probs: np.ndarray, y: np.ndarray, m: np.ndarray, thr: float = 0.5) -> float:
    pred = (probs >= thr).astype(np.float64)
    f1s = []
    for c in range(y.shape[1]):
        mask = m[:, c] > 0.5
        if mask.sum() == 0:
            continue
        yt = y[mask, c]
        yp = pred[mask, c]
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def per_class_auc(y: np.ndarray, scores: np.ndarray, m: np.ndarray) -> List[float]:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for c in range(y.shape[1]):
        mask = m[:, c] > 0.5
        yt = y[mask, c]
        if mask.sum() < 10 or len(np.unique(yt)) < 2:
            aucs.append(float("nan"))
            continue
        aucs.append(float(roc_auc_score(yt, scores[mask, c])))
    return aucs


def bootstrap_auc_diff_pvalue(
    y: np.ndarray,
    scores_ref: np.ndarray,
    scores_cmp: np.ndarray,
    m: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
) -> Tuple[float, float]:
    """Paired bootstrap on mean per-class AUROC (DeLong substitute when pydelong unavailable)."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    ref_aucs = per_class_auc(y, scores_ref, m)
    cmp_aucs = per_class_auc(y, scores_cmp, m)
    obs = float(np.nanmean(np.array(cmp_aucs) - np.array(ref_aucs)))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(
            float(np.nanmean(np.array(per_class_auc(y[idx], scores_cmp[idx], m[idx]))
            - np.array(per_class_auc(y[idx], scores_ref[idx], m[idx]))))
        )
    diffs = np.array(diffs, dtype=np.float64)
    p = float(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
    return obs, p


def benjamini_hochberg(p_values: List[float], q: float = 0.05) -> List[bool]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = np.zeros(m, dtype=bool)
    max_i = -1
    for i, (p, t) in enumerate(zip(ranked, thresh)):
        if p <= t:
            max_i = i
    if max_i >= 0:
        passed[order[: max_i + 1]] = True
    return passed.tolist()


def collect_seed_runs(repo: Path, model_id: str, protocol: str) -> List[Path]:
    base = repo / "data/processed/experiments" / model_id / protocol
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("seeds_")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--models", nargs="+", default=["cca", "vlm_mlp", "mlgcn", "qformer_adapter"])
    parser.add_argument("--reference", default="cca")
    parser.add_argument("--out_md", default="reports/comparison/stats.md")
    args = parser.parse_args()

    repo = Path(args.repo)
    rows = []
    ref_probs = ref_y = ref_m = None

    for model_id in args.models:
        runs = collect_seed_runs(repo, model_id, args.protocol)
        f1s = []
        for run_dir in runs:
            if not (run_dir / "test_predictions.json").exists():
                continue
            probs, y, m = load_predictions(run_dir)
            f1s.append(macro_f1(probs, y, m))
            if model_id == args.reference and ref_probs is None:
                ref_probs, ref_y, ref_m = probs, y, m
        mean, lo, hi = bootstrap_ci(f1s)
        rows.append((model_id, mean, lo, hi, len(f1s)))

    lines = ["# Multi-seed comparison (bootstrap 95% CI on test macro-F1 @0.5)", ""]
    lines.append("| Model | mean F1 | 95% CI | n seeds |")
    lines.append("|-------|---------|--------|---------|")
    for model_id, mean, lo, hi, n in rows:
        lines.append(f"| {model_id} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | {n} |")

    ref_run = None
    for model_id in args.models:
        if model_id != args.reference:
            continue
        runs = collect_seed_runs(repo, model_id, args.protocol)
        if runs and (runs[0] / "test_predictions.json").exists():
            ref_run = runs[0]
            break

    if ref_run is not None:
        ref_probs, ref_y, ref_m = load_predictions(ref_run)
        lines.extend(
            [
                "",
                f"## Bootstrap AUROC vs `{args.reference}` (paired, BH q=0.05)",
                "",
                "| Model | Δ mean AUROC | p (bootstrap) | BH reject |",
                "|-------|--------------|---------------|-----------|",
            ]
        )
        cmp_rows: List[Tuple[str, float, float]] = []
        for model_id in args.models:
            if model_id == args.reference:
                continue
            runs = collect_seed_runs(repo, model_id, args.protocol)
            if not runs or not (runs[0] / "test_predictions.json").exists():
                continue
            probs, _, _ = load_predictions(runs[0])
            delta, p = bootstrap_auc_diff_pvalue(ref_y, ref_probs, probs, ref_m)
            cmp_rows.append((model_id, delta, p))
        if cmp_rows:
            bh = benjamini_hochberg([p for _, _, p in cmp_rows])
            for (model_id, delta, p), ok in zip(cmp_rows, bh):
                lines.append(f"| {model_id} | {delta:+.4f} | {p:.4f} | {'yes' if ok else 'no'} |")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print({"wrote": str(out_path), "models": args.models})


if __name__ == "__main__":
    main()
