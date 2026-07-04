#!/usr/bin/env python3
"""
Aggregate multi-seed metrics, bootstrap CIs, paired bootstrap AUROC tests, Benjamini-Hochberg correction.

Supports:
  - Multi-seed dirs: seeds_s{0,1,...} or {prefix}_seeds_s{0,1,...} (e.g. lora_r8_trial27_seeds_s0)
  - Single-run dirs: *_default or any run with test_predictions.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SEED_DIR = re.compile(r"^(?P<prefix>.+_)?seeds_s(?P<seed>\d+)$")


def load_predictions(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with (run_dir / "test_predictions.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    probs = np.array(data["probs"], dtype=np.float64)
    y = np.array(data["y_true"], dtype=np.float64)
    m = np.array(data["y_mask"], dtype=np.float64)
    return probs, y, m


def load_metrics_json(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def macro_auroc(probs: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for c in range(y.shape[1]):
        mask = m[:, c] > 0.5
        yt = y[mask, c]
        if mask.sum() < 10 or len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, probs[mask, c])))
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def bootstrap_auc_diff_pvalue(
    y: np.ndarray,
    scores_ref: np.ndarray,
    scores_cmp: np.ndarray,
    m: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
) -> Tuple[float, float]:
    """Paired bootstrap on mean per-class AUROC (DeLong substitute)."""
    from sklearn.metrics import roc_auc_score

    def per_class_auc(scores: np.ndarray) -> List[float]:
        aucs = []
        for c in range(y.shape[1]):
            mask = m[:, c] > 0.5
            yt = y[mask, c]
            if mask.sum() < 10 or len(np.unique(yt)) < 2:
                aucs.append(float("nan"))
                continue
            aucs.append(float(roc_auc_score(yt, scores[mask, c])))
        return aucs

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    ref_aucs = per_class_auc(scores_ref)
    cmp_aucs = per_class_auc(scores_cmp)
    obs = float(np.nanmean(np.array(cmp_aucs) - np.array(ref_aucs)))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(
            float(
                np.nanmean(np.array(per_class_auc(scores_cmp[idx]))
                - np.array(per_class_auc(scores_ref[idx])))
            )
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


def group_seed_runs(base: Path) -> Dict[str, List[Path]]:
    """Group run dirs by seed-prefix (e.g. 'seeds' vs 'lora_r8_trial27_seeds')."""
    groups: Dict[str, List[Path]] = {}
    if not base.exists():
        return groups
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        m = _SEED_DIR.match(p.name)
        if m:
            prefix = m.group("prefix") or ""
            key = f"{prefix}seeds" if prefix else "seeds"
            groups.setdefault(key, []).append(p)
    return groups


def pick_seed_group(groups: Dict[str, List[Path]], prefer_prefix: Optional[str]) -> List[Path]:
    if not groups:
        return []
    if prefer_prefix:
        for key, runs in groups.items():
            if prefer_prefix in key or key == prefer_prefix:
                return sorted(runs)
    # Prefer group with most runs; tie-break by longest key (more specific name)
    best_key = max(groups.keys(), key=lambda k: (len(groups[k]), len(k)))
    return sorted(groups[best_key])


def find_single_run(base: Path, run_name: Optional[str] = None) -> Optional[Path]:
    if not base.exists():
        return None
    if run_name:
        cand = base / run_name
        if cand.is_dir() and (cand / "test_predictions.json").exists():
            return cand
    for suffix in ("_default", ""):
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            if p.name.endswith(suffix) and (p / "test_predictions.json").exists():
                if "_seeds_s" not in p.name:
                    return p
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "test_predictions.json").exists() and "_seeds_s" not in p.name:
            return p
    return None


def resolve_runs(
    repo: Path,
    model_id: str,
    protocol: str,
    seed_group_prefix: Optional[str],
    single_run_name: Optional[str],
) -> Tuple[List[Path], str]:
    base = repo / "data/processed/experiments" / model_id / protocol
    groups = group_seed_runs(base)
    multi = pick_seed_group(groups, seed_group_prefix if model_id == "cca" else None)
    if len(multi) >= 2:
        key = next(k for k, v in groups.items() if v == multi)
        return multi, f"multi-seed ({key}, n={len(multi)})"
    single = find_single_run(base, single_run_name)
    if single is not None:
        return [single], f"single ({single.name})"
    return [], "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--protocol", default="default")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cca", "qformer_adapter", "cbm_posthoc", "cbm_labelfree", "mlgcn"],
    )
    parser.add_argument("--reference", default="cca")
    parser.add_argument(
        "--cca_seed_group",
        default="lora_r8_trial27",
        help="Prefer this seed dir prefix under cca/ (e.g. lora_r8_trial27_seeds_s0).",
    )
    parser.add_argument("--out_md", default="reports/comparison/stats.md")
    args = parser.parse_args()

    repo = Path(args.repo)
    f1_rows: List[Tuple[str, float, float, float, int, str]] = []
    auroc_rows: List[Tuple[str, float, float, float, int, str]] = []
    ref_probs = ref_y = ref_m = None
    ref_label = ""

    for model_id in args.models:
        runs, label = resolve_runs(
            repo,
            model_id,
            args.protocol,
            args.cca_seed_group if model_id == "cca" else None,
            f"{model_id}_default" if model_id != "cca" else None,
        )
        f1s = []
        aucs = []
        for run_dir in runs:
            if not (run_dir / "test_predictions.json").exists():
                continue
            probs, y, m = load_predictions(run_dir)
            f1s.append(macro_f1(probs, y, m))
            aucs.append(macro_auroc(probs, y, m))
            if model_id == args.reference and ref_probs is None:
                ref_probs, ref_y, ref_m = probs, y, m
                ref_label = run_dir.name
        mean_f1, lo_f1, hi_f1 = bootstrap_ci(f1s)
        mean_auc, lo_auc, hi_auc = bootstrap_ci(aucs)
        f1_rows.append((model_id, mean_f1, lo_f1, hi_f1, len(f1s), label))
        auroc_rows.append((model_id, mean_auc, lo_auc, hi_auc, len(aucs), label))

    lines = [
        "# Multi-seed / baseline comparison",
        "",
        f"Protocol: `{args.protocol}`. CCA seed group: `{args.cca_seed_group}`.",
        "",
        "## Test macro-F1 @0.5 (bootstrap 95% CI)",
        "",
        "| Model | mean F1 | 95% CI | n | runs |",
        "|-------|---------|--------|---|------|",
    ]
    for model_id, mean, lo, hi, n, label in f1_rows:
        lines.append(f"| {model_id} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | {n} | {label} |")

    lines.extend(
        [
            "",
            "## Test macro-AUROC (bootstrap 95% CI)",
            "",
            "| Model | mean AUROC | 95% CI | n | runs |",
            "|-------|------------|--------|---|------|",
        ]
    )
    for model_id, mean, lo, hi, n, label in auroc_rows:
        lines.append(f"| {model_id} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | {n} | {label} |")

    if ref_probs is not None:
        lines.extend(
            [
                "",
                f"## Bootstrap AUROC vs `{args.reference}` (paired on test set; ref run `{ref_label}`)",
                "",
                "P-value: paired bootstrap on mean per-class AUROC (400 resamples). BH correction at q=0.05.",
                "",
                "| Model | Δ mean AUROC | p (bootstrap) | BH reject |",
                "|-------|--------------|---------------|-----------|",
            ]
        )
        cmp_rows: List[Tuple[str, float, float]] = []
        for model_id, _, _, _, _, _ in auroc_rows:
            if model_id == args.reference:
                continue
            runs, _ = resolve_runs(
                repo,
                model_id,
                args.protocol,
                args.cca_seed_group if model_id == "cca" else None,
                f"{model_id}_default" if model_id != "cca" else None,
            )
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
