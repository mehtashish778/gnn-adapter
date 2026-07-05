#!/usr/bin/env python3
"""Bar charts: Qwen2 vs Qwen3.5-2B on CheXpert (fair splits) and NIH cross-site (6k)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports" / "comparison" / "figures"

CHEXPERT_PATHS = {
    "Frozen VLM": {
        "Qwen2": REPO / "data/processed/splits/test_rows.json",
        "Qwen3.5-2B": REPO / "data/processed/splits/qwen35_qwen2_splits/test_rows.json",
    },
    "CBM post-hoc": {
        "Qwen2": REPO
        / "data/processed/experiments/cbm_posthoc/default/cbm_posthoc_default",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cbm_posthoc/default/cbm_posthoc_qwen35_qwen2_splits",
    },
    "CBM label-free": {
        "Qwen2": REPO
        / "data/processed/experiments/cbm_labelfree/default/cbm_labelfree_default",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cbm_labelfree/default/cbm_labelfree_qwen35_qwen2_splits",
    },
    "CCA": {
        "Qwen2": REPO / "data/processed/experiments/cca/default/cca_faithful",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cca/qwen35_qwen2_splits/cca_qwen35_vllm_2b_qwen2_splits",
    },
}

NIH_PATHS = {
    "Frozen VLM": {
        "Qwen2": REPO / "data/processed/experiments/vlm_zeroshot/nih/crosssite_eval",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/vlm_zeroshot/nih/crosssite_eval_qwen35_2b",
    },
    "CBM post-hoc": {
        "Qwen2": REPO / "data/processed/experiments/cbm_posthoc/nih/crosssite_eval",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cbm_posthoc/nih/crosssite_eval_qwen35_2b",
    },
    "CBM label-free": {
        "Qwen2": REPO / "data/processed/experiments/cbm_labelfree/nih/crosssite_eval",
        "Qwen3.5-2B": REPO
        / "data/processed/experiments/cbm_labelfree/nih/crosssite_eval_qwen35_2b",
    },
    "CCA": {
        "Qwen2": REPO / "data/processed/experiments/cca/nih/crosssite_eval",
        "Qwen3.5-2B": REPO / "data/processed/experiments/cca/nih/crosssite_eval_qwen35_2b",
    },
}


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _macro_auroc(probs: np.ndarray, y_true: np.ndarray, y_mask: np.ndarray) -> float:
    aurocs: list[float] = []
    for i in range(probs.shape[1]):
        m = y_mask[:, i] > 0.5
        if m.sum() < 2:
            continue
        p = np.clip(np.nan_to_num(probs[m, i], nan=0.5), 0.0, 1.0)
        y = y_true[m, i]
        n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
        if n_pos > 0 and n_neg > 0:
            aurocs.append(float(roc_auc_score(y, p)))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _f1_from_arrays(
    probs: np.ndarray, y_true: np.ndarray, y_mask: np.ndarray, threshold: float = 0.5
) -> float:
    pred = (probs >= threshold).astype(np.float64)
    f1s: list[float] = []
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


def _arrays_from_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _load_json(path)["rows"]
    probs = np.array([r["x_probs"] for r in rows], dtype=np.float64)
    y_true = np.array([r.get("y_true", r.get("y")) for r in rows], dtype=np.float64)
    n_classes = probs.shape[1]
    y_mask = np.array(
        [r.get("y_mask", [1] * n_classes) for r in rows],
        dtype=np.float64,
    )
    return probs, y_true, y_mask


def _arrays_from_preds(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_json(path)
    return (
        np.array(data["probs"], dtype=np.float64),
        np.array(data["y_true"], dtype=np.float64),
        np.array(data["y_mask"], dtype=np.float64),
    )


def _metric_from_source(source: Path, key: str) -> float:
    if source.suffix == ".json":
        data = _load_json(source)
        if source.name.endswith("rows.json"):
            probs, y_true, y_mask = _arrays_from_rows(source)
        else:
            probs, y_true, y_mask = _arrays_from_preds(source)
    elif (source / "test_predictions.json").exists():
        probs, y_true, y_mask = _arrays_from_preds(source / "test_predictions.json")
    elif (source / "test_rows.json").exists():
        probs, y_true, y_mask = _arrays_from_rows(source / "test_rows.json")
    else:
        raise FileNotFoundError(f"No predictions for {source}")

    if key == "test_macro_f1@0.5":
        metrics_path = source / "metrics.json" if source.is_dir() else None
        if metrics_path and metrics_path.exists():
            val = _load_json(metrics_path).get(key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
        return _f1_from_arrays(probs, y_true, y_mask)
    if key == "test_macro_auroc":
        metrics_path = source / "metrics.json" if source.is_dir() else None
        if metrics_path and metrics_path.exists():
            val = _load_json(metrics_path).get(key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
        return _macro_auroc(probs, y_true, y_mask)
    raise KeyError(key)


def _collect(paths: dict[str, dict[str, Path]], metric_key: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model, backends in paths.items():
        out[model] = {}
        for backend, source in backends.items():
            out[model][backend] = _metric_from_source(source, metric_key)
    return out


def _grouped_bars(
    ax: plt.Axes,
    data: dict[str, dict[str, float]],
    title: str,
    ylabel: str,
    *,
    y_pad: float = 0.008,
    y_max_scale: float = 1.18,
) -> None:
    models = list(data.keys())
    backends = ["Qwen2", "Qwen3.5-2B"]
    x = np.arange(len(models))
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    for i, backend in enumerate(backends):
        vals = [data[m][backend] for m in models]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=backend, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymax = max(max(v for v in d.values()) for d in data.values())
    ax.set_ylim(0, ymax * y_max_scale)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")


def _delta_chart(
    chex: dict[str, dict[str, float]],
    nih: dict[str, dict[str, float]],
    *,
    ylabel: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    models = list(chex.keys())
    deltas_chex = [chex[m]["Qwen3.5-2B"] - chex[m]["Qwen2"] for m in models]
    deltas_nih = [nih[m]["Qwen3.5-2B"] - nih[m]["Qwen2"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, deltas_chex, w, label="CheXpert Δ", color="#55A868")
    ax.bar(x + w / 2, deltas_nih, w, label="NIH Δ", color="#C44E52")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Qwen3.5 gain over Qwen2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"{stem}_delta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _comparison_figure(
    chex: dict[str, dict[str, float]],
    nih: dict[str, dict[str, float]],
    *,
    suptitle: str,
    ylabel: str,
    stem: str,
    y_pad: float = 0.008,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _grouped_bars(
        axes[0],
        chex,
        "CheXpert in-domain (fair splits, n=9,197 test)",
        ylabel,
        y_pad=y_pad,
    )
    _grouped_bars(
        axes[1],
        nih,
        "NIH cross-site (same 6k images, CheXpert-trained)",
        ylabel,
        y_pad=y_pad,
    )
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    png = OUT_DIR / f"{stem}_comparison.png"
    pdf = OUT_DIR / f"{stem}_comparison.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chex_f1 = _collect(CHEXPERT_PATHS, "test_macro_f1@0.5")
    nih_f1 = _collect(NIH_PATHS, "test_macro_f1@0.5")
    chex_auroc = _collect(CHEXPERT_PATHS, "test_macro_auroc")
    nih_auroc = _collect(NIH_PATHS, "test_macro_auroc")

    _comparison_figure(
        chex_f1,
        nih_f1,
        suptitle="Qwen2 vs Qwen3.5-2B — Macro F1 @0.5",
        ylabel="Macro F1 @0.5",
        stem="qwen2_vs_qwen35_f1",
    )
    _delta_chart(chex_f1, nih_f1, ylabel="Δ Macro F1 (Qwen3.5 − Qwen2)", stem="qwen2_vs_qwen35_f1")

    _comparison_figure(
        chex_auroc,
        nih_auroc,
        suptitle="Qwen2 vs Qwen3.5-2B — Macro AUROC",
        ylabel="Macro AUROC",
        stem="qwen2_vs_qwen35_auroc",
        y_pad=0.012,
    )
    _delta_chart(
        chex_auroc,
        nih_auroc,
        ylabel="Δ Macro AUROC (Qwen3.5 − Qwen2)",
        stem="qwen2_vs_qwen35_auroc",
    )

    summary = {
        "f1": {"chexpert": chex_f1, "nih": nih_f1},
        "auroc": {"chexpert": chex_auroc, "nih": nih_auroc},
    }
    summary_path = OUT_DIR / "qwen2_vs_qwen35_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
