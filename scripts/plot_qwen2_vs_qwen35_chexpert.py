#!/usr/bin/env python3
"""Generate CheXpert comparison figures from qwen2_vs_qwen35_chexpert_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
METRICS_JSON = REPO / "reports/comparison/qwen2_vs_qwen35_chexpert_metrics.json"
OUT_DIR = REPO / "reports/comparison/figures"

METHODS = ["Frozen VLM", "CBM post-hoc", "CBM label-free", "CCA", "LoRA r16"]
BACKENDS = ["Qwen2", "Qwen3.5-2B"]
COLORS = {"Qwen2": "#4C72B0", "Qwen3.5-2B": "#DD8452"}


def _load() -> dict:
    with METRICS_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def _get(data: dict, method: str, backend: str, key: str) -> float:
    val = data[method][backend].get(key)
    if val is None:
        return float("nan")
    return float(val)


def _grouped_bars(
    data: dict,
    metric_key: str,
    title: str,
    ylabel: str,
    stem: str,
    *,
    higher_is_better: bool = True,
    y_pad: float = 0.008,
) -> None:
    x = np.arange(len(METHODS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, backend in enumerate(BACKENDS):
        vals = [_get(data, m, backend, metric_key) for m in METHODS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=backend, color=COLORS[backend])
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finite = [
        _get(data, m, b, metric_key)
        for m in METHODS
        for b in BACKENDS
        if not np.isnan(_get(data, m, b, metric_key))
    ]
    if finite:
        lo, hi = min(finite), max(finite)
        if higher_is_better:
            ax.set_ylim(max(0, lo - 0.05), hi * 1.15)
        else:
            ax.set_ylim(0, hi * 1.25)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


def _delta_chart(data: dict, metric_key: str, ylabel: str, stem: str) -> None:
    deltas = [
        _get(data, m, "Qwen3.5-2B", metric_key) - _get(data, m, "Qwen2", metric_key)
        for m in METHODS
    ]
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, deltas, color=colors)
    for bar, val in zip(bars, deltas):
        va = "bottom" if val >= 0 else "top"
        offset = 0.005 if val >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset, f"{val:+.3f}", ha="center", va=va, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Qwen3.5 gain over Qwen2 — {ylabel}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _multi_panel(data: dict) -> None:
    panels = [
        ("test_macro_f1@0.5", "Macro F1 @0.5"),
        ("test_macro_auroc", "Macro AUROC"),
        ("test_macro_auprc", "Macro AUPRC"),
        ("test_subset_accuracy@0.5", "Subset accuracy @0.5"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(METHODS))
    width = 0.35

    for ax, (key, label) in zip(axes.ravel(), panels):
        for i, backend in enumerate(BACKENDS):
            vals = [_get(data, m, backend, key) for m in METHODS]
            ax.bar(x + (i - 0.5) * width, vals, width, label=backend, color=COLORS[backend])
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        finite = [_get(data, m, b, key) for m in METHODS for b in BACKENDS if not np.isnan(_get(data, m, b, key))]
        if finite:
            ax.set_ylim(0, max(finite) * 1.12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Qwen2-VL vs Qwen3.5-2B — CheXpert test (n=9,197, fair splits)", fontsize=13, y=1.04)
    fig.tight_layout()
    out = OUT_DIR / "qwen2_vs_qwen35_chexpert_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()

    _grouped_bars(
        data,
        "test_macro_f1@0.5",
        "CheXpert test — Macro F1 @0.5 (fair splits, n=9,197)",
        "Macro F1 @0.5",
        "qwen2_vs_qwen35_chexpert_f1",
    )
    _grouped_bars(
        data,
        "test_macro_auroc",
        "CheXpert test — Macro AUROC",
        "Macro AUROC",
        "qwen2_vs_qwen35_chexpert_auroc",
        y_pad=0.012,
    )
    _grouped_bars(
        data,
        "test_macro_auprc",
        "CheXpert test — Macro AUPRC",
        "Macro AUPRC",
        "qwen2_vs_qwen35_chexpert_auprc",
        y_pad=0.012,
    )
    _grouped_bars(
        data,
        "test_macro_ece",
        "CheXpert test — Macro ECE (lower is better)",
        "Macro ECE",
        "qwen2_vs_qwen35_chexpert_ece",
        higher_is_better=False,
    )
    _grouped_bars(
        data,
        "test_macro_brier",
        "CheXpert test — Macro Brier (lower is better)",
        "Macro Brier",
        "qwen2_vs_qwen35_chexpert_brier",
        higher_is_better=False,
    )
    _delta_chart(data, "test_macro_f1@0.5", "Δ Macro F1 (Qwen3.5 − Qwen2)", "qwen2_vs_qwen35_chexpert_f1_delta")
    _delta_chart(data, "test_macro_auroc", "Δ Macro AUROC (Qwen3.5 − Qwen2)", "qwen2_vs_qwen35_chexpert_auroc_delta")
    _multi_panel(data)


if __name__ == "__main__":
    main()
