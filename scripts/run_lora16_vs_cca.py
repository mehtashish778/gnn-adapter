#!/usr/bin/env python3
"""
Orchestrate Qwen2-VL LoRA-r16 training (cls + SFT), scoring, and comparison vs CCA.

Usage:
  python scripts/run_lora16_vs_cca.py --gpu_id 0
  python scripts/run_lora16_vs_cca.py --skip_train --report_only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_CCA_REF = _REPO / "data/processed/experiments/cca/default/lora_r8_trial27_seeds_s0"
_CCA_5SEED_F1 = (0.701, 0.005)
_CCA_5SEED_AUROC = (0.722, 0.004)
_CCA_PARAMS = 118_891


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    print({"run": " ".join(cmd)})
    subprocess.run(cmd, cwd=cwd or _REPO, check=True)


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pm(m: Dict[str, Any], key: str, default: str = "—") -> str:
    v = m.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return default
    return f"{float(v):.4f}"


def build_report(
    *,
    cca_ref_dir: Path,
    cls_dir: Optional[Path],
    sft_dir: Optional[Path],
    stats_md: Optional[Path],
    out_md: Path,
) -> None:
    cca_m = _load_metrics(cca_ref_dir)
    cls_m = _load_metrics(cls_dir) if cls_dir else {}
    sft_m = _load_metrics(sft_dir) if sft_dir else {}

    lines = [
        "# LoRA-16 (Qwen2-VL) vs CCA — CheXpert default",
        "",
        "Driver: `scripts/run_lora16_vs_cca.py`",
        "",
        "**CCA reference (5-seed leaderboard):** "
        f"F1 {_CCA_5SEED_F1[0]:.3f} ± {_CCA_5SEED_F1[1]:.3f}, "
        f"AUROC {_CCA_5SEED_AUROC[0]:.3f} ± {_CCA_5SEED_AUROC[1]:.3f}, "
        f"~{_CCA_PARAMS:,} trainable params (`cca_lora_r8_trial27`, seeds 0–4).",
        "",
        f"**Single-seed CCA ref run:** `{cca_ref_dir.relative_to(_REPO).as_posix()}`",
        "",
        "| Model | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Trainable params | GPU-hours |",
        "|-------|--------------|------------|------------|----------|------------|------------------|-----------|",
    ]

    def row(name: str, m: Dict[str, Any], run_label: str = "") -> str:
        params = m.get("trainable_params", "—")
        if isinstance(params, (int, float)):
            params = f"{int(params):,}"
        gh = m.get("gpu_hours")
        gh_s = f"{float(gh):.2f}" if gh is not None else "—"
        suffix = f" ({run_label})" if run_label else ""
        return (
            f"| {name}{suffix} | {_fmt_pm(m, 'test_macro_f1@0.5')} | "
            f"{_fmt_pm(m, 'test_macro_auroc')} | {_fmt_pm(m, 'test_macro_auprc')} | "
            f"{_fmt_pm(m, 'test_macro_ece')} | {_fmt_pm(m, 'test_macro_brier')} | {params} | {gh_s} |"
        )

    lines.append(
        row(
            "CCA (ref seed 0)",
            cca_m,
            cca_ref_dir.name,
        )
    )
    if cls_m:
        lines.append(row("Qwen2-VL LoRA-16 + cls head", cls_m, cls_dir.name if cls_dir else ""))
    else:
        lines.append("| Qwen2-VL LoRA-16 + cls head | — | — | — | — | — | — | — |")
    if sft_m:
        pf = sft_m.get("test_parse_failures", "—")
        lines.append(
            row(f"Qwen2-VL LoRA-16 + JSON SFT (parse fail test={pf})", sft_m, sft_dir.name if sft_dir else "")
        )
    else:
        lines.append("| Qwen2-VL LoRA-16 + JSON SFT | — | — | — | — | — | — | — |")

    # Δ vs CCA ref seed 0
    if cls_m and cca_m:
        df1 = float(cls_m.get("test_macro_f1@0.5", 0)) - float(cca_m.get("test_macro_f1@0.5", 0))
        dauc = float(cls_m.get("test_macro_auroc", 0)) - float(cca_m.get("test_macro_auroc", 0))
        lines.extend(
            [
                "",
                "## Δ vs CCA ref (seed 0, test)",
                "",
                f"- **LoRA-16 cls:** ΔF1 {df1:+.4f}, ΔAUROC {dauc:+.4f}",
            ]
        )
    if sft_m and cca_m:
        df1 = float(sft_m.get("test_macro_f1@0.5", 0)) - float(cca_m.get("test_macro_f1@0.5", 0))
        dauc = float(sft_m.get("test_macro_auroc", 0)) - float(cca_m.get("test_macro_auroc", 0))
        lines.append(f"- **LoRA-16 SFT:** ΔF1 {df1:+.4f}, ΔAUROC {dauc:+.4f}")

    if cls_m.get("trainable_params") and cca_m:
        ratio = _CCA_PARAMS / max(int(cls_m["trainable_params"]), 1)
        lines.extend(
            [
                "",
                "## Cost note",
                "",
                f"CCA uses ~{_CCA_PARAMS:,} adapter params; LoRA-16 cls uses ~{int(cls_m.get('trainable_params', 0)):,}. "
                f"CCA is ~{ratio*100:.2f}% of LoRA-16 cls trainable params (AAAI target: CCA < 0.1% of LoRA-16).",
            ]
        )

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "1. **Headline PEFT baseline:** LoRA-16 + classification head (masked BCE) is the primary comparison to CCA.",
            "2. **Protocol parity row:** JSON SFT matches the frozen zero-shot scoring path; parse failures are logged in `metrics.json`.",
            "3. **CCA 5-seed mean** remains the leaderboard bar; LoRA runs here are single-seed unless promoted via `scripts/run_seeds.py`.",
            "",
            "See also: [`docs/combined_experiments_report.md`](../docs/combined_experiments_report.md), "
            "[`docs/cca_experiment_results.md`](../docs/cca_experiment_results.md).",
        ]
    )

    if stats_md and stats_md.exists():
        lines.extend(["", "## stats_compare output", "", f"See [`{stats_md.relative_to(_REPO).as_posix()}`]({stats_md.relative_to(_REPO).as_posix()})."])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print({"wrote": str(out_md)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_sft", action="store_true", help="Skip generative SFT (faster).")
    parser.add_argument("--report_only", action="store_true", help="Only rebuild comparison markdown.")
    parser.add_argument("--run_id_cls", default="qwen2vl_lora_r16_default")
    parser.add_argument("--run_id_sft", default="qwen2vl_lora_r16_sft_default")
    parser.add_argument("--epochs_cls", type=int, default=3)
    parser.add_argument("--epochs_sft", type=int, default=1)
    parser.add_argument("--lr_cls", type=float, default=2e-5, help="LoRA cls learning rate (was 1e-3 via argparse default).")
    parser.add_argument("--max_train_samples_cls", type=int, default=0, help="Debug: cap cls train rows.")
    parser.add_argument("--max_train_samples_sft", type=int, default=0)
    parser.add_argument("--smoke_test", action="store_true", help="500 train rows, 1 cls epoch, skip sft.")
    args = parser.parse_args()
    max_val_cls = 0
    max_test_cls = 0
    if args.smoke_test:
        args.max_train_samples_cls = max(args.max_train_samples_cls, 500)
        max_val_cls = 500
        max_test_cls = 500
        args.epochs_cls = 1
        args.skip_sft = True
        args.run_id_cls = args.run_id_cls + "_smoke" if not args.run_id_cls.endswith("_smoke") else args.run_id_cls

    py = sys.executable
    cls_dir = _REPO / "data/processed/experiments/qwen2vl_lora_r16/default" / args.run_id_cls
    sft_dir = _REPO / "data/processed/experiments/qwen2vl_lora_r16_sft/default" / args.run_id_sft
    stats_md = _REPO / "reports/comparison/lora16_stats.md"
    out_md = _REPO / "reports/comparison/lora16_vs_cca.md"

    if not args.report_only:
        if not args.skip_train:
            cls_cmd = [
                py,
                str(_SCRIPTS / "train_qwen2vl_lora_cls.py"),
                "--model_id",
                "qwen2vl_lora_r16",
                "--protocol",
                "default",
                "--run_id",
                args.run_id_cls,
                "--gpu_id",
                str(args.gpu_id),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs_cls),
                "--batch_size",
                "1",
                "--grad_accum",
                "16",
                "--lr",
                str(args.lr_cls),
            ]
            if args.max_train_samples_cls > 0:
                cls_cmd.extend(["--max_train_samples", str(args.max_train_samples_cls)])
            if max_val_cls > 0:
                cls_cmd.extend(["--max_val_samples", str(max_val_cls)])
            if max_test_cls > 0:
                cls_cmd.extend(["--max_test_samples", str(max_test_cls)])
            _run(cls_cmd)
        elif not (cls_dir / "test_predictions.json").exists():
            _run(
                [
                    py,
                    str(_SCRIPTS / "score_qwen2vl_lora.py"),
                    "--run_dir",
                    str(cls_dir),
                    "--mode",
                    "cls",
                    "--model_id",
                    "qwen2vl_lora_r16",
                    "--protocol",
                    "default",
                    "--gpu_id",
                    str(args.gpu_id),
                ]
            )

        if not args.skip_sft and not args.skip_train:
            sft_cmd = [
                py,
                str(_SCRIPTS / "train_qwen2vl_lora_sft.py"),
                "--model_id",
                "qwen2vl_lora_r16_sft",
                "--protocol",
                "default",
                "--run_id",
                args.run_id_sft,
                "--gpu_id",
                str(args.gpu_id),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs_sft),
                "--batch_size",
                "1",
                    "--grad_accum",
                    "16",
                    "--lr",
                    str(args.lr_cls),
            ]
            if args.max_train_samples_sft > 0:
                sft_cmd.extend(["--max_train_samples", str(args.max_train_samples_sft)])
            _run(sft_cmd)

    if args.smoke_test:
        print(
            {
                "skip_stats_compare": True,
                "reason": "smoke uses capped val/test; stats_compare needs full paired splits",
            }
        )
    else:
        _run(
            [
                py,
                str(_SCRIPTS / "stats_compare.py"),
                "--repo",
                str(_REPO),
                "--protocol",
                "default",
                "--models",
                "cca",
                "qwen2vl_lora_r16",
                "qwen2vl_lora_r16_sft",
                "--reference",
                "cca",
                "--cca_seed_group",
                "lora_r8_trial27",
                "--out_md",
                str(stats_md),
            ]
        )

    build_report(
        cca_ref_dir=_CCA_REF,
        cls_dir=cls_dir if cls_dir.exists() else None,
        sft_dir=sft_dir if sft_dir.exists() and not args.skip_sft else None,
        stats_md=stats_md,
        out_md=out_md,
    )


if __name__ == "__main__":
    main()
