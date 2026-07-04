#!/usr/bin/env python3
"""
NIH ChestX-ray14 cross-site pipeline: data prep, VLM scoring, patch caches, all-model eval, report.
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

from crosssite_common import ALL_CROSSSITE_MODELS, CHEXPERT_RUN_NAMES  # noqa: E402


def _run(cmd: List[str]) -> None:
    print({"run": " ".join(cmd)})
    subprocess.run(cmd, cwd=_REPO, check=True)


def _run_eval(cmd: List[str]) -> bool:
    """Run an eval subprocess; return False (skip) on missing checkpoint, reraise other errors."""
    print({"run": " ".join(cmd)})
    result = subprocess.run(cmd, cwd=_REPO)
    if result.returncode == 0:
        return True
    # Re-run capturing stderr to detect missing-checkpoint errors vs real bugs
    probe = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if "FileNotFoundError" in probe.stderr or "No CheXpert run dir" in probe.stderr:
        print({"skip": cmd[cmd.index("--model_id") + 1], "reason": "missing CheXpert checkpoint"})
        return False
    raise subprocess.CalledProcessError(result.returncode, cmd)


def _rows_have_vlm(test_rows: Path) -> bool:
    if not test_rows.exists():
        return False
    with test_rows.open("r", encoding="utf-8") as f:
        rows = json.load(f).get("rows", [])
    return bool(rows) and "x_probs" in rows[0]


def build_crosssite_report(
    *,
    repo: Path,
    protocol: str,
    run_id: str,
    models: List[str],
    out_md: Path,
    n_test: int,
    smoke: bool,
    max_samples: int = 0,
) -> None:
    lines = [
        "# NIH ChestX-ray14 cross-site evaluation",
        "",
        f"Protocol: `{protocol}`. Train: CheXpert only. Test: NIH ({n_test:,} images"
        + (", **smoke subset**" if smoke else "")
        + (f", **subset cap {max_samples:,}**" if max_samples > 0 and not smoke else "")
        + ").",
        "",
        "| Model | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Trainable params |",
        "|-------|--------------|------------|------------|----------|------------|------------------|",
    ]

    def load_row(model_id: str) -> str:
        base = repo / "data/processed/experiments" / model_id / protocol
        cand = base / run_id
        if not (cand / "metrics.json").exists():
            for p in sorted(base.iterdir()) if base.exists() else []:
                if p.is_dir() and (p / "metrics.json").exists():
                    cand = p
                    break
        mpath = cand / "metrics.json"
        if not mpath.exists():
            return f"| {model_id} | — | — | — | — | — | — |"
        with mpath.open("r", encoding="utf-8") as f:
            m = json.load(f)
        params = m.get("trainable_params", "—")
        if isinstance(params, (int, float)):
            params = f"{int(params):,}"

        def fmt(key: str) -> str:
            v = m.get(key)
            if v is None:
                return "—"
            return f"{float(v):.4f}"

        return (
            f"| {model_id} | {fmt('test_macro_f1@0.5')} | {fmt('test_macro_auroc')} | "
            f"{fmt('test_macro_auprc')} | {fmt('test_macro_ece')} | {fmt('test_macro_brier')} | {params} |"
        )

    for mid in models:
        lines.append(load_row(mid))

    cca_m = {}
    lora_m = {}
    cca_p = repo / "data/processed/experiments/cca" / protocol / run_id / "metrics.json"
    lora_p = repo / "data/processed/experiments/qwen2vl_lora_r16" / protocol / run_id / "metrics.json"
    if cca_p.exists():
        with cca_p.open("r", encoding="utf-8") as f:
            cca_m = json.load(f)
    if lora_p.exists():
        with lora_p.open("r", encoding="utf-8") as f:
            lora_m = json.load(f)
    if cca_m and lora_m:
        df1 = float(lora_m.get("test_macro_f1@0.5", 0)) - float(cca_m.get("test_macro_f1@0.5", 0))
        dauc = float(lora_m.get("test_macro_auroc", 0)) - float(cca_m.get("test_macro_auroc", 0))
        lines.extend(
            [
                "",
                "## Headline pair (LoRA-16 cls vs CCA trial-27)",
                "",
                f"- ΔF1 (LoRA − CCA): {df1:+.4f}",
                f"- ΔAUROC (LoRA − CCA): {dauc:+.4f}",
            ]
        )
        tp = cca_m.get("trainable_params")
        lp = lora_m.get("trainable_params")
        if tp and lp:
            ratio = float(tp) / max(float(lp), 1) * 100
            lines.append(
                f"- CCA params are ~{ratio:.2f}% of LoRA-16 cls ({int(tp):,} vs {int(lp):,})."
            )

    lines.extend(
        [
            "",
            "Driver: `scripts/run_crosssite_nih.py`",
            "",
            "See also: [`reports/comparison/crosssite_nih_stats.md`](crosssite_nih_stats.md).",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print({"wrote": str(out_md)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Cap at --max_samples (default 500).")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit NIH rows (0 = all). Use with --random_sample for a random subset.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample --max_samples rows (passed to 01_build_canonical_labels_nih.py).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed when --random_sample is set.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_vlm", action="store_true")
    parser.add_argument("--skip_patches", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Models to eval: all | {' '.join(ALL_CROSSSITE_MODELS)}",
    )
    parser.add_argument("--include_sft", action="store_true")
    parser.add_argument("--nih_root", default="data/raw/nih_chestxray14")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--run_id", default="crosssite_eval")
    args = parser.parse_args()

    py = sys.executable
    max_n = args.max_samples
    if args.smoke and max_n == 0:
        max_n = 500
    subset_suffix = f"_n{max_n}" if max_n > 0 else ""
    test_rows = _REPO / f"data/processed/splits/nih/test_rows{subset_suffix}.json"
    canonical = _REPO / f"data/processed/multilabel/nih/canonical_labels{subset_suffix}.json"
    aligned = _REPO / f"data/processed/multilabel/nih/aligned_vlm_targets{subset_suffix}.json"
    vlm_dir = _REPO / f"data/outputs_vlm_nih{subset_suffix}"
    protocol = "nih"

    if args.models == ["all"]:
        models = list(ALL_CROSSSITE_MODELS)
        if args.include_sft:
            models.append("qwen2vl_lora_r16_sft")
    else:
        models = args.models

    nih_csv = _REPO / args.nih_root / "Data_Entry_2017.csv"
    if not args.skip_data and not nih_csv.is_file():
        raise FileNotFoundError(
            f"NIH CSV not found at {nih_csv}. Place ChestX-ray14 under {args.nih_root} "
            "(Data_Entry_2017.csv + images_* shards) or use --skip_data with existing test_rows."
        )

    if not args.skip_data:
        cmd = [py, str(_SCRIPTS / "01_build_canonical_labels_nih.py"), "--nih_root", args.nih_root]
        if max_n > 0:
            cmd.extend(["--max_samples", str(max_n)])
        if args.random_sample:
            cmd.extend(["--random_sample", "--seed", str(args.seed)])
        cmd.extend(["--out_json", str(canonical)])
        _run(cmd)
        val_rows = _REPO / f"data/processed/splits/nih/val_rows{subset_suffix}.json"
        _run(
            [
                py,
                str(_SCRIPTS / "03_make_multilabel_splits_nih.py"),
                "--canonical_json",
                str(canonical),
                "--test_rows_json",
                str(test_rows),
                "--val_rows_json",
                str(val_rows),
            ]
        )

    if not args.skip_vlm and not _rows_have_vlm(test_rows):
        cmd = [
            py,
            str(_SCRIPTS / "04_score_frozen_vlm_batch.py"),
            "--canonical_json",
            str(canonical),
            "--image_root",
            args.image_root,
            "--out_dir",
            str(vlm_dir),
            "--gpu_id",
            str(args.gpu_id),
        ]
        if max_n > 0:
            cmd.extend(["--max_samples", str(max_n)])
        _run(cmd)
        _run(
            [
                py,
                str(_SCRIPTS / "02_align_vlm_outputs.py"),
                "--canonical_json",
                str(canonical),
                "--vlm_dir",
                str(vlm_dir),
                "--out_json",
                str(aligned),
            ]
        )
        _run(
            [
                py,
                str(_SCRIPTS / "build_nih_test_rows.py"),
                "--canonical_json",
                str(canonical),
                "--aligned_json",
                str(aligned),
                "--out_json",
                str(test_rows),
            ]
        )

    n_test = 0
    if test_rows.exists():
        with test_rows.open("r", encoding="utf-8") as f:
            n_test = len(json.load(f).get("rows", []))

    lora_adapter = _REPO / "data/processed/embeddings/lora_r8_adapter"
    if not args.skip_patches and test_rows.exists():
        _run(
            [
                py,
                str(_SCRIPTS / "precompute_patch_cache.py"),
                "--rows_json",
                str(test_rows),
                "--image_root",
                args.image_root,
                "--protocol",
                protocol,
                "--split_name",
                "test",
                "--gpu_id",
                str(args.gpu_id),
            ]
        )
        if lora_adapter.is_dir():
            _run(
                [
                    py,
                    str(_SCRIPTS / "precompute_patch_cache.py"),
                    "--rows_json",
                    str(test_rows),
                    "--image_root",
                    args.image_root,
                    "--protocol",
                    protocol,
                    "--split_name",
                    "test",
                    "--lora_rank",
                    "8",
                    "--lora_adapter_dir",
                    str(lora_adapter),
                    "--gpu_id",
                    str(args.gpu_id),
                ]
            )

    if not args.skip_eval and test_rows.exists():
        eval_models = list(models)
        if "vlm_zeroshot" in eval_models:
            _run(
                [
                    py,
                    str(_SCRIPTS / "05_run_baseline_frozen_vlm.py"),
                    "--model_id",
                    "vlm_zeroshot",
                    "--protocol",
                    protocol,
                    "--run_id",
                    args.run_id,
                    "--test_rows_json",
                    str(test_rows),
                    "--skip_val",
                ]
            )
            eval_models = [m for m in eval_models if m != "vlm_zeroshot"]

        if "qwen2vl_lora_r16" in eval_models:
            ckpt = CHEXPERT_RUN_NAMES.get("qwen2vl_lora_r16", "qwen2vl_lora_r16_v2")
            _run(
                [
                    py,
                    str(_SCRIPTS / "score_qwen2vl_lora.py"),
                    "--model_id",
                    "qwen2vl_lora_r16",
                    "--protocol",
                    protocol,
                    "--out_run_id",
                    args.run_id,
                    "--checkpoint_run_dir",
                    str(_REPO / "data/processed/experiments/qwen2vl_lora_r16/default" / ckpt),
                    "--test_rows_json",
                    str(test_rows),
                    "--image_root",
                    args.image_root,
                    "--skip_val",
                    "--gpu_id",
                    str(args.gpu_id),
                ]
            )
            eval_models = [m for m in eval_models if m != "qwen2vl_lora_r16"]

        for model_id in eval_models:
            if model_id in ("vlm_zeroshot", "qwen2vl_lora_r16", "qwen2vl_lora_r16_sft"):
                continue
            chex_run = CHEXPERT_RUN_NAMES.get(model_id, f"{model_id}_default")
            _run_eval(
                [
                    py,
                    str(_SCRIPTS / "eval_crosssite.py"),
                    "--model_id",
                    model_id,
                    "--chexpert_run_dir",
                    str(_REPO / "data/processed/experiments" / model_id / "default" / chex_run),
                    "--test_rows_json",
                    str(test_rows),
                    "--image_root",
                    args.image_root,
                    "--protocol",
                    protocol,
                    "--run_id",
                    args.run_id,
                    "--gpu_id",
                    str(args.gpu_id),
                ]
            )

    if not args.skip_report:
        stats_models = [m for m in (args.models if args.models != ["all"] else ALL_CROSSSITE_MODELS)]
        _run(
            [
                py,
                str(_SCRIPTS / "stats_compare.py"),
                "--repo",
                str(_REPO),
                "--protocol",
                protocol,
                "--models",
                *stats_models,
                "--reference",
                "cca",
                "--cca_seed_group",
                "lora_r8_trial27",
                "--out_md",
                str(_REPO / "reports/comparison/crosssite_nih_stats.md"),
            ]
        )
        build_crosssite_report(
            repo=_REPO,
            protocol=protocol,
            run_id=args.run_id,
            models=stats_models,
            out_md=_REPO / "reports/comparison/crosssite_nih.md",
            n_test=n_test,
            smoke=args.smoke,
            max_samples=max_n,
        )


if __name__ == "__main__":
    main()
