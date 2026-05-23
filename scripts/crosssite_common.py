"""Shared helpers for cross-site (NIH) evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from common_multilabel import (
    load_per_class_thresholds,
    load_rows,
    masked_macro_f1,
    probabilistic_metrics,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from stats_compare import find_single_run

_REPO = Path(__file__).resolve().parents[1]

# CheXpert default run dir names (under experiments/{model_id}/default/)
CHEXPERT_RUN_NAMES: Dict[str, str] = {
    "cca": "lora_r8_trial27_seeds_s0",
    "vlm_mlp": "vlm_mlp_default",
    "cbm_posthoc": "cbm_posthoc_default",
    "cbm_labelfree": "cbm_labelfree_default",
    "mlgcn": "mlgcn_default",
    "gnn07_label_residual": "gnn07_label_residual_default",
    "qformer_adapter": "qformer_adapter_default",
    "gnn12_clip_vlm_homo": "gnn12_clip_vlm_homo_default",
    "gnn13_clip_bipartite": "repro_full_20260503",
    "qwen2vl_lora_r16": "qwen2vl_lora_r16_v2",
}

ALL_CROSSSITE_MODELS: List[str] = [
    "vlm_zeroshot",
    "vlm_mlp",
    "cbm_posthoc",
    "mlgcn",
    "gnn07_label_residual",
    "cca",
    "qformer_adapter",
    "cbm_labelfree",
    "gnn12_clip_vlm_homo",
    "gnn13_clip_bipartite",
    "qwen2vl_lora_r16",
]


def resolve_chexpert_run_dir(model_id: str, chexpert_run: Optional[str] = None) -> Path:
    if chexpert_run:
        p = Path(chexpert_run)
        if p.is_dir():
            return p.resolve()
        base = _REPO / "data/processed/experiments" / model_id / "default"
        cand = base / chexpert_run
        if cand.is_dir():
            return cand.resolve()
    base = _REPO / "data/processed/experiments" / model_id / "default"
    name = CHEXPERT_RUN_NAMES.get(model_id)
    if name:
        cand = base / name
        if cand.is_dir():
            return cand.resolve()
    found = find_single_run(base, f"{model_id}_default")
    if found is not None:
        return found.resolve()
    raise FileNotFoundError(f"No CheXpert run dir for model_id={model_id} under {base}")


def load_metrics_hparams(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    hp = data.get("hparams")
    if isinstance(hp, dict):
        return hp
    return data


def load_checkpoint_state(run_dir: Path) -> Dict[str, torch.Tensor]:
    ckpt_path = run_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    if isinstance(state, dict) and "adapter_state_dict" in state:
        return state["adapter_state_dict"]
    return state


@torch.no_grad()
def write_crosssite_eval(
    *,
    model_id: str,
    protocol: str,
    run_id: str,
    probs: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    trainable_params: Optional[int] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
    chexpert_run_dir: Optional[Path] = None,
) -> Path:
    out_dir = resolve_experiment_dir(
        out_dir=None,
        model_id=model_id,
        protocol=protocol,
        run_id=run_id,
        default_legacy_out_dir=f"data/processed/experiments/{model_id}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = probs.cpu()
    y_true = y_true.cpu()
    y_mask = y_mask.cpu()

    test_f1 = masked_macro_f1(probs, y_true, y_mask, threshold=0.5)
    pm = probabilistic_metrics(probs, y_true, y_mask)

    thr_path = _REPO / "data/processed/experiments/thresholds/per_class_thresholds.json"
    thr_list = load_per_class_thresholds(thr_path)

    metrics: Dict[str, Any] = {
        "variant": model_id,
        "protocol": protocol,
        "cross_site": True,
        "chexpert_run_dir": str(chexpert_run_dir) if chexpert_run_dir else None,
        "test_macro_f1@0.5": float(test_f1),
        "test_macro_auroc": pm["macro_auroc"],
        "test_macro_auprc": pm["macro_auprc"],
        "test_macro_ece": pm["macro_ece"],
        "test_macro_brier": pm["macro_brier"],
    }
    if thr_list and len(thr_list) == probs.shape[1]:
        metrics["test_macro_f1@per_class_thr"] = float(
            masked_macro_f1(probs, y_true, y_mask, threshold=thr_list)
        )
    if trainable_params is not None:
        metrics["trainable_params"] = int(trainable_params)
    if extra_metrics:
        metrics.update(extra_metrics)

    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "test_predictions.json",
        {
            "probs": probs.cpu().tolist(),
            "y_true": y_true.cpu().tolist(),
            "y_mask": y_mask.cpu().tolist(),
        },
    )
    update_run_registry(
        model_id=model_id,
        protocol=protocol,
        run_dir=out_dir,
        metrics={
            "test_macro_f1@0.5": metrics["test_macro_f1@0.5"],
            "test_macro_auroc": metrics["test_macro_auroc"],
        },
    )
    return out_dir


def require_vlm_scores(rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("Empty row list")
    if "x_probs" not in rows[0] or "x_logits" not in rows[0]:
        raise RuntimeError(
            "Rows missing x_probs/x_logits. Run 04_score_frozen_vlm_batch.py and build_nih_test_rows.py first."
        )
