"""
Unified training loop for tensor-batch adapter models (e.g. VLM MLP baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from common_multilabel import (
    compute_pos_weight,
    masked_macro_f1,
    masked_bce_with_logits,
    masked_subset_accuracy,
    set_seed,
    write_json,
)
from model_registry import update_run_registry


@dataclass
class TrainingConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gpu_id: int = 0
    seed: int = 42
    run_dir: Path = field(default_factory=lambda: Path("."))
    pos_weight_max: float = 100.0
    best_metric: str = "val_macro_f1@0.5"  # maximize val macro-F1 @0.5


def write_predictions_json(
    path: Path,
    probs: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
) -> None:
    write_json(
        path,
        {
            "probs": probs.detach().cpu().tolist(),
            "y_true": y_true.detach().cpu().tolist(),
            "y_mask": y_mask.detach().cpu().tolist(),
        },
    )


def run_training_loop(
    model: nn.Module,
    *,
    train_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    calib_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    config: TrainingConfig,
    model_id: str = "",
    protocol: str = "",
    hparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train a model on in-memory tensors; select best checkpoint by val macro-F1 @0.5.

    ``train_batch`` / ``val_batch`` / ``test_batch`` are ``(x, y_true, y_mask)`` tuples.
    ``forward_fn(model, x)`` must return logits of shape ``(N, C)``.
    """
    from common_multilabel import require_cuda_device

    set_seed(config.seed)
    device = require_cuda_device(config.gpu_id)
    config.run_dir = Path(config.run_dir)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    xtr, ytr, mtr = (t.to(device) for t in train_batch)
    xva, yva, mva = (t.to(device) for t in val_batch)
    xte, yte, mte = (t.to(device) for t in test_batch)
    xca = yca = mca = None
    if calib_batch is not None:
        xca, yca, mca = (t.to(device) for t in calib_batch)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    pos_weight = compute_pos_weight(ytr, mtr, max_weight=config.pos_weight_max)

    best_val_f1 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for _ in range(config.epochs):
        model.train()
        logits = forward_fn(model, xtr)
        loss = masked_bce_with_logits(logits, ytr, mtr, pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = forward_fn(model, xva)
            val_prob = torch.sigmoid(val_logits)
            val_f1 = masked_macro_f1(val_prob, yva, mva, threshold=0.5)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(forward_fn(model, xva))
        test_prob = torch.sigmoid(forward_fn(model, xte))

    val_subset = masked_subset_accuracy(val_prob, yva, mva, threshold=0.5)
    test_f1 = masked_macro_f1(test_prob, yte, mte, threshold=0.5)
    test_subset = masked_subset_accuracy(test_prob, yte, mte, threshold=0.5)

    write_predictions_json(config.run_dir / "val_predictions.json", val_prob, yva, mva)
    write_predictions_json(config.run_dir / "test_predictions.json", test_prob, yte, mte)
    if xca is not None:
        with torch.no_grad():
            calib_prob = torch.sigmoid(forward_fn(model, xca))
        write_predictions_json(config.run_dir / "calib_predictions.json", calib_prob, yca, mca)

    metrics = {
        "best_val_macro_f1": best_val_f1,
        "val_macro_f1@0.5": best_val_f1,
        "val_subset_accuracy@0.5": val_subset,
        "test_macro_f1@0.5": test_f1,
        "test_subset_accuracy@0.5": test_subset,
    }
    write_json(config.run_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), config.run_dir / "best_checkpoint.pt")

    if model_id and protocol:
        update_run_registry(
            model_id=model_id,
            protocol=protocol,
            run_dir=config.run_dir,
            metrics={
                "val_macro_f1@0.5": best_val_f1,
                "test_macro_f1@0.5": test_f1,
                "val_subset_accuracy@0.5": val_subset,
                "test_subset_accuracy@0.5": test_subset,
            },
            hparams=hparams or {},
        )

    return metrics
