"""
Model factory registry. Architecture classes live under ``models.architectures``.
Entry-point scripts remain in ``models/<model_id>/train.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .architectures.cca import CCAModel
from .architectures.gnn07_label_residual import ResidualLabelGNNModel
from .architectures.gnn12_clip_vlm_homo import ClipVlmGraphAdapter
from .architectures.gnn13_clip_bipartite import ClipObjectBipartiteGNN
from .architectures.vlm_mlp import VLMFeatureMLP
from .architectures.vlm_zeroshot import VLMZeroShot

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
  "vlm_zeroshot": VLMZeroShot,
  "vlm_mlp": VLMFeatureMLP,
  "gnn07_label_residual": ResidualLabelGNNModel,
  "gnn12_clip_vlm_homo": ClipVlmGraphAdapter,
  "gnn13_clip_bipartite": ClipObjectBipartiteGNN,
  "cca": CCAModel,
}


def get_model(model_id: str, **kwargs):
  """Instantiate a registered architecture by model_id."""
  if model_id not in MODEL_REGISTRY:
    raise KeyError(f"Unknown model_id {model_id!r}; known: {sorted(MODEL_REGISTRY)}")
  return MODEL_REGISTRY[model_id](**kwargs)


__all__ = [
  "MODEL_REGISTRY",
  "get_model",
  "VLMFeatureMLP",
  "VLMZeroShot",
  "ResidualLabelGNNModel",
  "ClipVlmGraphAdapter",
  "ClipObjectBipartiteGNN",
  "CCAModel",
]
