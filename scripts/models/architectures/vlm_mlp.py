"""MLP adapter over frozen VLM logits and probabilities."""

from __future__ import annotations

import torch.nn as nn


class VLMFeatureMLP(nn.Sequential):
  """Two-layer MLP: flattened [z; p] -> hidden -> C logits."""

  def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 64, dropout: float = 0.1):
    super().__init__(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, num_labels),
    )
