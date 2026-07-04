"""Residual message passing on a label graph (homogeneous GNN07)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualLabelGNNModel(nn.Module):
  def __init__(self, hidden_dim: int = 32, alpha: float = 0.5):
    super().__init__()
    self.fc1 = nn.Linear(2, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, 1)
    self.alpha = alpha

  def forward(self, logits: torch.Tensor, probs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    x = torch.stack([logits, probs], dim=-1)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x).squeeze(-1)
    x = torch.matmul(x, adj.T)
    return logits + self.alpha * x
