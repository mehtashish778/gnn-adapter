"""Simplified ML-GCN: label graph message passing on (logit, prob) nodes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLGCN(nn.Module):
    def __init__(self, num_labels: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.in_proj = nn.Linear(2, hidden)
        self.convs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, 1)
        self.register_buffer("adj", torch.eye(num_labels))

    def set_adjacency(self, adj: torch.Tensor) -> None:
        self.adj.copy_(adj)

    def forward(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        x = torch.stack([logits, probs], dim=-1)
        h = F.relu(self.in_proj(x))
        a = self.adj.unsqueeze(0)
        for conv in self.convs:
            msg = torch.matmul(a, h)
            h = F.relu(conv(msg) + h)
        return self.out(h).squeeze(-1)
