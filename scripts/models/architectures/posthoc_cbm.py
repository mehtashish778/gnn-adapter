"""Post-hoc Concept Bottleneck Model on VLM features."""

from __future__ import annotations

import torch
import torch.nn as nn


class PostHocCBM(nn.Module):
    def __init__(self, input_dim: int, num_concepts: int, num_labels: int):
        super().__init__()
        self.concept_proj = nn.Linear(input_dim, num_concepts)
        self.label_head = nn.Linear(num_concepts, num_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.sigmoid(self.concept_proj(x))
        logits = self.label_head(c)
        return logits, c
