"""Q-Former-style learnable query adapter over patch tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class QFormerAdapter(nn.Module):
    def __init__(
        self,
        patch_dim: int = 768,
        query_dim: int = 128,
        num_queries: int = 32,
        num_labels: int = 7,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(patch_dim, query_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.cross = nn.ModuleList(
            [
                nn.MultiheadAttention(query_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(query_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(query_dim * num_queries, num_labels)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b = patch_tokens.shape[0]
        kv = self.input_proj(patch_tokens)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        for attn, ln in zip(self.cross, self.norms):
            out, _ = attn(q, kv, kv, need_weights=False)
            q = ln(q + self.dropout(out))
        flat = q.reshape(b, -1)
        return self.readout(flat)
