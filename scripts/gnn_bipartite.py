"""
Bipartite attribute→object GNN (plain PyTorch, no PyG).
Attribute nodes send weighted-mean messages to each object node; object state is updated via concat + MLP + dropout.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_bipartite_edge_weights(
    vlm_probs: torch.Tensor,
    mode: Literal["all", "vlm_positive"],
    vlm_tau: float = 0.5,
) -> torch.Tensor:
    """
    Per-sample weights from each attribute index → object (shape [B, C]).
    "all": uniform 1.
    "vlm_positive": 1 where vlm_probs >= vlm_tau else 0; if a row has no edge, fallback to all-ones.
    """
    if mode == "all":
        return torch.ones_like(vlm_probs)
    w = (vlm_probs >= vlm_tau).float()
    empty = (w.sum(dim=1, keepdim=True) <= 0).float()
    return w + empty * torch.ones_like(w)


class BipartiteMessagePassingLayer(nn.Module):
    """
    Aggregate attribute messages into object nodes (weighted mean), project, concat with object, update.
    """

    def __init__(self, object_dim: int, attr_dim: int, mid_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.attr_to_mid = nn.Linear(attr_dim, mid_dim)
        self.mid_to_proj = nn.Linear(mid_dim, out_dim)
        self.update = nn.Sequential(
            nn.Linear(object_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        object_feats: torch.Tensor,
        attr_feats: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        # object_feats [B, O, D_obj], attr_feats [B, C, D_attr], edge_weight [B, C]
        b, o, d_obj = object_feats.shape
        msg = self.attr_to_mid(attr_feats)
        w = edge_weight.unsqueeze(-1)
        agg = (w * msg).sum(dim=1)
        den = w.sum(dim=1).clamp(min=1e-6)
        agg = agg / den
        proj = self.mid_to_proj(agg).unsqueeze(1).expand(-1, o, -1)
        x = torch.cat([object_feats, proj], dim=-1)
        return self.update(x)


class NativeGNNClassifier(nn.Module):
    """
    Stack of bipartite layers on one object (O=1) or more; readout Linear -> num_attributes, mean over objects.
    """

    def __init__(
        self,
        object_in_dim: int,
        attr_dim: int,
        hidden_dims: List[int],
        num_attributes: int,
        mid_dim: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_objects = 1
        dims = [object_in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        for i, out_dim in enumerate(hidden_dims):
            in_obj = dims[i]
            md = mid_dim if mid_dim is not None else out_dim
            self.layers.append(
                BipartiteMessagePassingLayer(
                    object_dim=in_obj,
                    attr_dim=attr_dim,
                    mid_dim=md,
                    out_dim=out_dim,
                    dropout=dropout,
                )
            )
        self.classifier = nn.Linear(hidden_dims[-1], num_attributes)

    def forward(
        self,
        object_feats: torch.Tensor,
        attr_feats: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = object_feats
        for layer in self.layers:
            x = layer(x, attr_feats, edge_weight)
        logits = self.classifier(x)
        return logits.mean(dim=1)
