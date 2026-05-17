"""CLIP + VLM features on a homogeneous label graph (GNN12)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipVlmGraphAdapter(nn.Module):
  """
  Broadcast CLIP image embedding to each label node, concatenate VLM logit/prob per node,
  encode, then K layers of adjacency message passing, residual add to VLM logits.
  """

  def __init__(
    self,
    clip_dim: int,
    num_labels: int,
    hidden_dim: int,
    gnn_layers: int,
    alpha: float,
  ):
    super().__init__()
    self.num_labels = num_labels
    self.alpha = alpha
    self.clip_to_h = nn.Linear(clip_dim, hidden_dim)
    self.node_encoder = nn.Linear(hidden_dim + 2, hidden_dim)
    self.gnn_layers = nn.ModuleList(
      [nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, gnn_layers))]
    )
    self.score_head = nn.Linear(hidden_dim, 1)

  def forward(
    self,
    clip_emb: torch.Tensor,
    vlm_logits: torch.Tensor,
    vlm_probs: torch.Tensor,
    adj: torch.Tensor,
  ) -> torch.Tensor:
    b, c = vlm_logits.shape
    if c != self.num_labels:
      raise ValueError(f"Expected {self.num_labels} labels, got {c}")
    z = F.relu(self.clip_to_h(clip_emb))
    z = z.unsqueeze(1).expand(b, c, -1)
    x = torch.cat([z, vlm_logits.unsqueeze(-1), vlm_probs.unsqueeze(-1)], dim=-1)
    h = F.relu(self.node_encoder(x))
    for lin in self.gnn_layers:
      h = torch.einsum("ij,bjh->bih", adj, h)
      h = F.relu(lin(h))
    delta = self.score_head(h).squeeze(-1)
    return vlm_logits + self.alpha * delta
