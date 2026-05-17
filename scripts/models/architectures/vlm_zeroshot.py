"""Passthrough adapter: frozen VLM logits with no learnable head."""

from __future__ import annotations

import torch.nn as nn


class VLMZeroShot(nn.Module):
  """Returns input logits unchanged (no parameters)."""

  def forward(self, x_logits, x_probs=None, adj=None, **kwargs):
    return x_logits
