"""
Faithfulness metrics for concept-gated CCA models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from common_multilabel import masked_macro_f1


def hoyer_sparsity(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Hoyer measure of sparsity in [0, 1]; 1 = maximally sparse."""
    n = M.numel()
    l1 = M.abs().sum()
    l2 = torch.sqrt((M**2).sum() + eps)
    sqrt_n = float(n) ** 0.5
    density = (sqrt_n - l1 / (l2 + eps)) / max(sqrt_n - 1.0, eps)
    return density.clamp(0.0, 1.0)


def gate_density(M: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Fraction of active gates (hard or soft)."""
    if M.numel() == 0:
        return torch.tensor(0.0, device=M.device)
    return (M > threshold).float().mean()


def sparsity_target_loss(M: torch.Tensor, target: float = 0.10) -> torch.Tensor:
    """Penalize deviation from target gate density (5–15% band center)."""
    return (gate_density(M) - target) ** 2


def intervention_faithfulness_loss(
    y_hat: torch.Tensor,
    y_intervened: torch.Tensor,
    M_tilde: torch.Tensor,
    p_indices: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
  Penalize downstream change where gate says primitive p should NOT affect finding c.
  M_tilde: (C, P) or (B, C, P); p_indices: (B,) primitive masked per sample.
    """
    b, c_dim = y_hat.shape
    delta = (y_hat - y_intervened) ** 2
    if M_tilde.dim() == 2:
        # broadcast same gate for batch
        m_cp = M_tilde.unsqueeze(0).expand(b, -1, -1)
    else:
        m_cp = M_tilde
    p_idx = p_indices.view(b, 1, 1).expand(b, c_dim, 1)
    m_at_p = m_cp.gather(dim=2, index=p_idx).squeeze(2)
    # (1 - M[c,p]) * delta[c]: no dependence -> should not change
    penalty = ((1.0 - m_at_p) * delta).mean()
    return penalty


def intervention_consistency(
    y_hat: torch.Tensor,
    y_intervened: torch.Tensor,
    M_tilde: torch.Tensor,
    p_indices: torch.Tensor,
    influence_threshold: float = 0.01,
    gate_threshold: float = 0.5,
) -> float:
    """Agreement between binarized gate and empirical influence."""
    with torch.no_grad():
        delta = (y_hat - y_intervened).abs()
        influenced = (delta > influence_threshold).float()
        b, c_dim = y_hat.shape
        if M_tilde.dim() == 2:
            m_cp = M_tilde.unsqueeze(0).expand(b, -1, -1)
        else:
            m_cp = M_tilde
        p_idx = p_indices.view(b, 1, 1).expand(b, c_dim, 1)
        gate_on = (m_cp.gather(dim=2, index=p_idx).squeeze(2) > gate_threshold).float()
        agree = ((gate_on == influenced).float()).mean()
        return float(agree.item())


@torch.no_grad()
def necessity_sufficiency_scores(
    model: torch.nn.Module,
    patch_tokens: torch.Tensor,
    vlm_logits: torch.Tensor,
    vlm_probs: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    M_tilde: torch.Tensor,
    radgraph_prior: Optional[torch.Tensor] = None,
    gate_threshold: float = 0.5,
) -> Dict[str, float]:
    """Mask primitives by gate and measure macro-F1 change."""
    device = patch_tokens.device
    out_full, _, _ = model(patch_tokens, vlm_logits, vlm_probs, radgraph_prior=radgraph_prior)
    prob_full = torch.sigmoid(out_full)
    f1_full = float(masked_macro_f1(prob_full, y_true, y_mask, threshold=0.5))

    if not hasattr(model, "forward_from_comp_feats"):
        return {"f1_full": f1_full, "necessity_drop": 0.0, "sufficiency_f1": f1_full}

    primitive_feats, _, _ = model.layer1(patch_tokens)
    comp_feats = model.layer2(primitive_feats, radgraph_prior=radgraph_prior)
    m_hard = (M_tilde > gate_threshold).float() if M_tilde.dim() == 2 else (M_tilde > gate_threshold).float().mean(0)

    # Necessity: zero gated primitives only
    masked = comp_feats.clone()
    for p in range(comp_feats.shape[1]):
        if m_hard[:, p].max() > 0:
            masked[:, p, :] = 0.0
    out_mask, _ = model.forward_from_comp_feats(masked, vlm_logits, vlm_probs)
    f1_mask = float(masked_macro_f1(torch.sigmoid(out_mask), y_true, y_mask, threshold=0.5))

    # Sufficiency: keep only top-gated primitives per finding (union of active p)
    keep_p = (m_hard.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
    if keep_p.numel() == 0:
        f1_keep = 0.0
    else:
        kept = torch.zeros_like(comp_feats)
        kept[:, keep_p, :] = comp_feats[:, keep_p, :]
        out_keep, _ = model.forward_from_comp_feats(kept, vlm_logits, vlm_probs)
        f1_keep = float(masked_macro_f1(torch.sigmoid(out_keep), y_true, y_mask, threshold=0.5))

    return {
        "f1_full": f1_full,
        "necessity_drop": f1_full - f1_mask,
        "sufficiency_f1": f1_keep,
    }
