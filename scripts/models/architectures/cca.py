"""
Compositional Concept Adapter (CCA): patch cross-attention → compositional self-attn → findings readout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONCEPT_PHRASES: List[str] = [
    "atelectasis",
    "pleural effusion",
    "cardiomegaly",
    "pulmonary edema",
    "pneumonia",
    "consolidation",
    "no finding",
    "opacity",
    "haziness",
    "air space opacity",
    "blunting",
    "costophrenic angle blunting",
    "cardiac silhouette enlarged",
    "vascular congestion",
    "interstitial markings",
    "airspace disease",
    "lobar collapse",
    "subsegmental atelectasis",
    "perihilar opacity",
    "bilateral opacities",
    "unilateral opacity",
    "pleural thickening",
    "linear opacity",
    "reticular pattern",
    "ground glass opacity",
    "pulmonary nodule",
    "hyperinflation",
    "tracheal deviation",
    "mediastinal widening",
    "rib fracture",
    "pneumothorax",
    "enlarged heart",
    "pulmonary vascular redistribution",
    "support devices",
    "chest wall abnormality",
]


def sample_gumbel(shape: torch.Size, device: torch.device, eps: float = 1e-8) -> torch.Tensor:
    u = torch.rand(shape, device=device).clamp(eps, 1.0 - eps)
    return -torch.log(-torch.log(u))


class GumbelGate(nn.Module):
    """Relaxed binary gate M_tilde (C, P) via Gumbel-sigmoid (binary concrete)."""

    def __init__(self, num_findings: int, num_primitives: int):
        super().__init__()
        self.num_findings = num_findings
        self.num_primitives = num_primitives
        self.logits = nn.Parameter(torch.zeros(num_findings, num_primitives))

    def forward(self, tau: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.training and not hard:
            g = sample_gumbel(self.logits.shape, self.logits.device)
            y = torch.sigmoid((self.logits + g) / max(tau, 1e-4))
        else:
            y = (torch.sigmoid(self.logits) > 0.5).float()
        aux = {"M_tilde": y, "logits": self.logits, "tau": torch.tensor(tau)}
        return y, aux

    def hard_gate(self) -> torch.Tensor:
        return (torch.sigmoid(self.logits) > 0.5).float()


class PrimitiveConceptLayer(nn.Module):
    """Layer 1: P concept queries cross-attend over frozen patch tokens."""

    def __init__(
        self,
        patch_dim: int,
        query_dim: int,
        num_primitives: int,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_primitives = num_primitives
        self.query_dim = query_dim
        self.input_proj = nn.Linear(patch_dim, query_dim)
        self.concept_queries = nn.Parameter(torch.randn(num_primitives, query_dim) * 0.02)
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(query_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(max(0, n_layers))
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(query_dim) for _ in range(max(0, n_layers))])
        self.activation_head = nn.Linear(query_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = patch_tokens.shape[0]
        kv = self.input_proj(patch_tokens)
        queries = self.concept_queries.unsqueeze(0).expand(b, -1, -1)
        attn_accum = None

        if len(self.cross_attn_layers) == 0:
            primitive_feats = queries
        else:
            for attn_layer, ln in zip(self.cross_attn_layers, self.layer_norms):
                out, weights = attn_layer(queries, kv, kv, need_weights=True, average_attn_weights=True)
                queries = ln(queries + self.dropout(out))
                if weights is not None:
                    attn_accum = weights if attn_accum is None else attn_accum + weights

        primitive_feats = queries
        primitive_acts = self.activation_head(primitive_feats).squeeze(-1)
        if attn_accum is None:
            n_patches = patch_tokens.shape[1]
            attn_weights = torch.zeros(b, self.num_primitives, n_patches, device=patch_tokens.device)
        else:
            attn_weights = attn_accum / len(self.cross_attn_layers)
        return primitive_feats, primitive_acts, attn_weights


class CompositionalLayer(nn.Module):
    """Layer 2: self-attention over P primitive features with optional RadGraph bias."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(query_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(max(0, n_layers))
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(query_dim) for _ in range(max(0, n_layers))])
        self.dropout = nn.Dropout(dropout)
        self.radgraph_bias: Optional[nn.Parameter] = None

    def set_radgraph_prior(self, prior: Optional[torch.Tensor]) -> None:
        if prior is None:
            self.radgraph_bias = None
            return
        if self.radgraph_bias is None or self.radgraph_bias.shape != prior.shape:
            self.radgraph_bias = nn.Parameter(prior.clone())
        else:
            with torch.no_grad():
                self.radgraph_bias.copy_(prior)

    def forward(
        self,
        primitive_feats: torch.Tensor,
        radgraph_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = primitive_feats
        bias = radgraph_prior if radgraph_prior is not None else self.radgraph_bias

        if len(self.self_attn_layers) == 0:
            return x

        for attn_layer, ln in zip(self.self_attn_layers, self.layer_norms):
            if bias is not None and bias.shape[0] == x.shape[1]:
                x = x + torch.matmul(F.softmax(bias, dim=-1), x)
            out, _ = attn_layer(x, x, x, need_weights=False)
            x = ln(x + self.dropout(out))
        return x


class FindingsReadoutLayer(nn.Module):
    """Layer 3: attention readout to C findings + VLM residual gating."""

    def __init__(
        self,
        query_dim: int,
        num_findings: int,
        n_heads: int = 2,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_findings = num_findings
        self.alpha = alpha
        self.finding_queries = nn.Parameter(torch.randn(num_findings, query_dim) * 0.02)
        self.readout_attn = nn.MultiheadAttention(query_dim, n_heads, dropout=dropout, batch_first=True)
        self.readout_norm = nn.LayerNorm(query_dim)
        self.score_head = nn.Linear(query_dim, 1)
        self.vlm_gate = nn.Linear(num_findings * 2, num_findings)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        compositional_feats: torch.Tensor,
        vlm_logits: torch.Tensor,
        vlm_probs: torch.Tensor,
        gate_M: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = compositional_feats.shape[0]
        kv = compositional_feats
        if gate_M is not None:
            gated = torch.einsum("cp,bpd->bcd", gate_M, compositional_feats)
            kv = gated

        queries = self.finding_queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.readout_attn(queries, kv, kv, need_weights=False)
        out = self.readout_norm(queries + self.dropout(out))
        readout_logits = self.score_head(out).squeeze(-1)

        vlm_mix = self.vlm_gate(torch.cat([vlm_logits, vlm_probs], dim=-1))
        return readout_logits + self.alpha * vlm_mix


class CCAModel(nn.Module):
    """Compositional Concept Adapter."""

    def __init__(
        self,
        patch_dim: int = 768,
        query_dim: int = 128,
        num_primitives: int = 30,
        num_findings: int = 7,
        n_heads: int = 2,
        n_cross_attn_layers: int = 2,
        n_self_attn_layers: int = 2,
        alpha: float = 1.0,
        dropout: float = 0.1,
        use_gate_M: bool = True,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.num_primitives = num_primitives
        self.num_findings = num_findings
        self.query_dim = query_dim
        self.gumbel_tau = gumbel_tau
        self.layer1 = PrimitiveConceptLayer(
            patch_dim=patch_dim,
            query_dim=query_dim,
            num_primitives=num_primitives,
            n_heads=n_heads,
            n_layers=n_cross_attn_layers,
            dropout=dropout,
        )
        self.layer2 = CompositionalLayer(
            query_dim=query_dim,
            n_heads=n_heads,
            n_layers=n_self_attn_layers,
            dropout=dropout,
        )
        self.layer3 = FindingsReadoutLayer(
            query_dim=query_dim,
            num_findings=num_findings,
            n_heads=n_heads,
            alpha=alpha,
            dropout=dropout,
        )
        self.use_gate_M = use_gate_M
        if use_gate_M:
            self.gate = GumbelGate(num_findings, num_primitives)
        else:
            self.gate = None

    def _encode(
        self,
        patch_tokens: torch.Tensor,
        radgraph_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        primitive_feats, primitive_acts, attn_maps = self.layer1(patch_tokens)
        comp_feats = self.layer2(primitive_feats, radgraph_prior=radgraph_prior)
        return primitive_feats, comp_feats, attn_maps

    def forward_from_comp_feats(
        self,
        comp_feats: torch.Tensor,
        vlm_logits: torch.Tensor,
        vlm_probs: torch.Tensor,
        gate_M: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        logits = self.layer3(comp_feats, vlm_logits, vlm_probs, gate_M=gate_M)
        return logits, None

    def forward(
        self,
        patch_tokens: torch.Tensor,
        vlm_logits: torch.Tensor,
        vlm_probs: torch.Tensor,
        radgraph_prior: Optional[torch.Tensor] = None,
        gumbel_tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        tau = self.gumbel_tau if gumbel_tau is None else gumbel_tau
        _, comp_feats, attn_maps = self._encode(patch_tokens, radgraph_prior=radgraph_prior)
        gate_aux: Dict[str, Any] = {}
        gate_M = None
        if self.use_gate_M and self.gate is not None:
            gate_M, gate_aux = self.gate(tau=tau, hard=not self.training)
            gate_aux["gate_density"] = gate_M.mean()
        logits = self.layer3(comp_feats, vlm_logits, vlm_probs, gate_M=gate_M)
        return logits, attn_maps, gate_aux

    def forward_with_intervention(
        self,
        patch_tokens: torch.Tensor,
        vlm_logits: torch.Tensor,
        vlm_probs: torch.Tensor,
        p_indices: torch.Tensor,
        radgraph_prior: Optional[torch.Tensor] = None,
        gumbel_tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Mask primitive p_indices[b] per sample and re-readout."""
        tau = self.gumbel_tau if gumbel_tau is None else gumbel_tau
        _, comp_feats, attn_maps = self._encode(patch_tokens, radgraph_prior=radgraph_prior)
        gate_aux: Dict[str, Any] = {}
        gate_M = None
        if self.use_gate_M and self.gate is not None:
            gate_M, gate_aux = self.gate(tau=tau, hard=True)
        logits = self.layer3(comp_feats, vlm_logits, vlm_probs, gate_M=gate_M)

        comp_int = comp_feats.clone()
        b = comp_feats.shape[0]
        for i in range(b):
            p = int(p_indices[i].item())
            if 0 <= p < comp_int.shape[1]:
                comp_int[i, p, :] = 0.0
        logits_int = self.layer3(comp_int, vlm_logits, vlm_probs, gate_M=gate_M)
        gate_aux["logits_intervened"] = logits_int
        return logits, logits_int, gate_aux

    def init_concept_queries_from_text(self, text_embeddings: torch.Tensor) -> None:
        n = min(self.num_primitives, text_embeddings.shape[0])
        d = self.query_dim
        if text_embeddings.shape[-1] != d:
            raise ValueError(
                f"text_embeddings dim {text_embeddings.shape[-1]} != query_dim {d}; project before calling."
            )
        with torch.no_grad():
            self.layer1.concept_queries.data[:n] = text_embeddings[:n].to(self.layer1.concept_queries.device)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Backward compat for checkpoints with gate_M Parameter
    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        if "gate_M" in state_dict and self.gate is not None:
            state_dict = dict(state_dict)
            state_dict["gate.logits"] = state_dict.pop("gate_M")
        return super().load_state_dict(state_dict, strict=strict)
