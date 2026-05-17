"""
Compositional Concept Adapter (CCA): patch cross-attention → compositional self-attn → findings readout.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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
        """
        Args:
            patch_tokens: (B, N_patches, patch_dim)
        Returns:
            primitive_feats: (B, P, D)
            primitive_acts: (B, P)
            attn_weights: (B, P, N_patches) averaged over layers
        """
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
        """Register a learnable P×P bias initialized from RadGraph (optional)."""
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
            # gate_M: (C, P) weights primitive contributions per finding
            gated = torch.einsum("cp,bpd->bcd", torch.sigmoid(gate_M), compositional_feats)
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
    ):
        super().__init__()
        self.num_primitives = num_primitives
        self.num_findings = num_findings
        self.query_dim = query_dim
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
            self.gate_M = nn.Parameter(torch.zeros(num_findings, num_primitives))
        else:
            self.register_parameter("gate_M", None)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        vlm_logits: torch.Tensor,
        vlm_probs: torch.Tensor,
        radgraph_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        primitive_feats, _primitive_acts, attn_maps = self.layer1(patch_tokens)
        comp_feats = self.layer2(primitive_feats, radgraph_prior=radgraph_prior)
        gate = self.gate_M if self.use_gate_M and self.gate_M is not None else None
        logits = self.layer3(comp_feats, vlm_logits, vlm_probs, gate_M=gate)
        return logits, attn_maps

    def init_concept_queries_from_text(self, text_embeddings: torch.Tensor) -> None:
        """Copy projected text embeddings into concept_queries (first min(P, N) rows)."""
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
