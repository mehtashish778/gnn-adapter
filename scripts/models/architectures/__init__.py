"""Neural architecture definitions (one module per model_id)."""

from .gnn07_label_residual import ResidualLabelGNNModel
from .gnn12_clip_vlm_homo import ClipVlmGraphAdapter
from .gnn13_clip_bipartite import (
    BipartiteMessagePassingLayer,
    ClipObjectBipartiteGNN,
    NativeGNNClassifier,
    build_bipartite_edge_weights,
)
from .cca import (
    CCAModel,
    CompositionalLayer,
    DEFAULT_CONCEPT_PHRASES,
    FindingsReadoutLayer,
    PrimitiveConceptLayer,
)
from .vlm_mlp import VLMFeatureMLP
from .vlm_zeroshot import VLMZeroShot

__all__ = [
    "VLMFeatureMLP",
    "VLMZeroShot",
    "ResidualLabelGNNModel",
    "ClipVlmGraphAdapter",
    "ClipObjectBipartiteGNN",
    "NativeGNNClassifier",
    "BipartiteMessagePassingLayer",
    "build_bipartite_edge_weights",
    "CCAModel",
    "PrimitiveConceptLayer",
    "CompositionalLayer",
    "FindingsReadoutLayer",
    "DEFAULT_CONCEPT_PHRASES",
]
