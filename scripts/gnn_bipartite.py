"""
Backward-compatible re-exports. Implementation lives in models.architectures.gnn13_clip_bipartite.
"""

from models.architectures.gnn13_clip_bipartite import (
  BipartiteMessagePassingLayer,
  ClipObjectBipartiteGNN,
  NativeGNNClassifier,
  build_bipartite_edge_weights,
)

__all__ = [
  "BipartiteMessagePassingLayer",
  "ClipObjectBipartiteGNN",
  "NativeGNNClassifier",
  "build_bipartite_edge_weights",
]
