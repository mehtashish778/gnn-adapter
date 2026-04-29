#!/usr/bin/env bash
# Run all three GNN training pipelines (same splits, shared CLIP cache for image-based runs).
# Usage: from repo root,   bash scripts/run_all_gnn_variants.sh
# Optional:   GPU_ID=1 CLIP_CACHE=path/to.pt bash scripts/run_all_gnn_variants.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GPU_ID="${GPU_ID:-0}"
CLIP_CACHE="${CLIP_CACHE:-data/processed/embeddings/clip_vitb32_cache.pt}"

export PYTHONPATH="${REPO}/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "=== 07: residual label-graph (VLM only, C×C adj) ==="
python scripts/07_train_gnn_adapter.py --gpu_id "$GPU_ID"

echo "=== 12: CLIP + VLM, homogeneous K× adj @ H ==="
python scripts/12_train_clip_vlm_gnn_adapter.py --gpu_id "$GPU_ID" --clip_cache_pt "$CLIP_CACHE"

echo "=== 13: CLIP object + bipartite attribute→object GNN ==="
python scripts/13_train_bipartite_gnn_adapter.py --gpu_id "$GPU_ID" --clip_cache_pt "$CLIP_CACHE"

echo "Done. Outputs:"
echo "  data/processed/experiments/gnn_adapter/"
echo "  data/processed/experiments/clip_vlm_gnn_adapter/"
echo "  data/processed/experiments/bipartite_clip_gnn_adapter/"
