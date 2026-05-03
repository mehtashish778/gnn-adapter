#!/usr/bin/env bash
# Regenerate metrics.json / *predictions.json from existing checkpoints (no weight updates).
# Prereqs: run 01→04 already (splits + graph exist). CUDA required for 06–13.
#
# Usage: from repo root
#   bash scripts/regenerate_metrics_default.sh
#   GPU_ID=1 CLIP_CACHE=data/processed/embeddings/clip_vitb32_cache.pt bash scripts/regenerate_metrics_default.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"

for need in "$PY" data/processed/splits/train_rows.json data/processed/graph/edge_index.json; do
  if [[ ! -e "$need" ]]; then
    echo "Missing $need — run first:"
    echo "  .venv/bin/python scripts/01_build_canonical_labels.py"
    echo "  .venv/bin/python scripts/02_align_vlm_outputs.py"
    echo "  .venv/bin/python scripts/03_make_multilabel_splits.py"
    echo "  .venv/bin/python scripts/04_build_coerror_graph.py"
    exit 1
  fi
done

# Table 6.1 default MLP weights: prefer frozen snapshot from the 2026-04-30 archive (paper run), else canonical path.
MLP_SNAPSHOT="$REPO/data/processed/experiments_backup_20260430/experiments/vlm_mlp/default/fresh_full_retrain_20260430/best_checkpoint.pt"
MLP_CANON="$REPO/data/processed/experiments/vlm_mlp/default/fresh_full_retrain_20260430/best_checkpoint.pt"
pick_mlp_ckpt () {
  if [[ -f "$MLP_SNAPSHOT" ]]; then echo "$MLP_SNAPSHOT"
  elif [[ -f "$MLP_CANON" ]]; then echo "$MLP_CANON"
  else echo ""; fi
}
CKPT_MLP="${CKPT_MLP:-$(pick_mlp_ckpt)}"
CKPT_GNN07="${CKPT_GNN07:-$REPO/data/processed/experiments/gnn07_label_residual/default/fresh_full_retrain_20260430/best_checkpoint.pt}"
LEGACY07="${LEGACY07:-$REPO/data/processed/experiments/gnn_adapter/best_checkpoint.pt}"
CKPT_GNN12="${CKPT_GNN12:-$REPO/data/processed/experiments/gnn12_clip_vlm_homo/default/fresh_full_retrain_20260430/best_checkpoint.pt}"
LEGACY12="${LEGACY12:-$REPO/data/processed/experiments/clip_vlm_gnn_adapter/best_checkpoint.pt}"
CKPT_GNN13="${CKPT_GNN13:-$REPO/data/processed/experiments/gnn13_clip_bipartite/default/fresh_full_retrain_20260430/best_checkpoint.pt}"
LEGACY13="${LEGACY13:-$REPO/data/processed/experiments/bipartite_clip_gnn_adapter/best_checkpoint.pt}"

CLIP_CACHE="${CLIP_CACHE:-$REPO/data/processed/embeddings/clip_vitb32_cache.pt}"
CLIP_EXTRA=()
[[ -n "${CLIP_CACHE:-}" && -f "$CLIP_CACHE" ]] && CLIP_EXTRA=(--clip_cache_pt "$CLIP_CACHE")

pick_ckpt () {
  local primary="$1" fallback="$2"
  if [[ -f "$primary" ]]; then echo "$primary"
  elif [[ -f "$fallback" ]]; then echo "$fallback"
  else echo ""; fi
}

echo "=== 05 Frozen VLM (no checkpoint) → baseline_frozen_vlm ==="
"$PY" scripts/05_run_baseline_frozen_vlm.py --out_dir data/processed/experiments/baseline_frozen_vlm

if [[ -f "$CKPT_MLP" ]]; then
  echo "=== 06 MLP eval_only → baseline_mlp ==="
  "$PY" scripts/06_run_baseline_mlp.py --eval_only --resume_from "$CKPT_MLP" --gpu_id "$GPU_ID" \
    --out_dir data/processed/experiments/baseline_mlp
else
  echo "Skip MLP: missing $CKPT_MLP"
fi

CK07="$(pick_ckpt "$CKPT_GNN07" "$LEGACY07")"
if [[ -n "$CK07" ]]; then
  echo "=== 07 GNN residual eval_only → gnn_adapter ==="
  "$PY" scripts/07_train_gnn_adapter.py --eval_only --resume_from "$CK07" --gpu_id "$GPU_ID" \
    --out_dir data/processed/experiments/gnn_adapter
else
  echo "Skip gnn07: no checkpoint found"
fi

CK12="$(pick_ckpt "$CKPT_GNN12" "$LEGACY12")"
if [[ -n "$CK12" ]]; then
  echo "=== 12 CLIP+VLM GNN eval_only → clip_vlm_gnn_adapter ==="
  "$PY" scripts/12_train_clip_vlm_gnn_adapter.py --eval_only --resume_from "$CK12" --gpu_id "$GPU_ID" \
    "${CLIP_EXTRA[@]}" --out_dir data/processed/experiments/clip_vlm_gnn_adapter
else
  echo "Skip gnn12: no checkpoint found"
fi

CK13="$(pick_ckpt "$CKPT_GNN13" "$LEGACY13")"
if [[ -n "$CK13" ]]; then
  echo "=== 13 Bipartite GNN eval_only → bipartite_clip_gnn_adapter ==="
  "$PY" scripts/13_train_bipartite_gnn_adapter.py --eval_only --resume_from "$CK13" --gpu_id "$GPU_ID" \
    "${CLIP_EXTRA[@]}" --out_dir data/processed/experiments/bipartite_clip_gnn_adapter
else
  echo "Skip gnn13: no checkpoint found"
fi

echo "=== Optional: ablation CSV + markdown report ==="
mkdir -p data/processed/experiments/ablations reports/gnn_adapter reports/comparison
"$PY" scripts/10_run_ablations.py 2>/dev/null || true
"$PY" scripts/11_package_report.py 2>/dev/null || true

echo Done. Inspect data/processed/experiments/*/metrics.json
