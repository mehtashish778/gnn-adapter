#!/usr/bin/env bash
# End-to-end: build 4-way split + graph, train/eval every model variant, optional calib thresholds.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/scripts${PYTHONPATH:+:$PYTHONPATH}"

PY="${REPO}/.venv/bin/python"
# Separate caches per split regime: mixing default vs 4-way row orders in one .pt triggers verify_paths_order errors.
CLIP_CACHE_DEFAULT="${CLIP_CACHE_DEFAULT:-${REPO}/data/processed/embeddings/clip_vitb32_default.pt}"
CLIP_CACHE_4WAY="${CLIP_CACHE_4WAY:-${REPO}/data/processed/embeddings/clip_vitb32_calibrated4way.pt}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-reproduce_full_$(date +%Y%m%d_%H%M%S)}"

SPLIT="data/processed/splits"
SPLIT4="data/processed/splits_4way"
GRAPH="data/processed/graph"
GRAPH4="${GRAPH}_4way"

die() {
  echo "ERROR $*" >&2
  exit 1
}

[[ -x "$PY" ]] || die "venv missing at $PY (create .venv and pip install -r requirements.txt)"

echo "RUN_ID=${RUN_ID} GPU=${GPU} REPO=${REPO}"

echo "=== 1) Canonical data + splits (idempotent)"
"$PY" scripts/01_build_canonical_labels.py
"$PY" scripts/02_align_vlm_outputs.py
"$PY" scripts/03_make_multilabel_splits.py
"$PY" scripts/03_make_multilabel_splits_4way.py
"$PY" scripts/04_build_coerror_graph.py --train_rows_json "${SPLIT}/train_rows.json" --out_dir "$GRAPH"
"$PY" scripts/04_build_coerror_graph.py --train_rows_json "${SPLIT4}/train_fit_rows.json" --out_dir "$GRAPH4"

mkdir -p "$(dirname "${CLIP_CACHE_DEFAULT}")" "$(dirname "${CLIP_CACHE_4WAY}")"

leaky_val_threshold_eval_run() {
  local run_dir="$1"
  [[ -f "${run_dir}/val_predictions.json" ]] || die "missing val_predictions.json in ${run_dir}"
  "$PY" scripts/08_tune_thresholds.py \
    --val_predictions_json "${run_dir}/val_predictions.json" \
    --out_json "${run_dir}/per_class_thresholds_tuned_on_val_LEAKY.json"
  "$PY" scripts/09_evaluate_test.py \
    --test_predictions_json "${run_dir}/val_predictions.json" \
    --thresholds_json "${run_dir}/per_class_thresholds_tuned_on_val_LEAKY.json" \
    --out_json "${run_dir}/val_metrics_thr_tuned_on_val_LEAKY.json"
  "$PY" scripts/09_evaluate_test.py \
    --test_predictions_json "${run_dir}/test_predictions.json" \
    --thresholds_json "${run_dir}/per_class_thresholds_tuned_on_val_LEAKY.json" \
    --out_json "${run_dir}/test_metrics_thr_tuned_on_val_LEAKY.json"
}

calibrated_eval_run() {
  local run_dir="$1"
  [[ -f "${run_dir}/calib_predictions.json" ]] || die "missing calib_predictions.json in ${run_dir}"
  "$PY" scripts/08_tune_thresholds.py \
    --val_predictions_json "${run_dir}/calib_predictions.json" \
    --out_json "${run_dir}/per_class_thresholds.json"
  "$PY" scripts/09_evaluate_test.py \
    --test_predictions_json "${run_dir}/val_predictions.json" \
    --thresholds_json "${run_dir}/per_class_thresholds.json" \
    --out_json "${run_dir}/val_metrics_calibrated.json"
  "$PY" scripts/09_evaluate_test.py \
    --test_predictions_json "${run_dir}/test_predictions.json" \
    --thresholds_json "${run_dir}/per_class_thresholds.json" \
    --out_json "${run_dir}/test_metrics_calibrated.json"
}

echo "=== 2) Frozen VLM zeroshot (no training)"
"$PY" scripts/05_run_baseline_frozen_vlm.py \
  --model_id vlm_zeroshot \
  --protocol default \
  --run_id "${RUN_ID}" \
  --val_rows_json "${SPLIT}/val_rows.json" \
  --test_rows_json "${SPLIT}/test_rows.json" \
  --threshold 0.5

ZSDIR_DEFAULT="${REPO}/data/processed/experiments/vlm_zeroshot/default/${RUN_ID}"
"$PY" scripts/export_rows_to_predictions.py \
  --rows_json "${SPLIT}/val_rows.json" \
  --out_json "${ZSDIR_DEFAULT}/val_predictions.json"
"$PY" scripts/export_rows_to_predictions.py \
  --rows_json "${SPLIT}/test_rows.json" \
  --out_json "${ZSDIR_DEFAULT}/test_predictions.json"
leaky_val_threshold_eval_run "${ZSDIR_DEFAULT}"

"$PY" scripts/05_run_baseline_frozen_vlm.py \
  --model_id vlm_zeroshot \
  --protocol calibrated4way \
  --run_id "${RUN_ID}" \
  --val_rows_json "${SPLIT4}/val_rows.json" \
  --test_rows_json "${SPLIT4}/test_rows.json" \
  --threshold 0.5

# Calib leakage-free thresholds for zeroshot: predictions = raw x_probs from rows.
ZSDIR_CALIB="${REPO}/data/processed/experiments/vlm_zeroshot/calibrated4way/${RUN_ID}"
"$PY" scripts/export_rows_to_predictions.py \
  --rows_json "${SPLIT4}/calib_rows.json" \
  --out_json "${ZSDIR_CALIB}/calib_predictions.json"
"$PY" scripts/export_rows_to_predictions.py \
  --rows_json "${SPLIT4}/val_rows.json" \
  --out_json "${ZSDIR_CALIB}/val_predictions.json"
"$PY" scripts/export_rows_to_predictions.py \
  --rows_json "${SPLIT4}/test_rows.json" \
  --out_json "${ZSDIR_CALIB}/test_predictions.json"
calibrated_eval_run "${ZSDIR_CALIB}"

echo "=== 3) MLP baseline"
"$PY" scripts/06_run_baseline_mlp.py \
  --gpu_id "${GPU}" \
  --model_id vlm_mlp \
  --protocol default \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT}/train_rows.json" \
  --val_rows_json "${SPLIT}/val_rows.json" \
  --test_rows_json "${SPLIT}/test_rows.json"

"$PY" scripts/06_run_baseline_mlp.py \
  --gpu_id "${GPU}" \
  --model_id vlm_mlp \
  --protocol calibrated4way \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT4}/train_fit_rows.json" \
  --calib_rows_json "${SPLIT4}/calib_rows.json" \
  --val_rows_json "${SPLIT4}/val_rows.json" \
  --test_rows_json "${SPLIT4}/test_rows.json"
calibrated_eval_run "${REPO}/data/processed/experiments/vlm_mlp/calibrated4way/${RUN_ID}"
leaky_val_threshold_eval_run "${REPO}/data/processed/experiments/vlm_mlp/default/${RUN_ID}"

echo "=== 4) GNN-07 residual label graph"
"$PY" scripts/07_train_gnn_adapter.py \
  --gpu_id "${GPU}" \
  --model_id gnn07_label_residual \
  --protocol default \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT}/train_rows.json" \
  --val_rows_json "${SPLIT}/val_rows.json" \
  --test_rows_json "${SPLIT}/test_rows.json" \
  --edge_index_json "${GRAPH}/edge_index.json" \
  --edge_weight_json "${GRAPH}/edge_weight.json"

"$PY" scripts/07_train_gnn_adapter.py \
  --gpu_id "${GPU}" \
  --model_id gnn07_label_residual \
  --protocol calibrated4way \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT4}/train_fit_rows.json" \
  --calib_rows_json "${SPLIT4}/calib_rows.json" \
  --val_rows_json "${SPLIT4}/val_rows.json" \
  --test_rows_json "${SPLIT4}/test_rows.json" \
  --edge_index_json "${GRAPH4}/edge_index.json" \
  --edge_weight_json "${GRAPH4}/edge_weight.json"
calibrated_eval_run "${REPO}/data/processed/experiments/gnn07_label_residual/calibrated4way/${RUN_ID}"
leaky_val_threshold_eval_run "${REPO}/data/processed/experiments/gnn07_label_residual/default/${RUN_ID}"

echo "=== 5) GNN-12 CLIP + homogeneous graph"
"$PY" scripts/12_train_clip_vlm_gnn_adapter.py \
  --gpu_id "${GPU}" \
  --model_id gnn12_clip_vlm_homo \
  --protocol default \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT}/train_rows.json" \
  --val_rows_json "${SPLIT}/val_rows.json" \
  --test_rows_json "${SPLIT}/test_rows.json" \
  --edge_index_json "${GRAPH}/edge_index.json" \
  --edge_weight_json "${GRAPH}/edge_weight.json" \
  --clip_cache_pt "${CLIP_CACHE_DEFAULT}"

"$PY" scripts/12_train_clip_vlm_gnn_adapter.py \
  --gpu_id "${GPU}" \
  --model_id gnn12_clip_vlm_homo \
  --protocol calibrated4way \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT4}/train_fit_rows.json" \
  --calib_rows_json "${SPLIT4}/calib_rows.json" \
  --val_rows_json "${SPLIT4}/val_rows.json" \
  --test_rows_json "${SPLIT4}/test_rows.json" \
  --edge_index_json "${GRAPH4}/edge_index.json" \
  --edge_weight_json "${GRAPH4}/edge_weight.json" \
  --clip_cache_pt "${CLIP_CACHE_4WAY}"
calibrated_eval_run "${REPO}/data/processed/experiments/gnn12_clip_vlm_homo/calibrated4way/${RUN_ID}"
leaky_val_threshold_eval_run "${REPO}/data/processed/experiments/gnn12_clip_vlm_homo/default/${RUN_ID}"

echo "=== 6) GNN-13 bipartite CLIP"
"$PY" scripts/13_train_bipartite_gnn_adapter.py \
  --gpu_id "${GPU}" \
  --model_id gnn13_clip_bipartite \
  --protocol default \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT}/train_rows.json" \
  --val_rows_json "${SPLIT}/val_rows.json" \
  --test_rows_json "${SPLIT}/test_rows.json" \
  --clip_cache_pt "${CLIP_CACHE_DEFAULT}"

"$PY" scripts/13_train_bipartite_gnn_adapter.py \
  --gpu_id "${GPU}" \
  --model_id gnn13_clip_bipartite \
  --protocol calibrated4way \
  --run_id "${RUN_ID}" \
  --train_rows_json "${SPLIT4}/train_fit_rows.json" \
  --calib_rows_json "${SPLIT4}/calib_rows.json" \
  --val_rows_json "${SPLIT4}/val_rows.json" \
  --test_rows_json "${SPLIT4}/test_rows.json" \
  --clip_cache_pt "${CLIP_CACHE_4WAY}"
calibrated_eval_run "${REPO}/data/processed/experiments/gnn13_clip_bipartite/calibrated4way/${RUN_ID}"
leaky_val_threshold_eval_run "${REPO}/data/processed/experiments/gnn13_clip_bipartite/default/${RUN_ID}"

echo "=== 7) Package comparison report"
"$PY" scripts/11_package_report.py

echo "Done. Run artifacts under data/processed/experiments/*/${RUN_ID}/"
