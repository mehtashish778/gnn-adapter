# Metrics Documentation

This document explains how metrics in this repo are computed and how to compare runs correctly.

## Core Metric

Primary metric is macro F1 over labels with masking (`y_mask`):
- per class: F1 from TP/FP/FN on valid entries only
- macro F1: average across classes

Masking matters because some labels are missing/unmentioned.

## Subset accuracy (exact multi-label match)

Also reported beside macro F1 (for example `09_evaluate_test.py` outputs and keys like `*_subset_accuracy@0.5` in experiment `metrics.json`):

- For each sample with at least one label where `y_mask > 0`, threshold probabilities to binary predictions.
- The sample counts as correct only if predictions equal ground truth **on every** masked-in label simultaneously (exact set match on supervised dimensions).
- The rate is `# correct samples / # samples with any supervised label`. Rows with **no** supervised labels are skipped (same spirit as masking for F1).

This is **stricter** than macro F1 (which averages per-label F1).

## Where to Read Metrics

- Organized path (primary):
  - `data/processed/experiments/<model_id>/<protocol>/<run_id>/metrics.json`
  - calibrated eval files:
    - `val_metrics_calibrated.json`
    - `test_metrics_calibrated.json`
- Registry/pointers:
  - `data/processed/experiments/<model_id>/<protocol>/runs_index.json`
  - `.../latest.json`
  - `.../best.json`
- Legacy paths are still read by report scripts for backward compatibility.

## Thresholding Modes

Two modes are present:

1. Fixed threshold (`0.5`)
- Keys like `val_macro_f1@0.5`, `test_macro_f1@0.5`

2. Per-class tuned thresholds
- Keys like `val_macro_f1@per_class_thr`, `test_macro_f1@per_class_thr`
- Thresholds saved in `data/processed/experiments/thresholds/per_class_thresholds.json`

## Why Big Gaps Can Happen

Large jumps between `@0.5` and `@per_class_thr` usually indicate calibration mismatch:
- model probabilities can be systematically low/high
- class imbalance can make low thresholds recover recall strongly
- tuning on the same validation split can overfit to that split

## Fair Comparison Protocol

Use one of these protocols and keep it identical across models:

- Protocol A: fixed 0.5 for all models (cleanest baseline comparison)
- Protocol B: tune per-class thresholds for all models on a calibration split, then evaluate on untouched holdout

Avoid:
- comparing model A at fixed `0.5` vs model B with tuned thresholds
- tuning and evaluating thresholds on the exact same split without caveat

## Calibration Threshold Protocol (4-way split)

To avoid tuning thresholds on the same data used for validation/test comparison:

1. Create a 4-way patient split:

```bash
python scripts/03_make_multilabel_splits_4way.py
```

This writes to `data/processed/splits_4way/`:
- `train_fit_rows.json`, `calib_rows.json`, `val_rows.json`, `test_rows.json`

2. Build the co-error graph from `train_fit` (for GNN variants):

```bash
python scripts/04_build_coerror_graph.py \
  --train_rows_json data/processed/splits_4way/train_fit_rows.json \
  --out_dir data/processed/graph_4way
```

3. Train a model, exporting predictions for `calib`, `val`, and `test`:

- MLP example:
```bash
python scripts/06_run_baseline_mlp.py \
  --train_rows_json data/processed/splits_4way/train_fit_rows.json \
  --val_rows_json data/processed/splits_4way/val_rows.json \
  --test_rows_json data/processed/splits_4way/test_rows.json \
  --calib_rows_json data/processed/splits_4way/calib_rows.json \
  --out_dir data/processed/experiments/mlp_calibrated
```

- GNN example:
```bash
python scripts/07_train_gnn_adapter.py \
  --train_rows_json data/processed/splits_4way/train_fit_rows.json \
  --val_rows_json data/processed/splits_4way/val_rows.json \
  --test_rows_json data/processed/splits_4way/test_rows.json \
  --calib_rows_json data/processed/splits_4way/calib_rows.json \
  --edge_index_json data/processed/graph_4way/edge_index.json \
  --edge_weight_json data/processed/graph_4way/edge_weight.json \
  --out_dir data/processed/experiments/gnn_calibrated
```

4. Tune per-class thresholds on `calib_predictions.json`, then evaluate once on untouched `val` and `test`:

```bash
python scripts/08_tune_thresholds.py \
  --val_predictions_json data/processed/experiments/mlp_calibrated/calib_predictions.json \
  --out_json data/processed/experiments/mlp_calibrated/per_class_thresholds.json

python scripts/09_evaluate_test.py \
  --test_predictions_json data/processed/experiments/mlp_calibrated/val_predictions.json \
  --thresholds_json data/processed/experiments/mlp_calibrated/per_class_thresholds.json \
  --out_json data/processed/experiments/mlp_calibrated/val_metrics_calibrated.json

python scripts/09_evaluate_test.py \
  --test_predictions_json data/processed/experiments/mlp_calibrated/test_predictions.json \
  --thresholds_json data/processed/experiments/mlp_calibrated/per_class_thresholds.json \
  --out_json data/processed/experiments/mlp_calibrated/test_metrics_calibrated.json
```

Repeat the same 3 commands for `gnn_calibrated/` using its `calib_predictions.json`.

For the other GNN variants (`scripts/12_train_clip_vlm_gnn_adapter.py` and `scripts/13_train_bipartite_gnn_adapter.py`), pass `--calib_rows_json .../calib_rows.json` during training; then run the same `08_tune_thresholds.py` + `09_evaluate_test.py` commands on their `calib_predictions.json` inside the chosen `--out_dir`.

## Recommended Reporting Format

For each experiment, report at least:
- validation macro F1 @0.5
- test macro F1 @0.5
- validation macro F1 @per-class-threshold (if used)
- test macro F1 @per-class-threshold (if used)
- threshold source split (calibration/validation)

This removes ambiguity when selecting the best model.

## Multiple Runs and Retraining

When tuning hyperparameters or retraining, keep each run isolated:
- assign a unique `--run_id` (or let scripts auto-generate one)
- keep protocol explicit via `--protocol default|calibrated4way`
- for warm-start/retrain, pass `--resume_from <checkpoint.pt>`

This allows:
- side-by-side run history in `runs_index.json`
- stable "latest" and "best" selection
- reproducible comparisons without overwriting previous runs
