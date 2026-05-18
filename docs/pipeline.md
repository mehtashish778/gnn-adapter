# Pipeline Documentation

For the full methodological narrative and evaluation framing, see **`docs/academic_report.md`** (canonical academic reference).

This document maps each script to its inputs/outputs so you can debug or rerun stages independently.

## Canonical model IDs

- `vlm_zeroshot` -> `VLMZeroShot`
- `vlm_mlp` -> `VLMFeatureMLP`
- `gnn07_label_residual` -> `LabelGraphResidualGNN`
- `gnn12_clip_vlm_homo` -> `ClipVlmHomogeneousGNN`
- `gnn13_clip_bipartite` -> `ClipBipartiteAttributeGNN`
- `cca` -> `CCAModel` (Concept-Evidence Adapter / CCA)
- `cbm_posthoc`, `cbm_labelfree`, `qformer_adapter`, `mlgcn` -> Phase 4 baselines

Organized output path:
- `data/processed/experiments/<model_id>/<protocol>/<run_id>/...`
- `protocol in {default, calibrated4way}`

## 1) Canonical Labels

Script: `scripts/01_build_canonical_labels.py`  
Goal: define canonical label space and normalization for multi-label tasks.

## 2) Align VLM Outputs

Script: `scripts/02_align_vlm_outputs.py`  
Goal: align external VLM predictions with dataset rows using `join_key` from `configs/data.yaml`.

Expected artifacts include aligned rows with:
- `x_logits`
- `x_probs`
- `y_true`
- `y_mask`

## 3) Train/Val/Test Splits

Script: `scripts/03_make_multilabel_splits.py`  
Goal: build row-wise split files consumed by all downstream experiments.

Primary outputs:
- `data/processed/splits/train_rows.json`
- `data/processed/splits/val_rows.json`
- `data/processed/splits/test_rows.json`

## 4) Co-Error Graph

Script: `scripts/04_build_coerror_graph.py`  
Config: `configs/graph.yaml`  
Goal: build label graph edges/weights for graph-based adapters.

Primary outputs:
- `data/processed/graph/edge_index.json`
- `data/processed/graph/edge_weight.json`

## 5) Baselines

### Frozen VLM Baseline
Script: `scripts/05_run_baseline_frozen_vlm.py`
Wrapper: `scripts/models/vlm_zeroshot/run_default.py`

Output:
- `data/processed/experiments/baseline_frozen_vlm/metrics.json`

### MLP Baseline
Script: `scripts/06_run_baseline_mlp.py`  
Model: MLP over flattened `[x_logits, x_probs]` features.
Wrapper: `scripts/models/vlm_mlp/train.py`

Output:
- `data/processed/experiments/baseline_mlp/metrics.json`

## 6) GNN Training Variants

### Residual Label-Graph Adapter
Script: `scripts/07_train_gnn_adapter.py`  
Config reference: `configs/train_gnn.yaml`
Wrapper: `scripts/models/gnn07_label_residual/train.py`

Outputs:
- `data/processed/experiments/gnn_adapter/metrics.json`
- `data/processed/experiments/gnn_adapter/val_predictions.json`
- `data/processed/experiments/gnn_adapter/test_predictions.json`
- `data/processed/experiments/gnn_adapter/best_checkpoint.pt`

### CLIP + VLM GNN Adapter
Script: `scripts/12_train_clip_vlm_gnn_adapter.py`  
Config reference: `configs/train_clip_gnn.yaml`
Wrapper: `scripts/models/gnn12_clip_vlm_homo/train.py`

Output directory:
- `data/processed/experiments/clip_vlm_gnn_adapter/`

### Bipartite GNN Adapter
Script: `scripts/13_train_bipartite_gnn_adapter.py`  
Helper module: `scripts/gnn_bipartite.py`
Wrapper: `scripts/models/gnn13_clip_bipartite/train.py`

Output directory:
- `data/processed/experiments/bipartite_clip_gnn_adapter/`

### One-shot run for all variants
Script: `scripts/run_all_gnn_variants.sh`

### Concept-Evidence Adapter (CCA)
Script: `scripts/14_train_cca.py` (core: `scripts/cca_train_core.py`)  
Config reference: `configs/train_cca.yaml`  
Optuna HPO: `scripts/tune_cca_optuna.py` — full write-up in **`docs/cca_optuna_hpo.md`**  
Faithfulness metrics: `scripts/faithfulness_metrics.py`  
Concept priors: `scripts/build_concept_prior.py`, `scripts/permute_prior.py`, ablation driver `scripts/run_prior_ablation.py`

Outputs:
- `data/processed/experiments/cca/<protocol>/<run_id>/best_checkpoint.pt`
- `metrics.json`, `val_predictions.json`, `test_predictions.json`, `attention_maps.pt`
- Patch cache: `data/processed/embeddings/*_patch_v2_fp16.pt`

**Full results log:** [`docs/cca_experiment_results.md`](cca_experiment_results.md)  
**Reproduction recipe + hyperparameter reference:** [`docs/cca_reproduction.md`](cca_reproduction.md)

Documented runs (default protocol):
- **`cca_lora_r8_trial27`** — best overall (test F1 @0.5 ≈ **0.701**, LoRA patches + trial-27)
- `cca_frozen_trial27_f1` — frozen patches + trial-27 + F1 ckpt (test ≈ 0.694)
- `run_20260516_183647` — default hparams (test F1 @0.5 ≈ 0.653)
- `cca_faithful` — default + faithfulness (test ≈ 0.674)
- `best_optuna_cca_hpo` — Optuna trial-27 hparams, 60-epoch final, val_bce ckpt (test ≈ 0.658)
- `data/processed/experiments/cca/optuna/best_trial.json` — best tuning trial (val F1 @0.5 ≈ 0.701)
- LoRA variant table: [`reports/comparison/cca_lora_variants.md`](../reports/comparison/cca_lora_variants.md)
- Concept-prior ablation table: [`reports/comparison/cca_prior_ablation.md`](../reports/comparison/cca_prior_ablation.md)
- 5-seed leaderboard config (`lora_r8_trial27_seeds_s{0..4}`): see [`docs/cca_experiment_results.md §4a`](cca_experiment_results.md#4a-5-seeds--lora--trial-27-hparams-the-leaderboard-config)
- Baselines (`cbm_posthoc`, `cbm_labelfree`, `qformer_adapter`, `mlgcn`): [`docs/cca_experiment_results.md §5`](cca_experiment_results.md#5-baselines-same-default-split-05)
- Held-out-concept JSONs: `reports/holdout/cca_lora_r8_*.json`, `reports/holdout/cca_frozen_faithful.json`

### Phase 4 baselines (same patch cache as CCA)
| Script | model_id |
|--------|----------|
| `scripts/15_train_posthoc_cbm.py` | `cbm_posthoc` |
| `scripts/16_train_labelfree_cbm.py` | `cbm_labelfree` |
| `scripts/17_train_qformer_adapter.py` | `qformer_adapter` |
| `scripts/18_train_mlgcn.py` | `mlgcn` |
| `scripts/19_train_lora_clip_vision.py` | LoRA vision cache (`peft`) |
| `scripts/20_holdout_concept.py` | held-out primitive ablation eval |

Multi-seed stats: `scripts/run_seeds.py --use_numbered_script --stats_after` → `scripts/stats_compare.py`

## 7) Thresholds, Evaluation, and Reporting

### Tune thresholds
Script: `scripts/08_tune_thresholds.py`

Output:
- `data/processed/experiments/thresholds/per_class_thresholds.json`

### Evaluate
Script: `scripts/09_evaluate_test.py`  
Config reference: `configs/eval.yaml`

Output:
- `data/processed/experiments/final_eval/test_metrics.json`

### Build ablations table
Script: `scripts/10_run_ablations.py`

Outputs:
- `data/processed/experiments/ablations/ablation_table.csv`
- `data/processed/experiments/ablations/ablation_notes.json`

### Package markdown report
Script: `scripts/11_package_report.py`

Output:
- `reports/gnn_adapter/report.md`

## GPU Notes

- Training scripts enforce CUDA availability.
- Set `--gpu_id` for Python scripts.
- For the shell runner:

```bash
GPU_ID=0 bash scripts/run_all_gnn_variants.sh
```

## Run Versioning

Training/eval scripts support:
- `--model_id`
- `--protocol`
- `--run_id`
- `--resume_from` (train scripts)

After each run (when model/protocol are provided), metadata is tracked in:
- `runs_index.json`
- `latest.json`
- `best.json`
