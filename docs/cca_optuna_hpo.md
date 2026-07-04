# CCA Optuna hyperparameter optimization

**All experiment results:** [`docs/cca_experiment_results.md`](cca_experiment_results.md)  
**Full reproduction recipe (env, data, every run, all hyperparameters):** [`docs/cca_reproduction.md`](cca_reproduction.md)

This document records the first Optuna study for the Concept-Evidence Adapter (CCA) on the CheXpert **default** split (`train` / `val` / `test`). Implementation: `scripts/tune_cca_optuna.py` (training core: `scripts/cca_train_core.py`).

## How to run

```powershell
cd C:\Users\MYPC\mbzai
$env:PYTHONPATH = "scripts"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
python scripts/tune_cca_optuna.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --n_trials 20 --tune_epochs 25 --tune_early_stop_patience 8 `
  --final_epochs 60 --final_early_stop_patience 16
```

Or: `.\scripts\run.ps1 cca_optuna`

**Resume** an interrupted study (same SQLite DB):

```powershell
python scripts/tune_cca_optuna.py ...  # same --study_name cca_hpo and --storage URL
```

**Tuning only** (skip final 60-epoch train): add `--skip_final_train`.

## Study setup

| Setting | Value |
|---------|--------|
| Study name | `cca_hpo` |
| Storage | `sqlite:///data/processed/experiments/cca/optuna/study.db` |
| Objective | Maximize `val_macro_f1@0.5` |
| Sampler | TPE (`seed=42`) |
| Pruner | Median (warmup 5 epochs, startup 3 trials) |
| Trials per run (2026-05-17) | 20 new (+ prior trials in DB → 40 total) |
| Patch cache | Loaded once per process (`patch_v2_fp16`, ~17.5 GB) |
| Checkpoint metric during train | `val_bce` (default; differs from Optuna objective) |

### Search space

| Hyperparameter | Search |
|----------------|--------|
| `num_primitives` | {15, 30, 50} |
| `query_dim` | {64, 128, 192} |
| `n_cross_attn_layers` | 1–2 |
| `n_self_attn_layers` | 1–2 |
| `n_heads` | {2, 4} |
| `alpha` | {0.5, 1.0} |
| `dropout` | uniform 0.05–0.25 |
| `lr` | log-uniform 1e-4–5e-4 |
| `weight_decay` | log-uniform 1e-5–1e-3 |
| `batch_size` | {8, 16, 32} |
| `use_gate_M` | {true, false} |
| `init_queries_from_text` | {true, false} |

Trials with trainable params ≥ 1M are pruned. CUDA OOM trials are pruned.

## Results (2026-05-17)

Hardware: RTX 4060 (8 GB). Wall time for 20 trials + final train: ~3 h.

### Best tuning trial (trial 27)

This is the **peak F1** observed in the study (short 25-epoch budget, 13 epochs ran).

| Metric | Value |
|--------|-------|
| Val macro-F1 @0.5 | **0.701** |
| Test macro-F1 @0.5 | **0.691** |
| Trainable params | **118,891** |

**Hyperparameters:**

```yaml
num_primitives: 30
query_dim: 64
n_cross_attn_layers: 1
n_self_attn_layers: 2
n_heads: 4
alpha: 0.5
dropout: 0.1001
lr: 0.000479
weight_decay: 0.000111
batch_size: 8
use_gate_M: false
init_queries_from_text: true
```

### Final model (`best_optuna_cca_hpo`)

After Optuna, the script retrains with best params for up to 60 epochs (`early_stop_patience=16`), still selecting checkpoints on **`val_bce`**.

| Metric | Value |
|--------|-------|
| Val macro-F1 @0.5 | 0.665 |
| Test macro-F1 @0.5 | 0.658 |
| Test macro-F1 @per_class_thr | 0.664 |
| Epochs ran | 30 (early stop) |
| Trainable params | 118,891 |

**Gap vs trial 27:** final val/test F1 are ~0.036 / ~0.033 lower because longer training + BCE-based checkpoint selection does not match the F1 objective used by Optuna. For deployment, consider retraining with `--best_metric val_macro_f1_05` or saving the val-F1-best epoch explicitly.

### Other strong trials (validation @0.5)

| Trial | Val F1 | Notes |
|-------|--------|--------|
| 27 | 0.701 | Best overall |
| 13 | 0.698 | `query_dim=192` (larger model) |
| 25 | 0.698 | Same family as 27, lower lr |
| 35 | 0.694 | Lower dropout |
| 7, 22 | ~0.692 | Early good region in search |

### Baseline comparison (default split, test macro-F1 @0.5)

| Model | Test F1 @0.5 | Source |
|-------|--------------|--------|
| **CCA Optuna trial 27** | **0.691** | `best_trial.json` user attrs |
| CCA Optuna final | 0.658 | `best_optuna_cca_hpo/metrics.json` |
| CCA default hparams | 0.653 | `run_20260516_183647/metrics.json` |
| gnn13_clip_bipartite | 0.637 | `docs/academic_report.md` (paper repro) |
| gnn12_clip_vlm_homo | 0.601 | same |
| vlm_mlp | 0.519 | same |

**Calibrated 4-way** (leakage-free test, per-class thresholds on `calib`): gnn13 **0.689** — CCA has not been tuned/evaluated on that protocol yet.

## Design lessons

1. **Smaller adapter helped:** D=64, one cross-attention layer, ~119K params beat the ~435K default (D=128, two cross layers).
2. **Weaker VLM residual:** `alpha=0.5` preferred over `1.0`.
3. **`use_gate_M=false`** often won; sparse C×P gate was not needed for best val F1 in this sweep.
4. **Align checkpoint metric with HPO objective** when running final training to avoid losing trial-level gains.

## Artifact paths

| File | Description |
|------|-------------|
| `data/processed/experiments/cca/optuna/study.db` | Optuna SQLite study |
| `data/processed/experiments/cca/optuna/best_trial.json` | Best trial summary + params |
| `data/processed/experiments/cca/optuna/final_metrics.json` | Final 60-epoch run metrics |
| `data/processed/experiments/cca/default/best_optuna_cca_hpo/` | Checkpoint, predictions, history |
| `data/processed/experiments/cca/default/run_20260516_183647/` | Pre-HPO default CCA run |

## LoRA CLIP patches + CCA

Build LoRA-adapted ViT patches (train/val/test):

```powershell
python scripts/19_train_lora_clip_vision.py --lora_r 8 --gpu_id 0
```

If train/val were cached earlier without test, add test only:

```powershell
python scripts/19_train_lora_clip_vision.py --lora_r 8 --gpu_id 0 --encode_only
```

Train CCA on LoRA patches (same hparams as frozen; fair comparison):

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27 --best_metric val_macro_f1_05 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use-gate-M --init_queries_from_text
```

Frozen baseline uses default patch cache (`--lora_rank` omitted).

## Phase 2 faithfulness training

Enable sparse Gumbel gate + intervention loss:

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --use_gate_M --lambda_sparse 0.01 --lambda_faithful 0.1 `
  --best_metric val_macro_f1_05 --run_id cca_faithful
```

CLI flags: `--lambda_sparse`, `--lambda_faithful`, `--sparsity_target`, `--gumbel_tau_init`, `--gumbel_tau_min`, `--gumbel_anneal_epochs`, `--intervention_per_step`.

## Concept prior ablation

```powershell
# Build priors only
python scripts/run_prior_ablation.py --dry_run

# Train CCA with none / co-occur / co-error / radgraph / permuted priors
python scripts/run_prior_ablation.py --gpu_id 0 --epochs 30 --run_id_prefix prior_ablation
```

Or stepwise: `scripts/build_concept_prior.py`, `scripts/permute_prior.py`, then `14_train_cca.py --radgraph_prior_json <path>`.

## Recommended next steps

1. Retrain best params with `--best_metric val_macro_f1_05 --run_id cca_best_f1`.
2. Multi-seed stability: `python scripts/run_seeds.py --model_id cca --protocol default --seeds 0,1,2,3,4 --use_numbered_script -- ...best hparams...`
3. Run **calibrated4way** with tuned hparams for fair comparison to gnn13 (0.689 calibrated test).
4. Register best config as a named preset in `configs/train_cca.yaml` once validated across seeds.

## Retrain with best hyperparameters (manual)

```powershell
$env:PYTHONPATH = "scripts"
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --run_id cca_trial27_best --best_metric val_macro_f1_05 --epochs 60 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use-gate-M --init_queries_from_text
```
