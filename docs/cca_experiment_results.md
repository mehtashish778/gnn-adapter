# CCA Experiment Results (CheXpert default protocol)

**Last updated:** 2026-05-17  
**Dataset:** `data/processed/splits/{train,val,test}_rows.json` (43,778 / 9,357 / 9,197 rows)  
**Metric:** masked **macro-F1** unless noted; primary threshold **t = 0.5**  
**VLM:** frozen Qwen2-VL scores (`x_logits`, `x_probs`) — never fine-tuned in these runs  
**Vision patches:** CLIP ViT-B/16 (`openai/clip-vit-base-patch16`), frozen or LoRA r=8  

Artifact root: `data/processed/experiments/cca/default/<run_id>/metrics.json`

---

## Executive summary

| Best single run | Test F1 @0.5 | Config |
|-----------------|--------------|--------|
| **`cca_lora_r8_trial27`** | **0.701** | LoRA patches + Optuna trial-27 hparams, `val_macro_f1_05` checkpoint |
| Optuna trial 27 (tuning only) | 0.691 | Frozen patches, 25-epoch tune |
| Frozen trial-27 + F1 ckpt | 0.694 | `cca_frozen_trial27_f1` |
| CCA default (first run) | 0.653 | Frozen patches, default hparams, `val_bce` ckpt |

**Recommendation for reporting:** `cca_lora_r8_trial27` with trial-27 hyperparameters and `--best_metric val_macro_f1_05`. Calibrated 4-way comparison to `gnn13` (0.689 calibrated test) still pending.

---

## 1. Optuna hyperparameter search

**Study:** `cca_hpo` — `data/processed/experiments/cca/optuna/study.db`  
**Script:** `scripts/tune_cca_optuna.py` (20 trials requested, 40 logged with prunes)  
**Objective:** `val_macro_f1@0.5` during 25-epoch tuning  

### Best trial (27)

| | Value |
|---|--------|
| Val F1 @0.5 (tune) | **0.701** |
| Test F1 @0.5 (tune) | **0.691** |
| Trainable params | 118,891 |

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `num_primitives` | 30 |
| `query_dim` | 64 |
| `n_cross_attn_layers` | 1 |
| `n_self_attn_layers` | 2 |
| `n_heads` | 4 |
| `alpha` | 0.5 |
| `dropout` | 0.1001 |
| `lr` | 4.79e-4 |
| `weight_decay` | 1.11e-4 |
| `batch_size` | 8 |
| `use_gate_M` | false |
| `init_queries_from_text` | true |

### Optuna final train (60 epochs)

| Run ID | Checkpoint | Val F1 | Test F1 | Notes |
|--------|------------|--------|---------|-------|
| `best_optuna_cca_hpo` | `val_bce` | 0.665 | 0.658 | Same trial-27 hparams; BCE checkpoint hurts vs tune |

See also: [`docs/cca_optuna_hpo.md`](cca_optuna_hpo.md), [`data/processed/experiments/cca/optuna/best_trial.json`](../data/processed/experiments/cca/optuna/best_trial.json)

---

## 2. LoRA CLIP patch variants (2026-05-17)

**Patch cache:** `openai_clip-vit-base-patch16_lora_r8` / `patch_v2_fp16_lora_r8`  
**Script:** `scripts/run_cca_lora_variants.py`  
**All runs:** seed 42, `--best_metric val_macro_f1_05`, 60 max epochs (early stop)

| Run ID | Patches | Val F1 @0.5 | Test F1 @0.5 | Test F1 @thr | Params | Epochs |
|--------|---------|---------------|--------------|--------------|--------|--------|
| **`cca_lora_r8_trial27`** | LoRA | **0.704** | **0.701** | 0.682 | 118,891 | 23 |
| `cca_lora_r8_trial27_faithful` | LoRA | 0.704 | 0.701 | 0.682 | 118,891 | 23 |
| `cca_frozen_trial27_f1` | Frozen | 0.702 | 0.694 | 0.670 | 118,891 | 34 |
| `cca_lora_r8_default` | LoRA | 0.683 | 0.677 | 0.655 | 435,261 | 18 |
| `cca_lora_r8_faithful` | LoRA | 0.686 | 0.677 | 0.659 | 435,261 | 24 |

### Δ test F1 vs frozen trial-27 (0.694)

| Run | Δ |
|-----|---|
| LoRA trial-27 | **+0.007** |
| LoRA trial-27 + faithfulness | +0.007 |
| LoRA default | −0.017 |
| LoRA faithful (default arch) | −0.017 |

### Variant descriptions

- **trial27:** Optuna best architecture, no gate, no faithfulness loss.
- **trial27_faithful:** Same + `lambda_sparse=0.01`, `lambda_faithful=0.1` (no gate → no effect on F1).
- **default:** `query_dim=128`, 2× cross/self layers, gate on, batch 16.
- **faithful:** Default + faithfulness losses + gate.

Short report: [`reports/comparison/cca_lora_variants.md`](../reports/comparison/cca_lora_variants.md)

---

## 3. Single-seed CCA runs (frozen patches)

| Run ID | Checkpoint | Val F1 | Test F1 | Params | Epochs |
|--------|------------|--------|---------|--------|--------|
| `run_20260516_183647` | val_bce | 0.654 | 0.653 | 435,261 | — |
| `cca_faithful` | val_macro_f1_05 | 0.677 | 0.674 | 435,261 | 18 |
| `best_optuna_cca_hpo` | val_bce | 0.665 | 0.658 | 118,891 | 30 |
| `cca_frozen_trial27_f1` | val_macro_f1_05 | 0.702 | 0.694 | 118,891 | 34 |

### Faithfulness (`cca_faithful`, frozen patches)

| Metric | Value |
|--------|-------|
| `lambda_sparse` / `lambda_faithful` | 0.01 / 0.1 |
| Gate density (eval) | 0.438 |
| Intervention consistency | 0.554 |
| Necessity drop | 0.290 |
| Sufficiency F1 | 0.675 |

---

## 4. Multi-seed stability (frozen default hparams)

**Script:** `scripts/run_seeds.py --use_numbered_script --stats_after`  
**Config:** default CCA (~435K params), **frozen** patches, `val_bce` checkpoint  

| Seed | Val F1 | Test F1 |
|------|--------|---------|
| 0 | 0.660 | 0.652 |
| 1 | 0.546 | 0.541 |
| 2 | 0.605 | 0.598 |
| 3 | **0.692** | **0.679** |
| 4 | 0.638 | 0.636 |

| Aggregate | Test F1 |
|-----------|---------|
| Mean | 0.621 |
| 95% CI (bootstrap) | [0.575, 0.660] |

High variance across seeds; not yet repeated for LoRA + trial-27.

Stats table: [`reports/comparison/stats.md`](../reports/comparison/stats.md)

---

## 5. Baselines (same default split @0.5)

| Model | Run(s) | Test F1 @0.5 | Notes |
|-------|--------|--------------|-------|
| **CCA LoRA trial-27** | `cca_lora_r8_trial27` | **0.701** | This work |
| CCA frozen trial-27 | `cca_frozen_trial27_f1` | 0.694 | |
| CCA 5-seed best (seed 3) | `seeds_s3` | 0.679 | frozen default |
| CCA faithful | `cca_faithful` | 0.674 | frozen default |
| CCA default | `run_20260516_183647` | 0.653 | |
| CCA 5-seed mean | seeds 0–4 | 0.621 | |
| vlm_mlp (3 seeds) | seeds_s0–2 | 0.485 mean | [0.432, 0.587] CI |
| vlm_mlp | `fix_test` | 0.532 | single run |
| baseline_mlp (legacy path) | — | 0.620 | |
| gnn07 (legacy artifact) | — | 0.047 | stale/broken path |

### Historical GNN adapters (calibrated 4-way protocol)

From [`docs/academic_report.md`](academic_report.md) / `reports/comparison/overall.json`:

| Model | Calibrated test macro-F1 |
|-------|--------------------------|
| gnn13_clip_bipartite | **0.689** |
| gnn12_clip_vlm_homo | 0.678 |
| vlm_mlp | 0.654 |
| frozen VLM (calibrated) | 0.651 |

*Not directly comparable to CCA @0.5 on default split without running CCA on `calibrated4way`.*

---

## 6. Phase 3 / 4 items (not fully evaluated)

| Item | Status |
|------|--------|
| Concept prior ablation (none/co-occur/co-error/permuted) | Priors built (`data/processed/graph/prior_ablation/`); CCA trains not run |
| `mlgcn`, `qformer_adapter`, `cbm_*` | Scripts exist; no `metrics.json` in registry |
| LoRA ranks 4 / 16 | Not trained |
| 5-seed LoRA + trial-27 | Not run |
| CCA on `calibrated4way` | Not run |

---

## 7. Reproduction commands

**Full reproduction guide (every leaderboard row + hyperparameter reference):** [`docs/cca_reproduction.md`](cca_reproduction.md)

Quick start:

```powershell
# Best model (LoRA + trial-27)
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27 --best_metric val_macro_f1_05 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text

# All LoRA variants + comparison table
python scripts/run_cca_lora_variants.py --gpu_id 0 --skip_existing
python scripts/run_cca_lora_variants.py --compare_only

# Optuna (long)
python scripts/tune_cca_optuna.py --model_id cca --protocol default --gpu_id 0 `
  --num_workers 0 --n_trials 20 --tune_epochs 25 --final_epochs 60
```

---

## 8. Related documentation

| Document | Content |
|----------|---------|
| [`docs/cca_reproduction.md`](cca_reproduction.md) | **Full reproduction recipe + complete hyperparameter reference** |
| [`docs/cca_optuna_hpo.md`](cca_optuna_hpo.md) | Optuna search space, trial-27, CLI |
| [`reports/comparison/cca_lora_variants.md`](../reports/comparison/cca_lora_variants.md) | LoRA variant table |
| [`reports/comparison/cca_optuna_summary.md`](../reports/comparison/cca_optuna_summary.md) | Short Optuna summary |
| [`reports/comparison/stats.md`](../reports/comparison/stats.md) | Multi-seed bootstrap |
| [`configs/train_cca.yaml`](../configs/train_cca.yaml) | Training presets |
| [`docs/pipeline.md`](pipeline.md) | Script map |
