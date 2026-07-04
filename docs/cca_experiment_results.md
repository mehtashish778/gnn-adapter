# CCA Experiment Results (CheXpert default protocol)

**Last updated:** 2026-05-18  
**Dataset:** `data/processed/splits/{train,val,test}_rows.json` (43,778 / 9,357 / 9,197 rows)  
**Metric:** masked **macro-F1** unless noted; primary threshold **t = 0.5**  
**VLM:** frozen Qwen2-VL scores (`x_logits`, `x_probs`) — never fine-tuned in these runs  
**Vision patches:** CLIP ViT-B/16 (`openai/clip-vit-base-patch16`), frozen or LoRA r=8  

Artifact root: `data/processed/experiments/cca/default/<run_id>/metrics.json`

All evaluation now also reports **AUROC / AUPRC / ECE / Brier** (helper: `common_multilabel.probabilistic_metrics`, wired into `scripts/cca_train_core.py` and all four baseline scripts).

---

## Executive summary

| Best single run | Test F1 @0.5 | Test AUROC | Config |
|-----------------|--------------|------------|--------|
| **`cca_lora_r8_trial27` (seed 42)** | **0.701** | — | LoRA patches + Optuna trial-27 hparams, `val_macro_f1_05` ckpt |
| **5-seed LoRA + trial-27 (mean)** | **0.701 ± 0.005** | **0.722 ± 0.004** | Same config, seeds 0–4 |
| Frozen trial-27 + F1 ckpt | 0.694 | — | `cca_frozen_trial27_f1` |
| CCA default (first run) | 0.653 | — | Frozen patches, default hparams, `val_bce` ckpt |
| QFormer adapter (baseline) | 0.676 | 0.708 | Cross-attn over CLIP patches, 30 ep |
| PostHoc CBM | 0.621 | 0.542 | Linear bottleneck over VLM features |
| Label-free CBM | 0.476 | 0.616 | CLIP concept scores → linear head |
| MLGCN | 0.470 | 0.535 | Label-graph propagation only |

**Headline:** CCA LoRA trial-27 beats every CheXpert baseline in this repo by **+2.5 F1 / +1.4 AUROC** over the strongest baseline (QFormer), with **5-seed stability σ = 0.005 F1 / 0.004 AUROC**. Concept prior ablation, calibrated 4-way protocol, and LoRA-rank scan still pending.

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

## 4. Multi-seed stability

### 4a. 5 seeds × LoRA + trial-27 hparams (the leaderboard config)

**Run prefix:** `lora_r8_trial27_seeds_s{0..4}`  
**Script:** `scripts/run_seeds.py --use_numbered_script` (with trial-27 LoRA flags, see §7)  
**Best metric:** `val_macro_f1_05`, max 60 epochs, early-stop patience 16

| Seed | Val F1 | Test F1 | Val AUROC | Test AUROC | Test AUPRC | Test ECE | Test Brier | Epochs |
|------|--------|---------|-----------|------------|------------|----------|------------|--------|
| 0 | 0.711 | 0.707 | 0.724 | 0.717 | 0.622 | 0.115 | 0.182 | 19 |
| 1 | 0.708 | 0.705 | 0.727 | 0.723 | 0.622 | 0.131 | 0.189 | 22 |
| 2 | 0.708 | 0.698 | 0.728 | 0.727 | 0.630 | 0.099 | 0.171 | 20 |
| 3 | 0.703 | 0.697 | 0.720 | 0.724 | 0.628 | 0.089 | 0.168 | 19 |
| 4 | 0.704 | 0.700 | 0.721 | 0.720 | 0.622 | 0.092 | 0.172 | 21 |
| **mean ± σ** | **0.707 ± 0.003** | **0.701 ± 0.004** | **0.724 ± 0.004** | **0.722 ± 0.004** | **0.625 ± 0.004** | **0.105 ± 0.018** | **0.176 ± 0.009** | 20.2 ± 1.3 |

Aggregate file: `data/processed/experiments/cca/default/seeds_summary.parquet` (+ `seeds_summary.json`).

### 4b. 5 seeds × frozen default CCA (legacy)

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

LoRA + trial-27 reduces seed variance from **σ ≈ 0.054 → 0.005** (≈11× tighter) while raising mean test F1 by **+0.080**.

Stats table: [`reports/comparison/stats.md`](../reports/comparison/stats.md)

---

## 5. Baselines (same default split @0.5)

Newly trained (2026-05-18) with `scripts/15_…18_train_*.py`. All write to `data/processed/experiments/<model_id>/default/<run_id>/`.

| Model | Run ID | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Params |
|-------|--------|--------------|------------|------------|----------|------------|--------|
| **CCA LoRA trial-27 (5-seed mean)** | `lora_r8_trial27_seeds_*` | **0.701** | **0.722** | **0.625** | **0.105** | **0.176** | 118,891 |
| CCA LoRA trial-27 (seed 42) | `cca_lora_r8_trial27` | 0.701 | — | — | — | — | 118,891 |
| CCA frozen trial-27 | `cca_frozen_trial27_f1` | 0.694 | — | — | — | — | 118,891 |
| CCA faithful (frozen) | `cca_faithful` | 0.674 | — | — | — | — | 435,261 |
| CCA default (frozen) | `run_20260516_183647` | 0.653 | — | — | — | — | 435,261 |
| **QFormer adapter** | `qformer_adapter_default` | 0.676 | 0.708 | 0.607 | 0.127 | 0.183 | 263,815 |
| PostHoc CBM | `cbm_posthoc_default` | 0.621 | 0.542 | 0.472 | 0.115 | 0.213 | 667 |
| Label-free CBM | `cbm_labelfree_default` | 0.476 | 0.616 | 0.528 | 0.121 | 0.213 | 217 |
| MLGCN | `mlgcn_default` | 0.470 | 0.535 | 0.468 | 0.130 | 0.219 | 8,577 |
| vlm_mlp (3 seeds) | seeds_s0–2 | 0.485 mean | — | — | — | — | — |
| baseline_mlp (legacy) | — | 0.620 | — | — | — | — | — |

**Δ to CCA LoRA trial-27 (test F1):** QFormer −0.025, CBM-PH −0.080, MLGCN −0.231, CBM-LF −0.225.  
**Δ to CCA (test AUROC):** QFormer −0.014, CBM-LF −0.106, PostHoc CBM −0.180, MLGCN −0.187.

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

## 6. Held-out-concept probe (gate column ablation)

**Script:** `scripts/20_holdout_concept.py` (sweeps all P=30 primitives; reports Δ F1 / AUROC / AUPRC / Brier on 4096-row val sample).

| Checkpoint | `use_gate_M` | Full AUROC | Max ΔAUROC | Mean ΔAUROC | Notes |
|------------|--------------|------------|------------|-------------|-------|
| `cca_lora_r8_trial27` | false | 0.702 (F1) | 0.000 | 0.000 | No gate → no holdout effect (expected) |
| `cca_lora_r8_faithful` | true | 0.637 | **4.3e-5** | 1.7e-5 | Gate density 0.42, but VLM residual dominates |
| `cca_lora_r8_default` | true | 0.620 | **2.2e-4** | 6.0e-5 | Best concept-dependence so far |
| `cca_faithful` (frozen) | true | 0.623 | 1.1e-5 | −1.3e-5 | Effectively no holdout effect |

Reports written to `reports/holdout/<checkpoint>.json` with per-primitive deltas.  
**Interpretation:** With the current `alpha=1.0` VLM residual and tight `lambda_sparse`, the readout head leans heavily on `vlm_mix` and primitives carry low independent decision mass. Lowering `alpha` or stiffening the gate's sparsity target would amplify the probe signal.

---

## 7. Concept-prior ablation (frozen patches, default CCA, 30 ep)

**Script:** `scripts/run_prior_ablation.py --gpu_id 0 --epochs 30 --run_id_prefix prior_ablation`  
**Run prefix:** `prior_ablation_<name>` under `data/processed/experiments/cca/default/`  
**Prior matrices:** `data/processed/graph/prior_ablation/*.json` (P=30)  

| Variant | Source | Val F1 | Test F1 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Epochs |
|---------|--------|--------|---------|------------|------------|----------|------------|--------|
| `none` | no `radgraph_bias` term | **0.693** | **0.683** | 0.676 | 0.576 | 0.110 | 0.183 | 24 |
| `co_occur` | label co-occurrence (P=30, train) | 0.660 | 0.649 | 0.670 | 0.569 | 0.097 | 0.184 | 22 |
| `coerror` | normalized co-error matrix | 0.660 | 0.650 | 0.684 | 0.583 | 0.101 | 0.179 | 29 |
| `radgraph` | **stub** = copy of `co_occur` | 0.660 | 0.649 | 0.670 | 0.569 | 0.097 | 0.184 | 22 |
| `permuted` | row/col-shuffle of `radgraph` (control) | 0.656 | 0.650 | **0.689** | **0.590** | 0.102 | **0.178** | 26 |

**Findings:**
1. **`none` wins on F1** (+0.033 over best prior) — adding any P×P bias actually *hurts* test F1 at this scale.
2. `permuted` ≥ informative priors on AUROC/AUPRC — i.e. random structure is no worse than co-occurrence or co-error. The signal carried by these priors is below the noise floor of a P=30 / 30-epoch run.
3. `co_occur ≡ radgraph` because the RadGraph file is a placeholder copy until we wire the true MIMIC-CXR RadGraph entity graph (`source: radgraph_placeholder_cooccurrence`).
4. Calibration is slightly *better* with any prior (ECE 0.097 vs 0.110), at the cost of F1.

> **Headline:** With the current P=30 concept vocabulary, the compositional self-attention layer already absorbs label-structure signal; explicit priors are redundant on CheXpert. A future replacement with a real RadGraph entity matrix may change this — but is gated on RadGraph parsing infrastructure.

---

## 8. Phase 3 / 4 items (still pending)

| Item | Status |
|------|--------|
| Concept prior ablation (none / co-occur / co-error / radgraph-stub / permuted) | **Done** (see §7) |
| `mlgcn`, `qformer_adapter`, `cbm_posthoc`, `cbm_labelfree` | **Done** (see §5) |
| Held-out-concept eval | **Done** (see §6) |
| 5-seed LoRA + trial-27 | **Done** (see §4a) |
| AUROC / AUPRC / ECE / Brier in metrics writer | **Done** (`scripts/common_multilabel.probabilistic_metrics`) |
| LoRA ranks 4 / 16 | Not trained |
| CCA on `calibrated4way` | Not run |
| True RadGraph prior (replace stub) | Stub only |

---

## 9. Reproduction commands

**Full reproduction guide (every leaderboard row + hyperparameter reference):** [`docs/cca_reproduction.md`](cca_reproduction.md)

Quick start:

```powershell
# Best model (LoRA + trial-27, seed 42)
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27 --best_metric val_macro_f1_05 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text

# 5-seed sweep with trial-27 LoRA hparams
python scripts/run_seeds.py --model_id cca --protocol default --seeds 0,1,2,3,4 `
  --run_id_prefix lora_r8_trial27_seeds --use_numbered_script `
  -- --gpu_id 0 --num_workers 0 --lora_rank 8 --num_primitives 30 --query_dim 64 `
  --n_cross_attn_layers 1 --n_self_attn_layers 2 --n_heads 4 --alpha 0.5 --dropout 0.1001 `
  --lr 0.000479 --weight_decay 0.000111 --batch_size 8 --epochs 60 --early_stop_patience 16 `
  --best_metric val_macro_f1_05 --no-use_gate_M --init_queries_from_text

# Four baselines
python scripts/15_train_posthoc_cbm.py --model_id cbm_posthoc --protocol default --run_id cbm_posthoc_default --gpu_id 0 --epochs 25 --lr 1e-3 --num_concepts 30
python scripts/16_train_labelfree_cbm.py --model_id cbm_labelfree --protocol default --run_id cbm_labelfree_default --gpu_id 0 --epochs 25 --lr 1e-3 --num_concepts 30 --clip_batch_size 32
python scripts/17_train_qformer_adapter.py --model_id qformer_adapter --protocol default --run_id qformer_adapter_default --gpu_id 0 --num_workers 0 --epochs 30 --lr 3e-4 --batch_size 16 --num_queries 32 --query_dim 128 --n_heads 4 --n_cross_attn_layers 2 --dropout 0.1
python scripts/18_train_mlgcn.py --model_id mlgcn --protocol default --run_id mlgcn_default --gpu_id 0 --epochs 30 --lr 1e-3

# Held-out-concept sweep (any CCA checkpoint with use_gate_M=True)
python scripts/20_holdout_concept.py `
  --checkpoint data/processed/experiments/cca/default/cca_lora_r8_faithful/best_checkpoint.pt `
  --lora_rank 8 --num_workers 0 --gpu_id 0 `
  --summary_json reports/holdout/cca_lora_r8_faithful.json

# Concept prior ablation (5 variants × 30 ep; ~2.5h on RTX 4060)
python scripts/run_prior_ablation.py --gpu_id 0 --epochs 30 --run_id_prefix prior_ablation

# All LoRA variants + comparison table
python scripts/run_cca_lora_variants.py --gpu_id 0 --skip_existing
python scripts/run_cca_lora_variants.py --compare_only

# Optuna (long)
python scripts/tune_cca_optuna.py --model_id cca --protocol default --gpu_id 0 `
  --num_workers 0 --n_trials 20 --tune_epochs 25 --final_epochs 60
```

---

## 10. Related documentation

| Document | Content |
|----------|---------|
| [`docs/cca_reproduction.md`](cca_reproduction.md) | **Full reproduction recipe + complete hyperparameter reference** |
| [`docs/cca_optuna_hpo.md`](cca_optuna_hpo.md) | Optuna search space, trial-27, CLI |
| [`reports/comparison/cca_lora_variants.md`](../reports/comparison/cca_lora_variants.md) | LoRA variant table |
| [`reports/comparison/cca_optuna_summary.md`](../reports/comparison/cca_optuna_summary.md) | Short Optuna summary |
| [`reports/comparison/cca_prior_ablation.md`](../reports/comparison/cca_prior_ablation.md) | Concept-prior ablation table |
| [`reports/comparison/stats.md`](../reports/comparison/stats.md) | Multi-seed bootstrap |
| [`reports/holdout/`](../reports/holdout/) | Per-checkpoint held-out-concept JSONs |
| [`configs/train_cca.yaml`](../configs/train_cca.yaml) | Training presets |
| [`docs/pipeline.md`](pipeline.md) | Script map |
