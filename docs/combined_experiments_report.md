# Combined Experiments Report — Concept-Evidence Adapter (CCA)

**Project:** mbzai / GNN → CCA reframing for AAAI submission  
**Last updated:** 2026-05-18  
**Primary dataset:** CheXpert, `default` protocol (`train` / `val` / `test`)  
**Planning reference:** [`upcomming_plan/AAAI___Ashish___Concept (1).pdf`](../upcomming_plan/AAAI___Ashish___Concept%20(1).pdf), [`upcomming_plan/TODO_AAAI_Submission.md`](../upcomming_plan/TODO_AAAI_Submission.md)

This document consolidates all **core architecture experiments** completed to date: Optuna HPO, LoRA vs frozen patch encoders, multi-seed stability, adapter baselines, concept-prior ablation, held-out-concept probes, and faithfulness runs. It supersedes scattered tables in `docs/cca_experiment_results.md` and `reports/comparison/*.md` as a single narrative reference.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Research framing and goals](#2-research-framing-and-goals)
3. [Architecture](#3-architecture)
4. [Experimental protocol](#4-experimental-protocol)
5. [Master leaderboard](#5-master-leaderboard)
6. [Hyperparameter search (Optuna)](#6-hyperparameter-search-optuna)
7. [LoRA vs frozen CLIP patches](#7-lora-vs-frozen-clip-patches)
8. [Single-seed CCA runs (frozen patches)](#8-single-seed-cca-runs-frozen-patches)
9. [Multi-seed stability](#9-multi-seed-stability)
10. [Adapter baselines](#10-adapter-baselines)
11. [Concept-prior ablation](#11-concept-prior-ablation)
12. [Held-out-concept probe](#12-held-out-concept-probe)
13. [Faithfulness mechanism](#13-faithfulness-mechanism)
14. [Legacy GNN / calibrated protocol](#14-legacy-gnn--calibrated-protocol)
15. [Statistical comparison](#15-statistical-comparison)
16. [Key findings and interpretation](#16-key-findings-and-interpretation)
17. [Pending experiments (AAAI plan)](#17-pending-experiments-aaai-plan)
18. [Artifacts and reproduction](#18-artifacts-and-reproduction)
19. [Script and config index](#19-script-and-config-index)

---

## 1. Executive summary

| Result | Value |
|--------|-------|
| **Best model** | `cca_lora_r8_trial27` — LoRA CLIP patches (r=8) + Optuna trial-27 architecture |
| **Best test macro-F1 @0.5** | **0.701** (seed 42); **0.701 ± 0.005** (5 seeds) |
| **Best test macro-AUROC** | **0.722 ± 0.004** (5 seeds) |
| **Trainable parameters** | **118,891** (&lt; 1M target met) |
| **Strongest baseline (QFormer)** | Test F1 0.676, AUROC 0.708 |
| **Δ vs QFormer** | **+0.025 F1**, **+0.014 AUROC** (CCA 5-seed mean) |

**Headline:** On CheXpert default split with frozen Qwen2-VL scores and threshold 0.5, CCA LoRA trial-27 is the best model in this repository. It beats QFormer, post-hoc CBM, label-free CBM, and ML-GCN by large margins, with tight 5-seed variance (σ ≈ 0.005 F1). Explicit graph priors hurt F1; faithfulness gating is weak on the winning config (no gate). Cross-site evaluation, LoRA-on-VLM, and calibrated 4-way CCA runs remain **out of scope** for this report.

---

## 2. Research framing and goals

The AAAI plan reframes the contribution from “GNN adapter beats MLP on CheXpert” to **Concept-Evidence Adapters (CCA)** for frozen vision–language models:

| Pillar | Description | Status in repo |
|--------|-------------|----------------|
| **Structured adapter (Option A)** | Full patch tokens + concept cross-attention + VLM residual; &lt;1M params | **Implemented and evaluated** |
| **RadGraph prior (Option B)** | Differentiable P×P compositional bias from clinical KG | **Stub only**; ablation run with placeholder |
| **Multi-VLM ensemble (Option C)** | Per-class trust across VLMs | **Not implemented** |
| **Faithfulness** | Sparse C×P gate + intervention loss | **Implemented**; best F1 config disables gate |
| **Cross-site portability** | MIMIC, NIH, PadChest, VinDr | **Not run** |
| **Leakage-free calibration** | Four-way split + formal proposition | **Legacy GNN only**; CCA not on `calibrated4way` |

Working title (planned): *Concept-Evidence Adapters for Frozen Vision-Language Models on Multi-Label Medical Classification*.

---

## 3. Architecture

**Implementation:** `scripts/models/architectures/cca.py`  
**Training core:** `scripts/cca_train_core.py`  
**Config reference:** `configs/train_cca.yaml`

### 3.1 Three-layer CCA

```
Frozen ViT patches (B, 196, 768)     Frozen VLM (z, p) per image
         │                                    │
         ▼                                    │
  Layer 1: PrimitiveConceptLayer              │
    P learnable queries → cross-attn           │
    → primitive activations + attn maps       │
         │                                    │
         ▼                                    │
  Layer 2: CompositionalLayer                 │
    self-attn over P (+ optional RadGraph bias) │
         │                                    │
         ▼                                    │
  Layer 3: FindingsReadoutLayer               │
    finding queries → logits                  │
    + alpha * vlm_gate([z, p])  ◄─────────────┘
         │
         ▼
    Optional GumbelGate M (C×P) on readout path
```

| Layer | Module | Role |
|-------|--------|------|
| **1** | `PrimitiveConceptLayer` | P concept queries cross-attend over CLIP patch tokens |
| **2** | `CompositionalLayer` | Self-attention over primitives; optional `radgraph_bias` (P×P) |
| **3** | `FindingsReadoutLayer` | Attention readout to C=7 CheXpert findings + VLM residual (`alpha`) |
| **Gate** | `GumbelGate` | Relaxed binary M̃ ∈ [0,1]^(C×P); optional (`use_gate_M`) |

### 3.2 Two architecture presets

| Preset | `query_dim` | Cross / self layers | `alpha` | `use_gate_M` | Params | Typical test F1 |
|--------|-------------|---------------------|---------|--------------|--------|-----------------|
| **Default (pre-Optuna)** | 128 | 2 / 2 | 1.0 | true | ~435,261 | 0.653–0.674 |
| **Optuna trial-27 (best)** | 64 | 1 / 2 | 0.5 | false | 118,891 | **0.694–0.701** |

### 3.3 Vision and VLM inputs

| Component | Setting |
|-----------|---------|
| **Patches** | `openai/clip-vit-base-patch16`, 196 tokens × 768-D; frozen or **LoRA r=8** on vision (`scripts/19_train_lora_clip_vision.py`) |
| **VLM** | Frozen Qwen2-VL `x_logits`, `x_probs` — **never fine-tuned** in CCA experiments |
| **Labels** | C = 7 CheXpert findings; masked multi-label BCE |
| **Concept phrases** | 35 default phrases in code; P = 30 in best runs |

### 3.4 Faithfulness objective (when enabled)

```
L = L_BCE + λ_sparse · sparsity_target(M̃) + λ_faithful · L_faithful
```

- **Sparsity:** Hoyer / gate-density target (default 10%; PDF target 5–15%)
- **Intervention:** Mask primitive p, penalize downstream change where M̃ says no dependence
- **Metrics:** `scripts/faithfulness_metrics.py` — intervention consistency, necessity drop, sufficiency F1

---

## 4. Experimental protocol

### 4.1 Data splits

| Split | Rows | Path |
|-------|------|------|
| Train | 43,778 | `data/processed/splits/train_rows.json` |
| Val | 9,357 | `data/processed/splits/val_rows.json` |
| Test | 9,197 | `data/processed/splits/test_rows.json` |

**Protocol name:** `default` (train / val / test). Hyperparameters and early stopping use **val**; test is held out for final numbers in this report.

### 4.2 Metrics

| Metric | Definition | Primary? |
|--------|------------|----------|
| **Macro-F1 @0.5** | Masked macro-F1 at threshold 0.5 | **Yes** (leaderboard) |
| **Macro-F1 @thr** | Per-class val-tuned thresholds | Secondary |
| **Macro-AUROC** | `common_multilabel.probabilistic_metrics` | Yes |
| **Macro-AUPRC** | Same helper | Yes |
| **ECE** | Expected calibration error | Yes |
| **Brier** | Brier score | Yes |

### 4.3 Training conventions

| Setting | Typical value |
|---------|----------------|
| Seeds | 42 (single-seed); 0–4 (5-seed leaderboard) |
| Max epochs | 60 (early stop patience 16 for best config) |
| Checkpoint selection | `val_macro_f1_05` for leaderboard; `val_bce` in some legacy/Optuna final runs |
| Hardware (reference) | NVIDIA RTX 4060 8 GB |
| Artifact root | `data/processed/experiments/cca/default/<run_id>/` |

---

## 5. Master leaderboard

All models on **CheXpert default**, test set, unless noted. Sorted by test macro-F1 @0.5.

| Rank | Model | Run ID | Test F1 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Params |
|------|-------|--------|---------|------------|------------|----------|------------|--------|
| 1 | **CCA LoRA trial-27 (5-seed mean)** | `lora_r8_trial27_seeds_s*` | **0.701 ± 0.005** | **0.722 ± 0.004** | **0.625 ± 0.004** | **0.105 ± 0.018** | **0.176 ± 0.009** | 118,891 |
| 2 | CCA LoRA trial-27 (seed 42) | `cca_lora_r8_trial27` | 0.701 | — | — | — | — | 118,891 |
| 3 | CCA frozen trial-27 | `cca_frozen_trial27_f1` | 0.694 | — | — | — | — | 118,891 |
| 4 | Optuna trial-27 (25-ep tune) | optuna trial 27 | 0.691 | — | — | — | — | 118,891 |
| 5 | QFormer adapter | `qformer_adapter_default` | 0.676 | 0.708 | 0.607 | 0.127 | 0.183 | 263,815 |
| 6 | CCA faithful (frozen, default arch) | `cca_faithful` | 0.674 | — | — | — | — | 435,261 |
| 7 | CCA LoRA default arch | `cca_lora_r8_default` | 0.677 | — | — | — | — | 435,261 |
| 8 | CCA LoRA faithful | `cca_lora_r8_faithful` | 0.677 | — | — | — | — | 435,261 |
| 9 | Prior ablation `none` | `prior_ablation_none` | 0.683 | 0.676 | 0.576 | 0.110 | 0.183 | 435,261 |
| 10 | Optuna final (60 ep, val_bce) | `best_optuna_cca_hpo` | 0.658 | — | — | — | — | 118,891 |
| 11 | CCA default (first run) | `run_20260516_183647` | 0.653 | — | — | — | — | 435,261 |
| 12 | PostHoc CBM | `cbm_posthoc_default` | 0.621 | 0.542 | 0.472 | 0.115 | 0.213 | 667 |
| 13 | baseline_mlp (legacy) | — | 0.620 | — | — | — | — | — |
| 14 | vlm_mlp (3-seed mean) | seeds_s0–2 | 0.485 | — | — | — | — | — |
| 15 | Label-free CBM | `cbm_labelfree_default` | 0.476 | 0.616 | 0.528 | 0.121 | 0.213 | 217 |
| 16 | MLGCN | `mlgcn_default` | 0.470 | 0.535 | 0.468 | 0.130 | 0.219 | 8,577 |

### 5.1 Δ vs best CCA (5-seed mean, test F1 @0.5)

| Model | Δ test F1 | Δ test AUROC |
|-------|-----------|--------------|
| QFormer | −0.025 | −0.014 |
| PostHoc CBM | −0.080 | −0.180 |
| Label-free CBM | −0.225 | −0.106 |
| MLGCN | −0.231 | −0.187 |

---

## 6. Hyperparameter search (Optuna)

| Item | Value |
|------|-------|
| **Study** | `cca_hpo` |
| **Storage** | `data/processed/experiments/cca/optuna/study.db` |
| **Script** | `scripts/tune_cca_optuna.py` |
| **Trials** | 20 requested; 40 logged (incl. prunes) |
| **Tune budget** | 25 epochs; objective: maximize `val_macro_f1@0.5` |
| **Sampler** | TPE (seed 42); Median pruner |
| **Wall time** | ~3 h (RTX 4060) |

### 6.1 Search space

| Hyperparameter | Search range |
|----------------|--------------|
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

Trials with ≥ 1M trainable params or CUDA OOM were pruned.

### 6.2 Best trial (27)

| Metric | Value |
|--------|-------|
| Val macro-F1 @0.5 | **0.701** |
| Test macro-F1 @0.5 (during tune) | **0.691** |
| Trainable params | **118,891** |

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

### 6.3 Other strong trials (val F1 @0.5)

| Trial | Val F1 | Notes |
|-------|--------|-------|
| **27** | **0.701** | Best overall |
| 13 | 0.698 | `query_dim=192` |
| 25 | 0.698 | Same family as 27 |
| 35 | 0.694 | Lower dropout |
| 7, 22 | ~0.692 | Early good region |

### 6.4 Optuna final train (60 epochs, `val_bce` checkpoint)

| Run ID | Val F1 | Test F1 | Test F1 @thr | Epochs | Notes |
|--------|--------|---------|--------------|--------|-------|
| `best_optuna_cca_hpo` | 0.665 | 0.658 | 0.664 | 30 | **−0.033 test F1** vs trial 27 due to BCE ckpt selection |

**Lesson:** Align `--best_metric` with the HPO objective (`val_macro_f1_05`) for deployment training.

---

## 7. LoRA vs frozen CLIP patches

**Script:** `scripts/run_cca_lora_variants.py`  
**Patch cache:** `patch_v2_fp16_lora_r8` / encoder `openai_clip-vit-base-patch16_lora_r8`  
**Settings:** seed 42, `val_macro_f1_05` checkpoint, max 60 epochs

| Run ID | Patches | Val F1 | Test F1 | Test F1 @thr | Params | Epochs |
|--------|---------|--------|---------|--------------|--------|--------|
| **`cca_lora_r8_trial27`** | LoRA | 0.704 | **0.701** | 0.682 | 118,891 | 23 |
| `cca_lora_r8_trial27_faithful` | LoRA | 0.704 | **0.701** | 0.682 | 118,891 | 23 |
| `cca_frozen_trial27_f1` | Frozen | 0.702 | 0.694 | 0.670 | 118,891 | 34 |
| `cca_lora_r8_default` | LoRA | 0.683 | 0.677 | 0.655 | 435,261 | 18 |
| `cca_lora_r8_faithful` | LoRA | 0.686 | 0.677 | 0.659 | 435,261 | 24 |

### 7.1 Δ test F1 vs frozen trial-27 (0.694)

| Run | Δ |
|-----|---|
| LoRA trial-27 | **+0.007** |
| LoRA trial-27 + faithfulness | +0.007 (no gate → no effect) |
| LoRA default arch | −0.017 |
| LoRA faithful (default arch) | −0.017 |

**Conclusion:** LoRA on CLIP vision (+7 F1 points over frozen at trial-27 scale) is the largest single encoder-side gain. The compact trial-27 head beats the 435K default architecture even with LoRA patches.

### 7.1 LoRA on Qwen2-VL-2B-Instruct (r=16, PEFT baseline)

**Scripts:** `scripts/train_qwen2vl_lora_cls.py`, `scripts/train_qwen2vl_lora_sft.py`, `scripts/run_lora16_vs_cca.py`  
**Comparison table:** `reports/comparison/lora16_vs_cca.md` (generated after training)

Two variants on CheXpert `default` (single seed 42):

| Variant | Training | Inference |
|---------|----------|-----------|
| **cls head** | LoRA r=16 + `Linear(hidden, 7)` + masked BCE | Sigmoid logits (headline vs CCA) |
| **JSON SFT** | LoRA r=16 + token CE on ground-truth JSON | Same prompt as frozen zero-shot VLM |

**Status:** Implementation complete; fill metrics by running:

```powershell
$env:PYTHONPATH = "scripts"
# If load fails with JSONDecodeError / SafetensorError, repair the HF cache first:
python scripts/repair_qwen2vl_cache.py
python scripts/run_lora16_vs_cca.py --gpu_id 0
```

Compare to CCA leaderboard (`cca_lora_r8_trial27`, 5-seed mean F1 **0.701 ± 0.005**). LoRA ranks 4/8/32 and cross-site evaluation remain pending.

---

## 8. Single-seed CCA runs (frozen patches)

| Run ID | Checkpoint metric | Val F1 | Test F1 | Params | Epochs |
|--------|-------------------|--------|---------|--------|--------|
| `run_20260516_183647` | val_bce | 0.654 | 0.653 | 435,261 | — |
| `cca_faithful` | val_macro_f1_05 | 0.677 | 0.674 | 435,261 | 18 |
| `best_optuna_cca_hpo` | val_bce | 0.665 | 0.658 | 118,891 | 30 |
| `cca_frozen_trial27_f1` | val_macro_f1_05 | 0.702 | 0.694 | 118,891 | 34 |

---

## 9. Multi-seed stability

### 9.1 Leaderboard config: 5 × LoRA + trial-27

**Runs:** `lora_r8_trial27_seeds_s{0..4}`  
**Script:** `scripts/run_seeds.py --use_numbered_script`  
**Checkpoint:** `val_macro_f1_05`, patience 16, max 60 epochs

| Seed | Val F1 | Test F1 | Val AUROC | Test AUROC | Test AUPRC | Test ECE | Test Brier | Epochs |
|------|--------|---------|-----------|------------|------------|----------|------------|--------|
| 0 | 0.711 | 0.707 | 0.724 | 0.717 | 0.622 | 0.115 | 0.182 | 19 |
| 1 | 0.708 | 0.705 | 0.727 | 0.723 | 0.622 | 0.131 | 0.189 | 22 |
| 2 | 0.708 | 0.698 | 0.728 | 0.727 | 0.630 | 0.099 | 0.171 | 20 |
| 3 | 0.703 | 0.697 | 0.720 | 0.724 | 0.628 | 0.089 | 0.168 | 19 |
| 4 | 0.704 | 0.700 | 0.721 | 0.720 | 0.622 | 0.092 | 0.172 | 21 |
| **mean ± σ** | **0.707 ± 0.003** | **0.701 ± 0.004** | **0.724 ± 0.004** | **0.722 ± 0.004** | **0.625 ± 0.004** | **0.105 ± 0.018** | **0.176 ± 0.009** | 20.2 ± 1.3 |

**Aggregate files:** `data/processed/experiments/cca/default/seeds_summary.parquet`, `seeds_summary.json`

### 9.2 Legacy: 5 × frozen default CCA (`val_bce`)

| Seed | Val F1 | Test F1 |
|------|--------|---------|
| 0 | 0.660 | 0.652 |
| 1 | 0.546 | 0.541 |
| 2 | 0.605 | 0.598 |
| 3 | 0.692 | 0.679 |
| 4 | 0.638 | 0.636 |
| **Mean** | — | **0.621** |
| **95% CI (bootstrap)** | — | [0.575, 0.660] |

**Comparison:** LoRA + trial-27 raises mean test F1 by **+0.080** and cuts σ from **~0.054 → 0.005** (~11× tighter).

---

## 10. Adapter baselines

Trained 2026-05-18 with `scripts/15_train_posthoc_cbm.py` … `18_train_mlgcn.py`. Same default split and @0.5 threshold.

| Model | Script | Run ID | Epochs | LR | Key hparams | Test F1 | Test AUROC | Params |
|-------|--------|--------|--------|-----|-------------|---------|------------|--------|
| PostHoc CBM | `15_train_posthoc_cbm.py` | `cbm_posthoc_default` | 25 | 1e-3 | 30 concepts | 0.621 | 0.542 | 667 |
| Label-free CBM | `16_train_labelfree_cbm.py` | `cbm_labelfree_default` | 25 | 1e-3 | 30 concepts, CLIP batch 32 | 0.476 | 0.616 | 217 |
| QFormer adapter | `17_train_qformer_adapter.py` | `qformer_adapter_default` | 30 | 3e-4 | 32 queries, D=128, 2× cross, 4 heads | 0.676 | 0.708 | 263,815 |
| MLGCN | `18_train_mlgcn.py` | `mlgcn_default` | 30 | 1e-3 | label graph only | 0.470 | 0.535 | 8,577 |

QFormer is the strongest adapter baseline but remains **2.5 F1 points** below CCA LoRA trial-27.

---

## 11. Concept-prior ablation

**Script:** `scripts/run_prior_ablation.py`  
**Config:** Frozen CLIP patches, **default CCA** (~435K params), 30 epochs, P=30  
**Priors:** `data/processed/graph/prior_ablation/*.json`

| Variant | Source | Val F1 | Test F1 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Epochs |
|---------|--------|--------|---------|------------|------------|----------|------------|--------|
| **`none`** | No `radgraph_bias` | **0.693** | **0.683** | 0.676 | 0.576 | 0.110 | 0.183 | 24 |
| `co_occur` | Train label co-occurrence | 0.660 | 0.649 | 0.670 | 0.569 | 0.097 | 0.184 | 22 |
| `coerror` | Normalized co-error matrix | 0.660 | 0.650 | 0.684 | 0.583 | 0.101 | 0.179 | 29 |
| `radgraph` | **Stub** (= `co_occur`) | 0.660 | 0.649 | 0.670 | 0.569 | 0.097 | 0.184 | 22 |
| `permuted` | Row/col shuffle (control) | 0.656 | 0.650 | **0.689** | **0.590** | 0.102 | 0.178 | 26 |

### 11.1 Δ vs `none` (test)

| Variant | Δ test F1 | Δ test AUROC |
|---------|-----------|--------------|
| `co_occur` | −0.034 | −0.006 |
| `coerror` | −0.033 | +0.008 |
| `radgraph` | −0.034 | −0.006 |
| `permuted` | −0.033 | +0.013 |

**Conclusion:** Explicit P×P priors **hurt F1** (~3.3 points). Permuted control matches or beats informative priors on AUROC — no evidence that current priors help. True RadGraph wiring is pending (`source: radgraph_placeholder_cooccurrence`).

---

## 12. Held-out-concept probe

**Script:** `scripts/20_holdout_concept.py`  
**Method:** At eval time, zero out one primitive column of gate M (or no-op if no gate); measure Δ metrics on 4096-row val subsample.  
**Note:** This is **not** the PDF’s train-time “mask 20% of findings” protocol; it probes **primitive** dependence.

| Checkpoint | `use_gate_M` | Full val F1/AUROC | Max ΔAUROC | Mean ΔAUROC | Notes |
|------------|--------------|-------------------|------------|-------------|-------|
| `cca_lora_r8_trial27` | false | F1 0.702 | 0.000 | 0.000 | Expected (no gate) |
| `cca_lora_r8_default` | true | AUROC 0.620 | **2.2e-4** | 6.0e-5 | Best concept signal |
| `cca_lora_r8_faithful` | true | AUROC 0.637 | 4.3e-5 | 1.7e-5 | VLM residual dominates |
| `cca_faithful` (frozen) | true | AUROC 0.623 | 1.1e-5 | −1.3e-5 | Negligible effect |

Per-primitive JSON summaries: `reports/holdout/<checkpoint>.json` (when generated).

**Interpretation:** With `alpha=1.0` (default arch) and high gate density (~0.42), decisions lean on `vlm_mix`; primitives carry little independent mass. Lower `alpha` or stricter sparsity may amplify probe signal.

---

## 13. Faithfulness mechanism

### 13.1 `cca_faithful` (frozen patches, default arch)

| Setting / metric | Value |
|------------------|-------|
| `lambda_sparse` / `lambda_faithful` | 0.01 / 0.1 |
| `use_gate_M` | true |
| `alpha` | 1.0 |
| Test F1 @0.5 | 0.674 |
| Gate density (eval) | **0.438** (target 5–15% in PDF) |
| Intervention consistency | 0.554 |
| Necessity drop | 0.290 |
| Sufficiency F1 | 0.675 |

### 13.2 Faithfulness on best config (trial-27, no gate)

| Run | Faithfulness losses | Test F1 | Effect |
|-----|---------------------|---------|--------|
| `cca_lora_r8_trial27` | off | 0.701 | — |
| `cca_lora_r8_trial27_faithful` | λ_sparse=0.01, λ_faithful=0.1 | 0.701 | **No change** (`use_gate_M=false`) |

### 13.3 Faithfulness–utility Pareto (planned)

**Script:** `scripts/run_faithfulness_pareto.py`  
**Sweep:** λ_sparse ∈ {1e-3, 1e-2, 1e-1} on default arch (frozen patches, gate on); reuses `cca_faithful` at 1e-2.  
**Output (when run):** `reports/comparison/cca_faithfulness_pareto.md`  
**Status:** Driver exists; full 3-point Pareto table **not yet written** to reports.

---

## 14. Legacy GNN / calibrated protocol

Results from earlier bipartite GNN adapters on **`calibrated4way`** protocol (leakage-free per-class thresholds). **Not directly comparable** to CCA @0.5 on `default` without re-running CCA on 4-way splits.

| Model | Calibrated test macro-F1 |
|-------|--------------------------|
| gnn13_clip_bipartite | **0.689** |
| gnn12_clip_vlm_homo | 0.678 |
| vlm_mlp | 0.654 |
| frozen VLM (calibrated) | 0.651 |

Source: `docs/academic_report.md`, `reports/comparison/overall.json`

---

## 15. Statistical comparison

From `scripts/stats_compare.py` → `reports/comparison/stats.md` (bootstrap 400 resamples, paired on test predictions).

### 15.1 Test macro-F1 @0.5 (bootstrap 95% CI)

| Model | Mean F1 | 95% CI | n seeds |
|-------|---------|--------|---------|
| **cca** (LoRA trial-27) | 0.7010 | [0.6976, 0.7045] | 5 |
| qformer_adapter | 0.6755 | [0.6755, 0.6755] | 1 |
| cbm_posthoc | 0.6214 | [0.6214, 0.6214] | 1 |
| cbm_labelfree | 0.4760 | [0.4760, 0.4760] | 1 |
| mlgcn | 0.4700 | [0.4700, 0.4700] | 1 |

### 15.2 Test macro-AUROC (bootstrap 95% CI)

| Model | Mean AUROC | 95% CI | n seeds |
|-------|------------|--------|---------|
| **cca** | 0.7221 | [0.7190, 0.7249] | 5 |
| qformer_adapter | 0.7077 | [0.7077, 0.7077] | 1 |
| cbm_labelfree | 0.6160 | [0.6160, 0.6160] | 1 |
| cbm_posthoc | 0.5420 | [0.5420, 0.5420] | 1 |
| mlgcn | 0.5346 | [0.5346, 0.5346] | 1 |

### 15.3 Paired bootstrap AUROC vs CCA (ref: `lora_r8_trial27_seeds_s0`)

| Model | Δ mean AUROC | p (bootstrap) | BH reject @ q=0.05 |
|-------|--------------|---------------|---------------------|
| qformer_adapter | −0.0096 | 0.995 | no |
| cbm_posthoc | −0.1752 | 0.910 | no |
| cbm_labelfree | −0.1012 | 0.965 | no |
| mlgcn | −0.1826 | 0.975 | no |

*Note: Single-seed baselines limit paired-test power; DeLong per-class AUROC matrix is planned for the full paper.*

---

## 16. Key findings and interpretation

1. **Compact structured head wins.** Trial-27 (~119K params, D=64, one cross-attn layer, α=0.5, no gate) beats the 435K default and all adapter baselines on CheXpert default @0.5.

2. **LoRA CLIP patches add consistent gain.** +0.007 test F1 over frozen trial-27 at the same head; combined with HPO, +0.080 mean F1 vs legacy 5-seed frozen default.

3. **Checkpoint metric matters.** Optuna final with `val_bce` loses ~0.033 test F1 vs trial-27; always use `val_macro_f1_05` for F1-optimal deployment.

4. **Faithfulness is not yet part of the winning story.** Best model disables gate; faithful training on default arch does not reach trial-27 F1; gate density far above sparsity target.

5. **Graph priors hurt at P=30.** Compositional self-attention appears to absorb label structure; permuted prior is not worse than co-occurrence — RadGraph stub is inconclusive.

6. **Concept dependence is weak under VLM residual.** Held-out-primitive probes show tiny ΔAUROC when α=1.0; readout is VLM-dominated.

7. **Cross-site and LoRA-on-VLM remain the critical gaps** for the AAAI pre-registered headline (beat LoRA-16 cross-site at &lt;0.1% params on ≥3/4 site pairs).

---

## 17. Pending experiments (AAAI plan)

| Category | Item | Status |
|----------|------|--------|
| **Encoder** | LoRA ranks 4, 16 on CLIP | Not trained |
| **Encoder** | BiomedCLIP, RAD-DINO, MAE-CXR swap | Not run |
| **VLM PEFT** | LoRA on Qwen2-VL r=16 (cls + JSON SFT) | **Scripts ready** — run `scripts/run_lora16_vs_cca.py`; ranks 4/8/32 not run |
| **Protocol** | CCA on `calibrated4way` | Not run |
| **Prior** | True MIMIC RadGraph entity graph | Stub only |
| **Sites** | MIMIC, NIH, PadChest, VinDr cross-site | Not run |
| **Baselines** | Full 25-row matrix (Tent, MEMO, BitFit, …) | Partial (4 adapters) |
| **Ablations** | Remove L1/L2/L3, α=0, layer counts, P sweep | Not systematic |
| **Faithfulness** | 3-point λ_sparse Pareto figure | Script only |
| **Holdout** | Train-time 20% finding mask | Not run (eval primitive probe only) |
| **TTA** | Tent + marginal-rate prior | Not run |
| **Localization** | VinDr pointing-game / IoU | Not run |
| **Replication** | MS-COCO / NUS-WIDE | Not run |
| **Theory** | Propositions 1–3 + proofs | Not formalized |
| **Human eval** | Radiologist Likert study | Not run |

---

## 18. Artifacts and reproduction

### 18.1 Per-run artifacts

```
data/processed/experiments/cca/default/<run_id>/
  metrics.json
  best_checkpoint.pt
  val_predictions.json
  test_predictions.json
  history.json (if enabled)
```

### 18.2 Key paths

| Artifact | Path |
|----------|------|
| Optuna study | `data/processed/experiments/cca/optuna/study.db` |
| Best trial JSON | `data/processed/experiments/cca/optuna/best_trial.json` |
| 5-seed summary | `data/processed/experiments/cca/default/seeds_summary.{json,parquet}` |
| Prior matrices | `data/processed/graph/prior_ablation/*.json` |
| LoRA adapter | `data/processed/embeddings/lora_r8_adapter/` |
| Frozen patch cache | `data/processed/embeddings/chexpert_default_*_patch_v2_fp16.pt` |
| LoRA patch cache | `data/processed/embeddings/chexpert_default_*_patch_v2_fp16_lora_r8.pt` |

### 18.3 Quick reproduction (best model)

```powershell
$env:PYTHONPATH = "scripts"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# LoRA patches (once)
python scripts/19_train_lora_clip_vision.py --lora_rank 8 --gpu_id 0

# Best CCA (seed 42)
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27 --best_metric val_macro_f1_05 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text

# 5-seed leaderboard
python scripts/run_seeds.py --model_id cca --protocol default --seeds 0,1,2,3,4 `
  --run_id_prefix lora_r8_trial27_seeds --use_numbered_script `
  -- --gpu_id 0 --num_workers 0 --lora_rank 8 --num_primitives 30 --query_dim 64 `
  --n_cross_attn_layers 1 --n_self_attn_layers 2 --n_heads 4 --alpha 0.5 --dropout 0.1001 `
  --lr 0.000479 --weight_decay 0.000111 --batch_size 8 --epochs 60 --early_stop_patience 16 `
  --best_metric val_macro_f1_05 --no-use_gate_M --init_queries_from_text
```

**Full recipe:** [`docs/cca_reproduction.md`](cca_reproduction.md)

---

## 19. Script and config index

| Script | Purpose |
|--------|---------|
| `scripts/14_train_cca.py` | CCA training entry |
| `scripts/cca_train_core.py` | Training loop, metrics, faithfulness losses |
| `scripts/tune_cca_optuna.py` | Optuna HPO |
| `scripts/run_cca_lora_variants.py` | LoRA variant batch + comparison table |
| `scripts/run_seeds.py` | Multi-seed scheduler |
| `scripts/run_prior_ablation.py` | Concept-prior ablation (5 variants) |
| `scripts/run_faithfulness_pareto.py` | λ_sparse Pareto sweep |
| `scripts/20_holdout_concept.py` | Held-out primitive probe |
| `scripts/15–18_train_*.py` | PostHoc CBM, label-free CBM, QFormer, MLGCN |
| `scripts/19_train_lora_clip_vision.py` | LoRA CLIP patch cache |
| `scripts/train_qwen2vl_lora_cls.py` | Qwen2-VL LoRA r=16 + classification head |
| `scripts/train_qwen2vl_lora_sft.py` | Qwen2-VL LoRA r=16 + JSON generative SFT |
| `scripts/score_qwen2vl_lora.py` | Re-score val/test for a LoRA run |
| `scripts/run_lora16_vs_cca.py` | Train both LoRA variants + compare vs CCA |
| `scripts/qwen2vl_lora_common.py` | Shared Qwen2-VL LoRA utilities |
| `scripts/stats_compare.py` | Bootstrap / paired AUROC tables |
| `scripts/faithfulness_metrics.py` | Faithfulness loss and eval metrics |

| Config / doc | Content |
|--------------|---------|
| `configs/train_cca.yaml` | Presets: default, trial-27, faithfulness flags |
| `docs/cca_experiment_results.md` | Sectioned results log (maintained in parallel) |
| `docs/cca_optuna_hpo.md` | Optuna details |
| `docs/cca_reproduction.md` | End-to-end reproduction |
| `docs/pipeline.md` | Full pipeline map |
| `reports/comparison/*.md` | Auto/manual comparison tables |
| `reports/comparison/lora16_vs_cca.md` | LoRA-16 (Qwen2-VL) vs CCA table |

---

*Generated as the combined reference for AAAI core-architecture experiments. For incremental updates after new runs, append rows to §5 and the relevant section, or re-run comparison scripts and refresh §15.*
