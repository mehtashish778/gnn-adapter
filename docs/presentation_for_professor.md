---
marp: true
theme: default
paginate: true
size: 16:9
header: "Concept-Evidence Adapters for Frozen VLMs — Progress Update"
footer: "MBZAI · 2026-05-18"
---

# Concept-Evidence Adapters for Frozen Vision–Language Models on Multi-Label Chest X-Ray Classification

**Progress update for Prof. [Name]**

From legacy GNN adapters (gnn07 / gnn12 / gnn13) to the Concept-Evidence Adapter (CCA)

Date: 2026-05-18

---

## Slide 2 — Outline

1. Problem setup and what we inherited
2. Phase 1: Baselines (frozen VLM, MLP)
3. Phase 2: GNN adapters — gnn07, gnn12, gnn13
4. Why we reframed the project
5. Phase 3: CCA architecture
6. Phase 3 results (Optuna, LoRA, 5-seed, baselines)
7. Concept priors, held-out probe, faithfulness
8. Findings, gaps, AAAI roadmap

---

## Slide 3 — Problem setup

**Task:** Multi-label chest X-ray classification on CheXpert.

- 7 canonical findings (masked multi-label BCE; `-1` = uncertain → masked)
- Splits: 43,778 train / 9,357 val / 9,197 test
- Two evaluation protocols:
  - **`default`** — train / val / test, threshold 0.5
  - **`calibrated4way`** — train_fit / **calib** / val / test, per-class thresholds tuned on `calib` only (leakage-free)

**Frozen scorer:** Qwen2-VL zero-shot, providing `(x_logits, x_probs)` per row — never fine-tuned.

**Goal:** A small, structured adapter that beats the VLM and parameter-efficient fine-tuning, with clinical interpretability.

---

## Slide 4 — What we inherited (Phase 1 baselines)

Macro-F1 captions: **`default` @0.5 (val / test)** · **`calibrated4way` (val / test)**

| Model | Type | Default @0.5 | Calibrated4way |
|-------|------|--------------|----------------|
| `vlm_zeroshot` | Frozen VLM, no adapter | — | 0.651 |
| `vlm_mlp` | MLP on `[z; p]` | 0.521 / **0.519** | 0.655 / **0.654** |

Key takeaway: calibration is doing most of the work on the legacy protocol. Without calibration, the bare VLM threshold at 0.5 is uncompetitive.

---

## Slide 5 — Phase 2: Legacy GNN adapters

Three GNN adapters were built on top of the frozen VLM:

| Model ID | Architecture |
|----------|--------------|
| `gnn07_label_residual` | Residual message passing on **co-error** adjacency A |
| `gnn12_clip_vlm_homo` | CLIP image embedding + homogeneous label GNN |
| `gnn13_clip_bipartite` | CLIP object node + 7 VLM attribute nodes, bipartite |

All three share:
- Frozen VLM `(x_logits, x_probs)` features
- Trained with masked BCE
- Run under **both** `default` and `calibrated4way` protocols

---

## Slide 6 — GNN results (macro-F1)

| Model | Default @0.5 (val / test) | Calibrated4way (val / test) |
|-------|---------------------------|------------------------------|
| MLP | 0.521 / 0.519 | 0.655 / 0.654 |
| GNN07 (residual) | 0.044 / 0.042 | 0.651 / 0.651 |
| GNN12 (CLIP+VLM homo) | 0.610 / 0.601 | 0.679 / 0.678 |
| **GNN13 (bipartite)** | **0.654 / 0.637** | **0.692 / 0.689** |

**Best legacy model:** **`gnn13_clip_bipartite` — 0.689 calibrated test macro-F1**.

GNN07’s @0.5 collapse is the leakage-free calibration story: without per-class thresholds, raw logits from this head are nearly uninformative.

---

## Slide 7 — Why we had to reframe

Reviewer-style critique of the GNN story (also the core thesis of the AAAI plan PDF):

1. **A 7-node “GNN” is mathematically close to an MLP** over `[e_CLIP; z; p]`.
2. **Margin too small** vs MLP after calibration (~+3.5 F1) to defend at a top-tier venue.
3. **No structural inductive bias** that an MLP truly cannot capture.
4. **No cross-site evidence** — does the structure even help portability?
5. Frame “GNN adapter beats MLP” invites a **taxonomy attack** in round one.

**Decision:** drop “GNN” from the title; reframe around **Concept-Evidence Adapters (CCA)** for frozen VLMs.

---

## Slide 8 — New thesis (AAAI plan)

> *Concept-Evidence Adapters for Frozen Vision-Language Models on Multi-Label Medical Classification.*

Five contributions targeted in the plan:

1. **Hierarchical compositional concept architecture** (primitives → composition → findings)
2. **Faithfulness mechanism** (sparse gate + intervention loss)
3. **Test-time site adaptation** (entropy min + marginal-rate prior)
4. **Cross-site evaluation** over 4 chest-X-ray datasets + 1 non-medical
5. **Leakage-free calibration** generalized as a methodological contribution

This slide deck focuses on **contributions 1, 2, and 5** — what is built and measured today.

---

## Slide 9 — CCA architecture (high level)

```
Frozen ViT patches (B, 196, 768)      Frozen VLM (z, p)
         │                                    │
         ▼                                    │
  Layer 1: PrimitiveConceptLayer              │
   P concept queries × cross-attn             │
   → primitive activations + spatial maps     │
         │                                    │
         ▼                                    │
  Layer 2: CompositionalLayer                 │
   self-attn over P (+ optional RadGraph bias)│
         │                                    │
         ▼                                    │
  Layer 3: FindingsReadoutLayer               │
   finding queries → logits                   │
   + alpha · vlm_gate([z, p])  ◄──────────────┘
         │
         ▼ optional Gumbel gate M (C × P)
```

**Encoder:** CLIP ViT-B/16 (frozen or LoRA r=8) → 196 × 768 patch tokens (no pooling).

---

## Slide 10 — CCA components in code

Module: `scripts/models/architectures/cca.py`

| Layer | Module | Function |
|-------|--------|----------|
| 1 | `PrimitiveConceptLayer` | P learnable concept queries cross-attend over patch tokens |
| 2 | `CompositionalLayer` | Self-attention over P primitives; optional P×P RadGraph bias |
| 3 | `FindingsReadoutLayer` | Attention readout to C=7 findings + `alpha · g(z, p)` residual |
| gate | `GumbelGate` | Relaxed binary M̃ ∈ [0,1]^(C×P) for sparse faithful concept selection |

**Faithfulness losses** (`scripts/faithfulness_metrics.py`):
- `sparsity_target_loss(M)` toward 10% density
- `intervention_faithfulness_loss` — penalize downstream change when gate says “no dependence”

---

## Slide 11 — Two architectural presets

| Preset | `query_dim` | Cross / self layers | `alpha` | gate | Params | Test F1 |
|--------|-------------|---------------------|---------|------|--------|---------|
| Default (pre-Optuna) | 128 | 2 / 2 | 1.0 | on | **435,261** | 0.653–0.674 |
| **Optuna trial-27 (best)** | 64 | 1 / 2 | 0.5 | off | **118,891** | **0.694–0.701** |

**Headline:** a smaller, gate-free head **wins** at this scale. Trial-27 also keeps us **under the 1M-param target** from the AAAI plan.

---

## Slide 12 — Optuna hyperparameter search

| Item | Setting |
|------|---------|
| Script | `scripts/tune_cca_optuna.py` |
| Trials | 20 requested, 40 logged (incl. prunes) |
| Tune budget | 25 epochs, max F1 |
| Sampler / pruner | TPE seed 42 / Median |
| Wall time | ~3h on RTX 4060 |

Search dimensions: `num_primitives ∈ {15, 30, 50}`, `query_dim ∈ {64, 128, 192}`, cross-layers 1–2, self-layers 1–2, heads {2, 4}, `alpha ∈ {0.5, 1.0}`, dropout 0.05–0.25, lr 1e-4–5e-4, batch {8, 16, 32}, gate {on, off}, text-init {on, off}.

Trials ≥ 1M params or OOM were pruned.

---

## Slide 13 — Optuna best trial (#27)

| Metric | Value |
|--------|-------|
| Val macro-F1 @0.5 (tune) | **0.701** |
| Test macro-F1 @0.5 (tune) | **0.691** |
| Trainable params | **118,891** |

| Hyperparameter | Value |
|----------------|-------|
| `num_primitives` | 30 |
| `query_dim` | 64 |
| `n_cross_attn_layers` / `n_self_attn_layers` | 1 / 2 |
| `n_heads` / `alpha` / dropout | 4 / 0.5 / 0.10 |
| lr / weight_decay / batch_size | 4.8e-4 / 1.1e-4 / 8 |
| `use_gate_M` / `init_queries_from_text` | **false** / true |

Lesson: **align checkpoint metric with HPO objective.** 60-ep final retraining with `val_bce` lost ~0.033 test F1 vs the F1-selected ckpt.

---

## Slide 14 — LoRA vs frozen CLIP patches

LoRA r=8 on CLIP vision-only (`scripts/19_train_lora_clip_vision.py`), then CCA on the new patch cache.

| Run | Patches | Test F1 | Params |
|-----|---------|---------|--------|
| **`cca_lora_r8_trial27`** | LoRA r=8 | **0.701** | 119K |
| `cca_lora_r8_trial27_faithful` | LoRA r=8 | 0.701 | 119K |
| `cca_frozen_trial27_f1` | Frozen | 0.694 | 119K |
| `cca_lora_r8_default` | LoRA r=8 | 0.677 | 435K |
| `cca_lora_r8_faithful` | LoRA r=8 | 0.677 | 435K |

**+0.007 F1** from LoRA over frozen at trial-27 head; default 435K head underperforms even on LoRA patches.

---

## Slide 15 — 5-seed stability (leaderboard config)

Config: LoRA r=8 + trial-27, `val_macro_f1_05` checkpoint, max 60 epochs, patience 16.

| Seed | Val F1 | Test F1 | Val AUROC | Test AUROC | Test AUPRC | Test ECE | Test Brier |
|------|--------|---------|-----------|------------|------------|----------|------------|
| 0 | 0.711 | 0.707 | 0.724 | 0.717 | 0.622 | 0.115 | 0.182 |
| 1 | 0.708 | 0.705 | 0.727 | 0.723 | 0.622 | 0.131 | 0.189 |
| 2 | 0.708 | 0.698 | 0.728 | 0.727 | 0.630 | 0.099 | 0.171 |
| 3 | 0.703 | 0.697 | 0.720 | 0.724 | 0.628 | 0.089 | 0.168 |
| 4 | 0.704 | 0.700 | 0.721 | 0.720 | 0.622 | 0.092 | 0.172 |
| **mean ± σ** | **0.707 ± 0.003** | **0.701 ± 0.004** | **0.724 ± 0.004** | **0.722 ± 0.004** | **0.625 ± 0.004** | **0.105 ± 0.018** | **0.176 ± 0.009** |

vs legacy 5-seed (default frozen CCA): mean 0.621, **σ ≈ 0.054 → 0.005** (≈11× tighter), **+0.080 F1**.

---

## Slide 16 — CCA vs adapter baselines on CheXpert default

All trained on the same default split @0.5 (2026-05-18).

| Model | Test F1 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Params |
|-------|---------|------------|------------|----------|------------|--------|
| **CCA LoRA trial-27 (5-seed)** | **0.701 ± 0.005** | **0.722 ± 0.004** | **0.625 ± 0.004** | 0.105 | 0.176 | 118,891 |
| QFormer adapter | 0.676 | 0.708 | 0.607 | 0.127 | 0.183 | 263,815 |
| PostHoc CBM | 0.621 | 0.542 | 0.472 | 0.115 | 0.213 | 667 |
| Label-free CBM | 0.476 | 0.616 | 0.528 | 0.121 | 0.213 | 217 |
| MLGCN | 0.470 | 0.535 | 0.468 | 0.130 | 0.219 | 8,577 |

Δ vs CCA: QFormer −0.025 F1 / −0.014 AUROC; CBMs / MLGCN −0.08 to −0.23 F1.

---

## Slide 17 — Statistical comparison (bootstrap)

`scripts/stats_compare.py` → `reports/comparison/stats.md` (paired 400-resample bootstrap on test predictions).

**Test macro-F1 @0.5 (95% CI):**

| Model | Mean F1 | 95% CI | n seeds |
|-------|---------|--------|---------|
| **CCA** | **0.7010** | [0.6976, 0.7045] | 5 |
| QFormer | 0.6755 | single point | 1 |
| PostHoc CBM | 0.6214 | single point | 1 |
| Label-free CBM | 0.4760 | single point | 1 |
| MLGCN | 0.4700 | single point | 1 |

**Test macro-AUROC (95% CI for CCA):** [0.7190, 0.7249]. CCA CI excludes every baseline’s mean F1 and AUROC.

DeLong matrix and BH-corrected p-values are scheduled once baselines have ≥ 3 seeds each.

---

## Slide 18 — Concept-prior ablation

`scripts/run_prior_ablation.py` — frozen patches, default CCA (~435K), P=30, 30 epochs.

| Prior | Test F1 | Test AUROC | Test ECE |
|-------|---------|------------|----------|
| **`none`** | **0.683** | 0.676 | 0.110 |
| `co_occur` (train label co-occurrence) | 0.649 | 0.670 | 0.097 |
| `coerror` (normalized co-error) | 0.650 | 0.684 | 0.101 |
| `radgraph` (stub = co_occur) | 0.649 | 0.670 | 0.097 |
| **`permuted` (control, shuffled)** | 0.650 | **0.689** | 0.102 |

**Observation:**
- Adding **any P×P prior costs ~3.3 F1**.
- **Permuted control ≥ informative priors** on AUROC/AUPRC.
- True MIMIC RadGraph prior is still a placeholder; full ablation is gated on parsing infra.

---

## Slide 19 — Held-out-concept probe

`scripts/20_holdout_concept.py` — at eval, zero out one primitive column of gate M and measure Δ metrics.

| Checkpoint | `use_gate_M` | Max ΔAUROC | Mean ΔAUROC |
|------------|--------------|------------|-------------|
| `cca_lora_r8_trial27` | false | 0.000 | 0.000 |
| `cca_lora_r8_faithful` | true | 4.3e-5 | 1.7e-5 |
| `cca_lora_r8_default` | true | **2.2e-4** | 6.0e-5 |
| `cca_faithful` (frozen) | true | 1.1e-5 | −1.3e-5 |

**Interpretation:** with `alpha = 1.0` (default arch) and high gate density (~0.42), the readout leans on the VLM residual; primitive columns carry little independent decision mass. Need to **lower α** or **stiffen sparsity** to amplify the probe signal.

---

## Slide 20 — Faithfulness mechanism — current state

`cca_faithful` (frozen patches, default arch):

| Setting / metric | Value |
|------------------|-------|
| `lambda_sparse` / `lambda_faithful` | 0.01 / 0.1 |
| Gate density (eval) | **0.438** (target 5–15%) |
| Intervention consistency | 0.554 |
| Necessity drop | 0.290 |
| Sufficiency F1 | 0.675 |
| Test F1 | 0.674 |

On the **winning trial-27** (gate off), faithfulness losses are **no-ops** → F1 unchanged.

**Pending:** `scripts/run_faithfulness_pareto.py` sweeps λ_sparse ∈ {1e-3, 1e-2, 1e-1} on the gated default arch to draw the Pareto curve.

---

## Slide 21 — CCA vs everything: one chart in numbers

(Replace with a Pareto plot for the slide; numeric snapshot below.)

| Family | Best model | Test F1 | Params |
|--------|------------|---------|--------|
| Frozen VLM | calibrated zero-shot | 0.651 | 0 |
| MLP | `vlm_mlp` (cal4way) | 0.654 | small |
| GNN (residual) | `gnn07` (cal4way) | 0.651 | tiny |
| GNN (CLIP+VLM) | `gnn12` (cal4way) | 0.678 | small |
| GNN (bipartite) | `gnn13` (cal4way) | 0.689 | small |
| CBM family | PostHoc CBM | 0.621 | 667 |
| QFormer-style | `qformer_adapter` | 0.676 | 264K |
| **CCA (ours)** | **`cca_lora_r8_trial27`** | **0.701 ± 0.005** | **119K** |

---

## Slide 22 — Key observations

1. **Compact structured head wins.** D=64, 1 cross-attn layer, α=0.5, no gate beats the 435K default and every adapter baseline.
2. **LoRA on CLIP patches is the largest single encoder-side gain** (+0.007 over frozen at trial-27).
3. **Variance collapses with the right config** (σ 0.054 → 0.005 across seeds).
4. **Concept priors don’t help yet** at P=30; permuted == informative.
5. **Faithfulness is not yet part of the winning model** — best config disables the gate.
6. **Calibration matters** — picking `val_macro_f1_05` over `val_bce` costs +0.033 F1.

---

## Slide 23 — What is still missing (AAAI plan)

| Block | Missing |
|-------|---------|
| **Cross-site** | MIMIC-CXR, NIH, PadChest, VinDr |
| **VLM PEFT** | LoRA on Qwen2-VL at ranks 4 / 8 / 16 / 32 |
| **Encoder swap** | BiomedCLIP, RAD-DINO, MAE-CXR |
| **Faithfulness** | 3-point λ_sparse Pareto figure |
| **Prior** | Real MIMIC RadGraph entity graph |
| **Holdout** | Train-time 20% finding mask (zero-shot finding test) |
| **TTA** | Tent + marginal-rate prior |
| **Localization** | VinDr pointing-game / IoU |
| **Non-medical** | MS-COCO multi-label replication |
| **Theory** | Propositions 1–3 (Pareto, sufficiency, calibration consistency) |
| **Human eval** | Radiologist Likert study |

---

## Slide 24 — Next steps (3 work tracks)

**Track A — Lock in the headline (1–2 weeks)**
- CCA on `calibrated4way` to compare against GNN13’s 0.689 directly
- LoRA r ∈ {4, 16} scan on CLIP patches
- 3-point faithfulness Pareto + figure

**Track B — Cross-site portability (the AAAI-grade claim)**
- MIMIC-CXR-JPG cross-site eval, then NIH and PadChest
- LoRA on Qwen2-VL at r ∈ {8, 16} as the load-bearing baseline
- Pre-register the success criterion before touching test

**Track C — Interpretability & theory**
- Wire true MIMIC RadGraph prior, rerun prior ablation
- Draft Propositions 1 (Pareto), 3 (calibration consistency)
- Pointing-game evaluation on VinDr bounding boxes

---

## Slide 25 — Risks and contingencies

(from the AAAI plan’s contingency section)

- **If CCA does not beat LoRA-16 cross-site** → reframe around faithfulness + cost story; same architecture, different framing.
- **If sparsity/intervention loss is unstable** → fall back to architecture without intervention loss, score faithfulness post-hoc.
- **If radiologist eval is infeasible** → substitute VinDr bounding-box localization metrics; defer radiologist study to journal extension.
- **If RadGraph parsing is delayed** → keep the prior ablation honest with the permuted control as the negative baseline.

All three pivots preserve a publishable narrative.

---

## Slide 26 — Reproduction recipe (one command)

```powershell
$env:PYTHONPATH = "scripts"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# 1. LoRA CLIP patch cache (once)
python scripts/19_train_lora_clip_vision.py --lora_rank 8 --gpu_id 0

# 2. Best CCA (seed 42)
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27 --best_metric val_macro_f1_05 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text

# 3. 5-seed sweep
python scripts/run_seeds.py --model_id cca --protocol default --seeds 0,1,2,3,4 `
  --run_id_prefix lora_r8_trial27_seeds --use_numbered_script -- ...same flags...
```

Full guide: [`docs/cca_reproduction.md`](cca_reproduction.md).

---

## Slide 27 — Summary

- **Where we started:** frozen VLM ≈ MLP ≈ small GNN at ~0.65 macro-F1; bipartite GNN13 best legacy at **0.689 (calibrated)**.
- **Where we are:** **CCA LoRA trial-27**, **119K params**, **0.701 ± 0.005 F1**, **0.722 ± 0.004 AUROC** on CheXpert default, **+0.025 F1** over the strongest adapter baseline.
- **What is solid:** architecture, HPO, multi-seed, four adapter baselines, prior ablation, held-out probe, faithfulness stack.
- **What is next:** cross-site, LoRA-on-VLM, faithfulness Pareto, calibrated 4-way for CCA, true RadGraph.

**Documents to read in order:**
1. `docs/combined_experiments_report.md` (this update, all results)
2. `docs/cca_experiment_results.md` (sectioned log)
3. `docs/cca_reproduction.md` (one-command reproduction)
4. `upcomming_plan/AAAI___Ashish___Concept (1).pdf` (the plan)

---

## Slide 28 — Questions / discussion

Possible discussion topics:

- Acceptable cross-site margin to pre-register against LoRA-16
- Whether to chase **faithfulness story** (Pareto + radiologist study) or **portability story** (cross-site + TTA) first
- Whether to lock CCA on **calibrated4way** before any cross-site spend
- Priority of **RadGraph** parsing vs. **encoder swap** (BiomedCLIP / RAD-DINO) for next sprint

Thank you.
