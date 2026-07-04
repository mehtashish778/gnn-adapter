# AAAI Submission — TODO List

## Architecture

- [ ] Build **Option A (CCA)** as the primary architecture
  - [ ] Replace pooled CLIP embedding with full sequence of frozen image tokens (~196 spatial tokens for ViT-B/16)
  - [ ] Implement learnable concept query embeddings initialized from text-encoder embeddings
  - [ ] Implement cross-attention block (1–2 heads, 2 layers) for concept queries over image tokens
  - [ ] Add learned gating function to combine attended evidence with VLM logits
  - [ ] Verify parameter count stays under 1 million
- [ ] Layer **Option B (RadGraph prior)** as a separately ablatable component
  - [ ] Replace empirical co-error adjacency with RadGraph knowledge graph from MIMIC-CXR
  - [ ] Include 14 CheXpert findings, latent parent concepts, and modifier nodes
- [ ] Implement **Option C (Multi-VLM ensemble adapter)** as the applied-systems experiment
- [ ] Drop "GNN" from the working title → use *"Concept-Evidence Adapters for Frozen Vision-Language Models on Multi-Label Medical Classification"*

---

## CCA — Three Layers

- [ ] **Layer 1 — Primitive concepts**
  - [ ] Instantiate ~30–50 radiological primitive query embeddings
  - [ ] Initialize from text-encoder embeddings and RadGraph node embeddings where available
  - [ ] Implement single multi-head cross-attention layer over frozen image tokens
  - [ ] Output: scalar activation + spatial attention map per primitive
- [ ] **Layer 2 — Compositional reasoning**
  - [ ] Implement two-layer self-attention encoder on P primitive activations
  - [ ] Add learnable compositional bias initialized from RadGraph relation matrix
- [ ] **Layer 3 — Findings readout**
  - [ ] Read out C target findings from Layer 2 via attention
  - [ ] Add frozen VLM logits as residual/gating signal
  - [ ] Connect to leakage-free four-way calibration protocol

---

## Faithfulness Mechanism

- [ ] Implement **concept-disentangled sparse gating**
  - [ ] Define learnable gating matrix M ∈ {0,1}^(C×P)
  - [ ] Relax to M̃ ∈ [0,1]^(C×P) via Gumbel-softmax
  - [ ] Apply L0/Hoyer sparsity prior (target density 5–15%)
- [ ] Implement **causal-intervention training**
  - [ ] Sample primitive p and counterfactual activation c′ at each training step
  - [ ] Compute Lfaithful penalizing downstream change where gating says there should be none
- [ ] Set up total training objective: L = LBCE + λ_sparse‖M̃‖₀ + λ_faithful · Lfaithful
- [ ] Tune λ_sparse, λ_faithful on val only

---

## Test-Time Site Adaptation (TTA)

- [ ] Freeze primitive layer and gating matrix M at deployment
- [ ] Implement entropy-minimization on unlabeled new-site images (reuse Tent official implementation)
- [ ] Add marginal-rate prior for per-finding base rate anchoring (reuse BACS-style implementation)

---

## Datasets

- [ ] CheXpert (in-distribution train + test)
- [ ] MIMIC-CXR-JPG (PhysioNet credentialed access — cross-site)
- [ ] NIH ChestX-ray14 (cross-site)
- [ ] PadChest (cross-site, different language/population)
- [ ] VinDr-CXR (cross-site + ground-truth bounding boxes for localization)
- [ ] MS-COCO multi-label or NUS-WIDE (non-medical replication benchmark)

---

## Baselines (25 rows required)

### Frozen-output adapters
- [ ] Zero-shot VLM
- [ ] Linear probe on [z; p]
- [ ] MLP on [z; p]
- [ ] MLP on [z; p; e_CLIP]
- [ ] ML-GCN (faithful re-implementation on label graph)
- [ ] Q-Former-style learnable-query adapter
- [ ] Post-hoc CBM
- [ ] Label-free CBM
- [ ] Current bipartite head (prior version of this work)

### PEFT on the VLM
- [ ] LoRA on Qwen2-VL at ranks {4, 8, 16, 32}
- [ ] LoRA on vision tower only
- [ ] Prefix tuning
- [ ] Visual prompt tuning
- [ ] BitFit

### Medical-domain baselines
- [ ] BiomedCLIP linear probe
- [ ] RAD-DINO linear probe
- [ ] KAD re-run on our splits (official code)
- [ ] CXR-LLaVA (if checkpoint public)
- [ ] MAIRA-2 (if checkpoint public)

### Loss and calibration baselines
- [ ] Asymmetric loss (same backbone)
- [ ] Logit adjustment
- [ ] Temperature scaling
- [ ] Per-class isotonic regression (sklearn.isotonic)
- [ ] Deep ensembles

### TTA baselines
- [ ] Tent
- [ ] MEMO
- [ ] Source-free DA

---

## Metrics to Report

- [ ] Per-class and macro AUROC and AUPRC (`torchmetrics.classification.MultilabelAUROC`, cross-checked vs `sklearn.metrics.roc_auc_score`)
- [ ] F1, precision, recall at calibrated threshold per class
- [ ] ECE and Brier score per class + reliability diagrams
- [ ] Intervention-consistency score + necessity/sufficiency score (faithfulness)
- [ ] Pointing-game accuracy and IoU vs ground-truth bounding boxes (VinDr-CXR)
- [ ] ΔAUROC pre- vs post-TTA per cross-site pair
- [ ] Inference cost: trainable params, GPU-hours, latency (p50/p95), peak GPU memory

### Statistical protocol
- [ ] All numbers as mean ± std across **5 seeds**
- [ ] Paired bootstrap 95% CI (1,000 resamples)
- [ ] DeLong's test for AUROC comparisons
- [ ] Benjamini–Hochberg multiple-testing correction at q=0.05
- [ ] Pre-register success criterion before running cross-site experiments

---

## Ablations

- [ ] Image encoder swap: CLIP-B/32 → BiomedCLIP → RAD-DINO → MAE-CXR
- [ ] Concept-query initialization: random vs label-text encoding
- [ ] Graph prior: none / co-occurrence / co-error / RadGraph
- [ ] Number of cross-attention layers
- [ ] Calibration method: none / threshold / temperature / isotonic
- [ ] Primitive count P ∈ {15, 30, 50, 80}
- [ ] Sparsity target variation
- [ ] Remove primitive layer
- [ ] Remove compositional layer
- [ ] Remove residual VLM gating
- [ ] Remove sparsity prior
- [ ] Remove intervention loss
- [ ] Remove TTA
- [ ] **Permuted-graph control** (randomly shuffle RadGraph prior — main paper)
- [ ] **Held-out-concept test** (mask 20% of findings at train time — main paper)

---

## Theory

- [ ] **Proposition 1** — Faithfulness–utility tradeoff: Pareto frontier under sparse-gating L0 budget k
- [ ] **Proposition 2** — Sufficient-statistic characterization: when do frozen VLM scores (z, p) suffice vs when do raw image tokens carry residual signal
- [ ] **Proposition 3** — Calibration consistency: formalize the four-way protocol's O(1/√n_calib) decay vs val-tuned Θ(1) bias
- [ ] Full proofs in appendix; informal statements in main text (~half page each)

---

## Interpretability Deliverables

- [ ] **Radiologist evaluation** — 2–3 radiologists score 100 CheXpert test cases on 5-point Likert scale; compare CCA vs Grad-CAM-on-ResNet-50 vs Q-Former; report Krippendorff's α
- [ ] **Intervention case studies** — 10 clinically interesting cases; force one primitive off and show downstream changes as a causal trace
- [ ] **Failure-mode taxonomy** — 50–100 failure cases manually categorized (occluded findings, atypical presentations, label noise) with attention maps

---

## Tables (Main Paper)

- [ ] **Table 1** — Headline cross-site results (CCA + 5 strongest baselines; macro-AUROC in/cross-site, params, latency)
- [ ] **Table 2** — In-distribution CheXpert full 25-row baseline matrix (AUROC, AUPRC, F1, ECE, Brier, params, GPU-hours)
- [ ] **Table 3** — TTA gains (ΔAUROC per cross-site direction + significance markers)
- [ ] **Table 4** — Faithfulness metrics (CCA at 3 sparsity levels vs CBMs vs Q-Former)
- [ ] **Table 5** — Radiologist evaluation (Likert, Krippendorff's α, % rated ≥4)
- [ ] **Table 6** — Non-medical replication on MS-COCO (macro-mAP, macro-F1, params)
- [ ] **Table 7** — Cost summary (params, GPU-hours, latency p50/p95, peak memory, FLOPs)

## Tables (Appendix)

- [ ] Table A.1 — Per-class AUROC/AUPRC on CheXpert with 95% CIs
- [ ] Table A.2 — Per-class results on MIMIC-CXR, NIH, PadChest, VinDr-CXR
- [ ] Table A.3 — Full ablation matrix
- [ ] Table A.4 — Permuted-graph control
- [ ] Table A.5 — Held-out-concept generalization
- [ ] Table A.6 — Per-class ECE and Brier on every dataset
- [ ] Table A.7 — DeLong pairwise p-value matrix (CCA vs every baseline, per dataset)
- [ ] Table A.8 — Per-seed raw headline numbers
- [ ] Table A.9 — Hyperparameter table for every method
- [ ] Table A.10 — Compute cost breakdown by phase
- [ ] Table A.11 — Per-rater radiologist scores + pairwise agreement matrix
- [ ] Table A.12 — Failure-mode taxonomy counts
- [ ] Table A.13 — Calibration policy comparison (U-Zeros, U-Ones, U-Ignore)
- [ ] Table A.14 — Four-way protocol vs val-tuned protocol (Proposition 3 empirical)

---

## Figures (Main Paper)

- [ ] **Figure 1** — Pull figure: deployment scenario (left) + cross-site AUROC vs param count Pareto (right)
- [ ] **Figure 2** — Architecture diagram: 3 layers with frozen blocks grey, learnable coloured, VLM gating as separate path
- [ ] **Figure 3** — Qualitative concept-attention maps: 6 cases (1 normal, 5 pathologies) with attention overlays + GT bounding boxes
- [ ] **Figure 4** — Faithfulness–utility Pareto frontier (CCA at varying sparsity vs baseline points)
- [ ] **Figure 5** — Intervention case study with causal trace arrows
- [ ] **Figure 6** — Calibration reliability diagrams (one sub-panel per finding, CCA pre/post vs LoRA-16)
- [ ] **Figure 7** — TTA dynamics: AUROC vs optimization steps (CCA vs Tent-on-LoRA vs MEMO-on-LoRA)

## Figures (Appendix)

- [ ] Figure A.1 — Gating matrix M heatmap before/after training
- [ ] Figure A.2 — Sparsity convergence curves
- [ ] Figure A.3 — Per-class confusion matrices (CheXpert, MIMIC, NIH, PadChest)
- [ ] Figure A.4 — Extended qualitative gallery (40 cases)
- [ ] Figure A.5 — Failure-mode gallery (20 cases)
- [ ] Figure A.6 — t-SNE/UMAP of primitive embeddings coloured by clinical category
- [ ] Figure A.7 — Hyperparameter sensitivity heatmaps (λ_sparse, λ_faithful, LR)
- [ ] Figure A.8 — Training loss curves
- [ ] Figure A.9 — Radiologist Likert distributions per method and rater
- [ ] Figure A.10 — AUROC vs training-set size (data-efficiency curves)

---

## Analyses (Main Paper Prose)

- [ ] Headline cross-site analysis (gain over LoRA-16 per direction, which findings drive gain)
- [ ] Ablation analysis (most costly removal, any zero-cost removals)
- [ ] Faithfulness analysis (CCA vs CBM variants; cost in AUROC ≤0.5)
- [ ] Concept-emergence analysis (top-k primitives per finding from M; radiologist agreement rate)
- [ ] TTA analysis (decompose gain: zero-shot → calibration transfer → full TTA)
- [ ] Localization analysis (pointing-game + IoU vs Grad-CAM and Q-Former)
- [ ] Cost analysis (< 0.1% LoRA-16 params; ≤5% FLOPs)
- [ ] Failure analysis (majority = label noise / atypical presentation, not method failures)
- [ ] Statistical-rigor section (explicitly state full protocol to pre-empt reviewer concerns)

---

## Engineering

- [ ] Consolidate per-script duplication → single `train.py` (training loop, metrics, registry)
- [ ] Create `models/` package with one file per architecture
- [ ] Calibration as a dedicated post-hoc module
- [ ] Persist features as `.npz`/`.pt` (not JSON) with on-disk cache keyed by `(dataset, encoder, preprocessing-version)`
- [ ] Thin multi-seed scheduler: pins seeds, records git hashes, emits one row per run into results parquet
- [ ] Table-generation scripts consume results parquet
- [ ] Ensure `make reproduce` reproduces every result in the paper

---

## Contingency Planning

- [ ] If cross-site gain over LoRA-16 misses pre-registered margin → reframe around faithfulness story
- [ ] If faithfulness mechanism fails to converge → fall back to architecture without intervention loss; score faithfulness post-hoc
- [ ] If radiologist evaluation is infeasible → substitute VinDr-CXR bounding-box localization metrics; defer radiologist study to journal extension
