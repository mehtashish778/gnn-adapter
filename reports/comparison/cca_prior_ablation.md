# Concept-prior ablation (CCA, default CCA arch, frozen patches, 30 ep)

Driver: `scripts/run_prior_ablation.py --gpu_id 0 --epochs 30 --run_id_prefix prior_ablation`  
Run dirs: `data/processed/experiments/cca/default/prior_ablation_<name>/`  
Prior matrices: `data/processed/graph/prior_ablation/*.json` (P=30)

| Variant | Source | Val F1 | Test F1 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Epochs |
|---------|--------|--------|---------|------------|------------|----------|------------|--------|
| `none` | no `radgraph_bias` | **0.693** | **0.683** | 0.676 | 0.576 | 0.110 | 0.183 | 24 |
| `co_occur` | label co-occurrence (P=30, train) | 0.660 | 0.649 | 0.670 | 0.569 | 0.097 | 0.184 | 22 |
| `coerror` | normalized co-error matrix | 0.660 | 0.650 | 0.684 | 0.583 | 0.101 | 0.179 | 29 |
| `radgraph` | **stub** = copy of `co_occur` | 0.660 | 0.649 | 0.670 | 0.569 | 0.097 | 0.184 | 22 |
| `permuted` | row/col-shuffle of `radgraph` (control) | 0.656 | 0.650 | **0.689** | **0.590** | 0.102 | **0.178** | 26 |

## Δ vs `none` (test F1 / test AUROC)

| Variant | Δ test F1 | Δ test AUROC |
|---------|-----------|--------------|
| `co_occur` | −0.034 | −0.006 |
| `coerror` | −0.033 | **+0.008** |
| `radgraph` | −0.034 | −0.006 |
| `permuted` | −0.033 | **+0.013** |

## Takeaways

1. **No prior beats every prior on test F1.** Adding any explicit P×P bias term costs ≈3.3 F1 points.
2. **Permuted (control) prior is best on AUROC/AUPRC** — informative priors carry no signal above noise at this scale.
3. **`co_occur ≡ radgraph`** because RadGraph is currently a placeholder copy. A real RadGraph entity graph may change this conclusion; gated on RadGraph parsing infrastructure.
4. Adding any prior slightly improves calibration (ECE 0.097 vs 0.110) at the cost of F1.

See also: [`docs/cca_experiment_results.md §7`](../../docs/cca_experiment_results.md#7-concept-prior-ablation-frozen-patches-default-cca-30-ep), [`docs/cca_reproduction.md`](../../docs/cca_reproduction.md).
