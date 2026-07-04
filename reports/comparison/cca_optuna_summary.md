# CCA results summary (2026-05-17)

**Full results:** [`docs/cca_experiment_results.md`](../../docs/cca_experiment_results.md)  
**Reproduction recipe + hyperparameter reference:** [`docs/cca_reproduction.md`](../../docs/cca_reproduction.md)  
**LoRA variants:** [`cca_lora_variants.md`](cca_lora_variants.md)  
**Optuna HPO details:** [`docs/cca_optuna_hpo.md`](../../docs/cca_optuna_hpo.md)

## Leaderboard (test macro-F1 @0.5, default split)

| Rank | Run | Test F1 | Patches | Params |
|------|-----|---------|---------|--------|
| 1 | `cca_lora_r8_trial27` | **0.701** | LoRA r8 | 119K |
| 2 | `cca_frozen_trial27_f1` | 0.694 | Frozen | 119K |
| 3 | Optuna trial 27 (tune) | 0.691 | Frozen | 119K |
| 4 | `cca_faithful` | 0.674 | Frozen | 435K |
| 5 | `cca_lora_r8_default` | 0.677 | LoRA | 435K |
| 6 | `best_optuna_cca_hpo` | 0.658 | Frozen | 119K |
| 7 | `run_20260516_183647` | 0.653 | Frozen | 435K |
| 8 | CCA 5-seed mean | 0.621 | Frozen | 435K |

## Optuna trial 27 hyperparameters

`query_dim=64`, `n_cross_attn_layers=1`, `n_self_attn_layers=2`, `n_heads=4`, `alpha=0.5`, `batch_size=8`, `use_gate_M=false`, `lr≈4.8e-4`, `dropout≈0.10`

## vs legacy GNN (calibrated 4-way, different protocol)

| Model | Calibrated test F1 |
|-------|-------------------|
| gnn13 | 0.689 |
| gnn12 | 0.678 |
| vlm_mlp | 0.654 |
