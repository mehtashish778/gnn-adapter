# Faithfulness–utility Pareto (λ_sparse sweep, frozen patches, gate on)

Config: default CCA (~435K params), `lambda_faithful=0.1`, `use_gate_M=true`, frozen CLIP patches.

| run_id | λ_sparse | gate density | test F1 | test AUROC | intervention consistency | necessity drop |
|--------|----------|--------------|---------|------------|--------------------------|----------------|
| faith_pareto_ls1e3 | 1e-03 | 0.419 | 0.6763 | 0.6675 | 0.5798 | 0.5207 |
| cca_faithful | 1e-02 | 0.438 | 0.6737 | — | 0.5541 | 0.2903 |
| faith_pareto_ls1e1 | 1e-01 | 0.419 | 0.6763 | 0.6675 | 0.5798 | 0.5207 |

Utility axis: test macro-F1 @0.5. Faithfulness axis: intervention consistency (higher = more faithful).

See also: [`docs/cca_experiment_results.md`](../docs/cca_experiment_results.md).