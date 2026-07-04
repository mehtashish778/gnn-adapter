# NIH ChestX-ray14 cross-site evaluation

Protocol: `nih`. Train: CheXpert only. Test: NIH (6,000 images, **subset cap 6,000**).

| Model | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Trainable params |
|-------|--------------|------------|------------|----------|------------|------------------|
| vlm_zeroshot | 0.0592 | 0.5237 | 0.0573 | 0.2793 | 0.1957 | 0 |
| vlm_mlp | 0.0823 | 0.4828 | 0.0552 | 0.4520 | 0.2748 | — |
| cbm_posthoc | 0.0534 | 0.4893 | 0.0556 | 0.4311 | 0.2470 | 667 |
| mlgcn | 0.0861 | 0.5441 | 0.0584 | 0.8907 | 0.8679 | 8,577 |
| gnn07_label_residual | — | — | — | — | — | — |
| cca | 0.1358 | 0.6332 | 0.1048 | 0.3712 | 0.2570 | 118,891 |
| qformer_adapter | 0.1317 | 0.6426 | 0.1043 | 0.3363 | 0.1952 | 263,815 |
| cbm_labelfree | 0.0521 | 0.5389 | 0.0664 | 0.4405 | 0.2455 | 217 |
| gnn12_clip_vlm_homo | — | — | — | — | — | — |
| gnn13_clip_bipartite | — | — | — | — | — | — |
| qwen2vl_lora_r16 | 0.1137 | 0.6117 | 0.0829 | 0.3472 | 0.2002 | 18,475,527 |

## Headline pair (LoRA-16 cls vs CCA trial-27)

- ΔF1 (LoRA − CCA): -0.0221
- ΔAUROC (LoRA − CCA): -0.0216
- CCA params are ~0.64% of LoRA-16 cls (118,891 vs 18,475,527).

Driver: `scripts/run_crosssite_nih.py`

See also: [`reports/comparison/crosssite_nih_stats.md`](crosssite_nih_stats.md).