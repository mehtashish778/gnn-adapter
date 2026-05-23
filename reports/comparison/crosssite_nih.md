# NIH ChestX-ray14 cross-site evaluation

Protocol: `nih`. Train: CheXpert only. Test: NIH (500 images, **smoke subset**).

| Model | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Trainable params |
|-------|--------------|------------|------------|----------|------------|------------------|
| vlm_zeroshot | 0.0687 | 0.5127 | 0.1017 | 0.2463 | 0.1975 | 0 |
| vlm_mlp | 0.1328 | 0.4956 | 0.1024 | 0.4111 | 0.2747 | — |
| cbm_posthoc | 0.0744 | 0.5152 | 0.1058 | 0.3941 | 0.2539 | 667 |
| mlgcn | 0.1448 | 0.5303 | 0.1004 | 0.8317 | 0.8062 | 8,577 |
| gnn07_label_residual | — | — | — | — | — | — |
| cca | 0.2062 | 0.6775 | 0.1862 | 0.3989 | 0.2938 | 118,891 |
| qformer_adapter | 0.2235 | 0.7095 | 0.2090 | 0.3692 | 0.2364 | 263,815 |
| cbm_labelfree | 0.0788 | 0.6498 | 0.1494 | 0.4000 | 0.2471 | 217 |
| gnn12_clip_vlm_homo | — | — | — | — | — | — |
| gnn13_clip_bipartite | — | — | — | — | — | — |
| qwen2vl_lora_r16 | 0.2040 | 0.6675 | 0.1800 | 0.3417 | 0.2178 | 18,475,527 |

## Headline pair (LoRA-16 cls vs CCA trial-27)

- ΔF1 (LoRA − CCA): -0.0021
- ΔAUROC (LoRA − CCA): -0.0100
- CCA params are ~0.64% of LoRA-16 cls (118,891 vs 18,475,527).

Driver: `scripts/run_crosssite_nih.py`

See also: [`reports/comparison/crosssite_nih_stats.md`](crosssite_nih_stats.md).