# Multi-seed / baseline comparison

Protocol: `nih`. CCA seed group: `lora_r8_trial27`.

## Test macro-F1 @0.5 (bootstrap 95% CI)

| Model | mean F1 | 95% CI | n | runs |
|-------|---------|--------|---|------|
| vlm_zeroshot | 0.0801 | [0.0801, 0.0801] | 1 | single (crosssite_eval) |
| vlm_mlp | 0.1549 | [0.1549, 0.1549] | 1 | single (crosssite_eval) |
| cbm_posthoc | 0.0868 | [0.0868, 0.0868] | 1 | single (crosssite_eval) |
| mlgcn | 0.1689 | [0.1689, 0.1689] | 1 | single (crosssite_eval) |
| gnn07_label_residual | 0.0000 | [0.0000, 0.0000] | 0 | none |
| cca | 0.2405 | [0.2405, 0.2405] | 1 | single (crosssite_eval) |
| qformer_adapter | 0.2608 | [0.2608, 0.2608] | 1 | single (crosssite_eval) |
| cbm_labelfree | 0.0919 | [0.0919, 0.0919] | 1 | single (crosssite_eval) |
| gnn12_clip_vlm_homo | 0.0000 | [0.0000, 0.0000] | 0 | none |
| gnn13_clip_bipartite | 0.0000 | [0.0000, 0.0000] | 0 | none |
| qwen2vl_lora_r16 | 0.2380 | [0.2380, 0.2380] | 1 | single (crosssite_eval) |

## Test macro-AUROC (bootstrap 95% CI)

| Model | mean AUROC | 95% CI | n | runs |
|-------|------------|--------|---|------|
| vlm_zeroshot | 0.5127 | [0.5127, 0.5127] | 1 | single (crosssite_eval) |
| vlm_mlp | 0.4956 | [0.4956, 0.4956] | 1 | single (crosssite_eval) |
| cbm_posthoc | 0.5152 | [0.5152, 0.5152] | 1 | single (crosssite_eval) |
| mlgcn | 0.5303 | [0.5303, 0.5303] | 1 | single (crosssite_eval) |
| gnn07_label_residual | 0.0000 | [0.0000, 0.0000] | 0 | none |
| cca | 0.6775 | [0.6775, 0.6775] | 1 | single (crosssite_eval) |
| qformer_adapter | 0.7095 | [0.7095, 0.7095] | 1 | single (crosssite_eval) |
| cbm_labelfree | 0.6498 | [0.6498, 0.6498] | 1 | single (crosssite_eval) |
| gnn12_clip_vlm_homo | 0.0000 | [0.0000, 0.0000] | 0 | none |
| gnn13_clip_bipartite | 0.0000 | [0.0000, 0.0000] | 0 | none |
| qwen2vl_lora_r16 | 0.6675 | [0.6675, 0.6675] | 1 | single (crosssite_eval) |

## Bootstrap AUROC vs `cca` (paired on test set; ref run `crosssite_eval`)

P-value: paired bootstrap on mean per-class AUROC (400 resamples). BH correction at q=0.05.

| Model | Δ mean AUROC | p (bootstrap) | BH reject |
|-------|--------------|---------------|-----------|
| vlm_zeroshot | -0.1647 | 0.9750 | no |
| vlm_mlp | -0.1818 | 0.9700 | no |
| cbm_posthoc | -0.1623 | 0.9550 | no |
| mlgcn | -0.1471 | 0.9300 | no |
| qformer_adapter | +0.0321 | 0.9500 | no |
| cbm_labelfree | -0.0277 | 0.9950 | no |
| qwen2vl_lora_r16 | -0.0100 | 0.9650 | no |