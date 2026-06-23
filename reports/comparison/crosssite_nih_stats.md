# Multi-seed / baseline comparison

Protocol: `nih`. CCA seed group: `lora_r8_trial27`.

## Test macro-F1 @0.5 (bootstrap 95% CI)

| Model | mean F1 | 95% CI | n | runs |
|-------|---------|--------|---|------|
| vlm_zeroshot | 0.0690 | [0.0690, 0.0690] | 1 | single (crosssite_eval) |
| vlm_mlp | 0.0960 | [0.0960, 0.0960] | 1 | single (crosssite_eval) |
| cbm_posthoc | 0.0623 | [0.0623, 0.0623] | 1 | single (crosssite_eval) |
| mlgcn | 0.1005 | [0.1005, 0.1005] | 1 | single (crosssite_eval) |
| gnn07_label_residual | 0.0000 | [0.0000, 0.0000] | 0 | none |
| cca | 0.1584 | [0.1584, 0.1584] | 1 | single (crosssite_eval) |
| qformer_adapter | 0.1537 | [0.1537, 0.1537] | 1 | single (crosssite_eval) |
| cbm_labelfree | 0.0608 | [0.0608, 0.0608] | 1 | single (crosssite_eval) |
| gnn12_clip_vlm_homo | 0.0000 | [0.0000, 0.0000] | 0 | none |
| gnn13_clip_bipartite | 0.0000 | [0.0000, 0.0000] | 0 | none |
| qwen2vl_lora_r16 | 0.1327 | [0.1327, 0.1327] | 1 | single (crosssite_eval) |

## Test macro-AUROC (bootstrap 95% CI)

| Model | mean AUROC | 95% CI | n | runs |
|-------|------------|--------|---|------|
| vlm_zeroshot | 0.5237 | [0.5237, 0.5237] | 1 | single (crosssite_eval) |
| vlm_mlp | 0.4828 | [0.4828, 0.4828] | 1 | single (crosssite_eval) |
| cbm_posthoc | 0.4893 | [0.4893, 0.4893] | 1 | single (crosssite_eval) |
| mlgcn | 0.5441 | [0.5441, 0.5441] | 1 | single (crosssite_eval) |
| gnn07_label_residual | 0.0000 | [0.0000, 0.0000] | 0 | none |
| cca | 0.6332 | [0.6332, 0.6332] | 1 | single (crosssite_eval) |
| qformer_adapter | 0.6426 | [0.6426, 0.6426] | 1 | single (crosssite_eval) |
| cbm_labelfree | 0.5389 | [0.5389, 0.5389] | 1 | single (crosssite_eval) |
| gnn12_clip_vlm_homo | 0.0000 | [0.0000, 0.0000] | 0 | none |
| gnn13_clip_bipartite | 0.0000 | [0.0000, 0.0000] | 0 | none |
| qwen2vl_lora_r16 | 0.6117 | [0.6117, 0.6117] | 1 | single (crosssite_eval) |

## Bootstrap AUROC vs `cca` (paired on test set; ref run `crosssite_eval`)

P-value: paired bootstrap on mean per-class AUROC (400 resamples). BH correction at q=0.05.

| Model | Δ mean AUROC | p (bootstrap) | BH reject |
|-------|--------------|---------------|-----------|
| vlm_zeroshot | -0.1095 | 0.9550 | no |
| vlm_mlp | -0.1504 | 0.9950 | no |
| cbm_posthoc | -0.1439 | 0.9900 | no |
| mlgcn | -0.0891 | 0.9950 | no |
| qformer_adapter | +0.0094 | 0.9800 | no |
| cbm_labelfree | -0.0944 | 0.9400 | no |
| qwen2vl_lora_r16 | -0.0216 | 0.9900 | no |