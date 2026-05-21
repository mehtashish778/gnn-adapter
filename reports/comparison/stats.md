# Multi-seed / baseline comparison

Protocol: `default`. CCA seed group: `lora_r8_trial27`.

## Test macro-F1 @0.5 (bootstrap 95% CI)

| Model | mean F1 | 95% CI | n | runs |
|-------|---------|--------|---|------|
| cca | 0.7010 | [0.6976, 0.7045] | 5 | multi-seed (lora_r8_trial27_seeds, n=5) |
| qformer_adapter | 0.6755 | [0.6755, 0.6755] | 1 | single (qformer_adapter_default) |
| cbm_posthoc | 0.6214 | [0.6214, 0.6214] | 1 | single (cbm_posthoc_default) |
| cbm_labelfree | 0.4760 | [0.4760, 0.4760] | 1 | single (cbm_labelfree_default) |
| mlgcn | 0.4700 | [0.4700, 0.4700] | 1 | single (mlgcn_default) |

## Test macro-AUROC (bootstrap 95% CI)

| Model | mean AUROC | 95% CI | n | runs |
|-------|------------|--------|---|------|
| cca | 0.7221 | [0.7190, 0.7249] | 5 | multi-seed (lora_r8_trial27_seeds, n=5) |
| qformer_adapter | 0.7077 | [0.7077, 0.7077] | 1 | single (qformer_adapter_default) |
| cbm_posthoc | 0.5420 | [0.5420, 0.5420] | 1 | single (cbm_posthoc_default) |
| cbm_labelfree | 0.6160 | [0.6160, 0.6160] | 1 | single (cbm_labelfree_default) |
| mlgcn | 0.5346 | [0.5346, 0.5346] | 1 | single (mlgcn_default) |

## Bootstrap AUROC vs `cca` (paired on test set; ref run `lora_r8_trial27_seeds_s0`)

P-value: paired bootstrap on mean per-class AUROC (400 resamples). BH correction at q=0.05.

| Model | Δ mean AUROC | p (bootstrap) | BH reject |
|-------|--------------|---------------|-----------|
| qformer_adapter | -0.0096 | 0.9950 | no |
| cbm_posthoc | -0.1752 | 0.9100 | no |
| cbm_labelfree | -0.1012 | 0.9650 | no |
| mlgcn | -0.1826 | 0.9750 | no |