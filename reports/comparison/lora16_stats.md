# Multi-seed / baseline comparison

Protocol: `default`. CCA seed group: `lora_r8_trial27`.

## Test macro-F1 @0.5 (bootstrap 95% CI)

| Model | mean F1 | 95% CI | n | runs |
|-------|---------|--------|---|------|
| cca | 0.7010 | [0.6976, 0.7045] | 5 | multi-seed (lora_r8_trial27_seeds, n=5) |
| qwen2vl_lora_r16 | 0.5822 | [0.5822, 0.5822] | 1 | single (qwen2vl_lora_r16_v2) |
| qwen2vl_lora_r16_sft | 0.0090 | [0.0090, 0.0090] | 1 | single (qwen2vl_lora_r16_sft_v2) |

## Test macro-AUROC (bootstrap 95% CI)

| Model | mean AUROC | 95% CI | n | runs |
|-------|------------|--------|---|------|
| cca | 0.7221 | [0.7190, 0.7249] | 5 | multi-seed (lora_r8_trial27_seeds, n=5) |
| qwen2vl_lora_r16 | 0.6851 | [0.6851, 0.6851] | 1 | single (qwen2vl_lora_r16_v2) |
| qwen2vl_lora_r16_sft | 0.5014 | [0.5014, 0.5014] | 1 | single (qwen2vl_lora_r16_sft_v2) |

## Bootstrap AUROC vs `cca` (paired on test set; ref run `lora_r8_trial27_seeds_s0`)

P-value: paired bootstrap on mean per-class AUROC (400 resamples). BH correction at q=0.05.

| Model | Δ mean AUROC | p (bootstrap) | BH reject |
|-------|--------------|---------------|-----------|
| qwen2vl_lora_r16 | -0.0321 | 1.0000 | no |
| qwen2vl_lora_r16_sft | -0.2159 | 0.9500 | no |