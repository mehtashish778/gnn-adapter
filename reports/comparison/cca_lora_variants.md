# CCA variant comparison (LoRA vs frozen CLIP patches)

**Full experiment log:** [`docs/cca_experiment_results.md`](../../docs/cca_experiment_results.md)  
**Reproduce each row (exact commands):** [`docs/cca_reproduction.md`](../../docs/cca_reproduction.md#3-reproduce-every-cca-leaderboard-run)

All runs: `default` protocol, checkpoint `val_macro_f1_05`, seed 42 unless noted.

| Rank | Run ID | Patches | Val F1 @0.5 | Test F1 @0.5 | Test F1 @thr | Params | Epochs |
|------|--------|---------|-------------|--------------|--------------|--------|--------|
| 1 | `cca_lora_r8_trial27` | lora | 0.7044 | 0.7012 | 0.6816 | 118891 | 23 |
| 2 | `cca_lora_r8_trial27_faithful` | lora | 0.7044 | 0.7012 | 0.6816 | 118891 | 23 |
| 3 | `cca_frozen_trial27_f1` | frozen | 0.7016 | 0.6940 | 0.6700 | 118891 | 34 |
| 4 | `cca_lora_r8_default` | lora | 0.6827 | 0.6770 | 0.6554 | 435261 | 18 |
| 5 | `cca_lora_r8_faithful` | lora | 0.6859 | 0.6767 | 0.6594 | 435261 | 24 |

## Labels

- **cca_lora_r8_trial27**: LoRA + Optuna trial-27 (no gate)
- **cca_lora_r8_default**: LoRA + default CCA (gate on)
- **cca_lora_r8_faithful**: LoRA + default + faithfulness (gate)
- **cca_lora_r8_trial27_faithful**: LoRA + trial-27 + faithfulness (no gate)
- **cca_frozen_trial27_f1**: Frozen CLIP + trial-27 (F1 ckpt, baseline)

## Δ test F1 vs frozen trial-27 (0.6940)

- `cca_lora_r8_trial27`: **+0.0072** (0.7012)
- `cca_lora_r8_trial27_faithful`: **+0.0072** (0.7012)
- `cca_lora_r8_default`: **-0.0171** (0.6770)
- `cca_lora_r8_faithful`: **-0.0174** (0.6767)

## Notes

- **trial-27 + LoRA** wins; default 435K architecture underperforms on LoRA patches too.
- **Faithfulness** on trial-27 (no gate) did not change F1 vs trial-27 alone; on default arch, faithfulness ≈ same F1 with gate density ~0.42.
- Prior **frozen** `cca_faithful` (default arch): test **0.674** — below all LoRA trial-27 runs.

**Best test F1 @0.5:** `cca_lora_r8_trial27` = 0.7012