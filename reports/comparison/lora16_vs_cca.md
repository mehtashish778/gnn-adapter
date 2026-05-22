# LoRA-16 (Qwen2-VL) vs CCA — CheXpert default

Driver: `scripts/run_lora16_vs_cca.py`

**CCA reference (5-seed leaderboard):** F1 0.701 ± 0.005, AUROC 0.722 ± 0.004, ~118,891 trainable params (`cca_lora_r8_trial27`, seeds 0–4).

**Single-seed CCA ref run:** `data/processed/experiments/cca/default/lora_r8_trial27_seeds_s0`

| Model | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Trainable params | GPU-hours |
|-------|--------------|------------|------------|----------|------------|------------------|-----------|
| CCA (ref seed 0) (lora_r8_trial27_seeds_s0) | 0.7068 | 0.7172 | 0.6225 | 0.1146 | 0.1818 | 118,891 | — |
| Qwen2-VL LoRA-16 + cls head (qwen2vl_lora_r16_v2) | 0.5822 | 0.6851 | 0.5910 | 0.1079 | 0.1821 | 10,759 | 3.37 |
| Qwen2-VL LoRA-16 + JSON SFT (parse fail test=9197) (qwen2vl_lora_r16_sft_v3) | 0.6554 | 0.5000 | 0.4466 | 0.1912 | 0.2500 | 18,464,768 | 16.06 |

## Δ vs CCA ref (seed 0, test)

- **LoRA-16 cls:** ΔF1 -0.1246, ΔAUROC -0.0321
- **LoRA-16 SFT:** ΔF1 -0.0514, ΔAUROC -0.2172

## Cost note

CCA uses ~118,891 adapter params; LoRA-16 cls uses ~10,759. CCA is ~1105.04% of LoRA-16 cls trainable params (AAAI target: CCA < 0.1% of LoRA-16).

## Takeaways

1. **Headline PEFT baseline:** LoRA-16 + classification head (masked BCE) is the primary comparison to CCA.
2. **Protocol parity row:** JSON SFT matches the frozen zero-shot scoring path; parse failures are logged in `metrics.json`.
3. **CCA 5-seed mean** remains the leaderboard bar; LoRA runs here are single-seed unless promoted via `scripts/run_seeds.py`.

See also: [`docs/combined_experiments_report.md`](../docs/combined_experiments_report.md), [`docs/cca_experiment_results.md`](../docs/cca_experiment_results.md).

## stats_compare output

See [`reports/comparison/lora16_stats.md`](reports/comparison/lora16_stats.md).