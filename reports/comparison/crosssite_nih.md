# NIH ChestX-ray14 cross-site evaluation

Protocol: `nih`. Train: CheXpert only. Test: NIH (run pipeline below to populate this table).

## Quick start (iaceserver)

```bash
conda activate mbzai
export PYTHONPATH=scripts

# Smoke (~500 images) — validate data + VLM + all models
python scripts/run_crosssite_nih.py --smoke --gpu_id 1

# Full NIH (~112k) after smoke OK
python scripts/run_crosssite_nih.py --models all --gpu_id 1
```

**Data:** `data/raw/nih_chestxray14/Data_Entry_2017.csv` and extracted `images_*` shards.

| Model | Test F1 @0.5 | Test AUROC | Test AUPRC | Test ECE | Test Brier | Trainable params |
|-------|--------------|------------|------------|----------|------------|------------------|
| (pending) | — | — | — | — | — | — |

Driver: `scripts/run_crosssite_nih.py`

See also: [`reports/comparison/crosssite_nih_stats.md`](crosssite_nih_stats.md).
