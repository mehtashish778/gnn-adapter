# CCA Reproduction Guide

End-to-end recipe to reproduce every Concept-Evidence Adapter (CCA) run in [`docs/cca_experiment_results.md`](cca_experiment_results.md) — from raw CheXpert images to the leaderboard table — with exact hyperparameters for each variant.

> **Scope:** CCA + LoRA CLIP + faithfulness variants. For the legacy GNN/MLP baselines, use `bash scripts/reproduce_all_results.sh` (see [`README.md`](../README.md)).

---

## 0. Hardware, OS, environment

| Item | Reference value (what was used) |
|------|----------------------------------|
| OS | Windows 10 (PowerShell). Linux works; replace `` ` `` with `\` for line continuations. |
| GPU | NVIDIA RTX 4060 8 GB |
| CUDA | 12.1 (matches `requirements.txt` wheel index) |
| Python | **3.10–3.12** |
| Disk | ≥ 25 GB free under `data/processed/embeddings/` (patch caches are ~17.5 GB per encoder) |
| RAM | ≥ 16 GB |
| Wall-clock | Patch encode ~10 min · LoRA r=8 fine-tune ~10 min · single CCA train 5–15 min · Optuna 20-trial study ~3 h |

### Install

```powershell
cd C:\path\to\mbzai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Key packages: `torch` (CUDA 12.1), `transformers`, `peft`, `optuna`, `torchmetrics`, `scikit-learn`, `pillow`, `tqdm`, `pyarrow`.

### Set `PYTHONPATH` for every shell

All CCA scripts import sibling modules from `scripts/`. Either run from the repo root with:

```powershell
$env:PYTHONPATH = "scripts"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
```

…or invoke through the helpers in `scripts/run.ps1` / `Makefile` which set it automatically.

---

## 1. Data preparation (run once)

Build canonical labels, aligned VLM rows, splits, and the legacy co-error graph used by `vlm_mlp` / GNN baselines.

```powershell
python scripts/01_build_canonical_labels.py
python scripts/02_align_vlm_outputs.py
python scripts/03_make_multilabel_splits.py
python scripts/03_make_multilabel_splits_4way.py
python scripts/04_build_coerror_graph.py `
  --train_rows_json data/processed/splits/train_rows.json `
  --out_dir data/processed/graph
```

CCA itself only needs the **default** split files:

```
data/processed/splits/train_rows.json   # 43,778 rows
data/processed/splits/val_rows.json     #  9,357 rows
data/processed/splits/test_rows.json    #  9,197 rows
```

Each row must contain `path`, `x_logits`, `x_probs`, `y_true`, `y_mask`.

Configured in [`configs/data.yaml`](../configs/data.yaml). Full pipeline docs: [`docs/pipeline.md`](pipeline.md).

---

## 2. Patch caches (run once each)

CCA reads precomputed ViT patch tokens (B × 196 × 768). Two encoders are supported:

### 2a. Frozen CLIP ViT-B/16 (auto)

First CCA run **without** `--lora_rank` will encode and cache all 3 splits (~10 min, ~17.5 GB). No manual step needed; cached under:

```
data/processed/embeddings/chexpert_default_<split>_openai_clip-vit-base-patch16_patch_v2_fp16.pt
```

### 2b. LoRA-tuned CLIP ViT-B/16

Builds a CheXpert-adapted vision encoder, saves the adapter, then encodes train/val/test:

```powershell
python scripts/19_train_lora_clip_vision.py --lora_rank 8 --gpu_id 0
```

Defaults: 3 proxy epochs, `lora_alpha = 2 × r`, targets `q_proj` / `v_proj`, batch 8, lr 1e-4. Outputs:

```
data/processed/embeddings/lora_r8_adapter/                                        # PEFT adapter
data/processed/embeddings/chexpert_default_<split>_..._lora_r8_patch_v2_fp16_lora_r8.pt
data/processed/embeddings/lora_r8_meta.json
```

If the adapter already exists and you only need to re-encode splits:

```powershell
python scripts/19_train_lora_clip_vision.py --lora_rank 8 --gpu_id 0 --encode_only
```

For ranks 4 or 16: change `--lora_rank`. Cache keys become `_lora_r{4|16}`.

---

## 3. Reproduce every CCA leaderboard run

All runs land in `data/processed/experiments/cca/default/<run_id>/`:

```
best_checkpoint.pt
metrics.json
val_predictions.json
test_predictions.json
attention_maps.pt
```

Common flags omitted from the per-run commands below for brevity (assume present in every command):

```text
--model_id cca --protocol default --gpu_id 0 --num_workers 0 --seed 42 --epochs 60
```

### 3.1. `cca_lora_r8_trial27` — **best overall** (test F1 @0.5 = 0.701)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27 --best_metric val_macro_f1_05 --epochs 60 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text
```

### 3.2. `cca_frozen_trial27_f1` — frozen patches, same hparams (test F1 = 0.694)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --run_id cca_frozen_trial27_f1 --best_metric val_macro_f1_05 --epochs 60 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text
```

### 3.3. `cca_lora_r8_default` — LoRA + default arch + gate (test F1 = 0.677)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_default --best_metric val_macro_f1_05 --epochs 60 `
  --num_primitives 30 --query_dim 128 --n_cross_attn_layers 2 --n_self_attn_layers 2 `
  --n_heads 2 --alpha 1.0 --dropout 0.1 --batch_size 16 `
  --use_gate_M --init_queries_from_text
```

### 3.4. `cca_lora_r8_faithful` — LoRA + default arch + faithfulness (test F1 = 0.677)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_faithful --best_metric val_macro_f1_05 --epochs 60 `
  --num_primitives 30 --query_dim 128 --n_cross_attn_layers 2 --n_self_attn_layers 2 `
  --n_heads 2 --alpha 1.0 --dropout 0.1 --batch_size 16 `
  --use_gate_M --init_queries_from_text `
  --lambda_sparse 0.01 --lambda_faithful 0.1
```

### 3.5. `cca_lora_r8_trial27_faithful` — LoRA + trial-27 + faithfulness (test F1 = 0.701)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --lora_rank 8 --run_id cca_lora_r8_trial27_faithful --best_metric val_macro_f1_05 --epochs 60 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text `
  --lambda_sparse 0.01 --lambda_faithful 0.1
```

### 3.6. `cca_faithful` — frozen + default arch + faithfulness (test F1 = 0.674)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --run_id cca_faithful --best_metric val_macro_f1_05 --epochs 60 `
  --use_gate_M --lambda_sparse 0.01 --lambda_faithful 0.1
```

(All non-listed args use the script defaults; equivalent to **default** CCA + faithfulness.)

### 3.7. `run_20260516_183647` — original default CCA (test F1 = 0.653)

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0
```

This is the **out-of-the-box** behaviour (default architecture, `val_bce` checkpoint). Run id is auto-generated as `run_<timestamp>`.

### 3.8. `best_optuna_cca_hpo` — Optuna final 60-ep run (test F1 = 0.658)

This is the artifact emitted by the Optuna script after HPO. Reproduce with:

```powershell
python scripts/tune_cca_optuna.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --n_trials 20 --tune_epochs 25 --tune_early_stop_patience 8 `
  --final_epochs 60 --final_early_stop_patience 16
```

Or skip the search and just reproduce the **final** train with trial-27 hparams on `val_bce`:

```powershell
python scripts/14_train_cca.py --model_id cca --protocol default --gpu_id 0 --num_workers 0 `
  --run_id best_optuna_cca_hpo --best_metric val_bce --epochs 60 `
  --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
  --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
  --batch_size 8 --no-use_gate_M --init_queries_from_text
```

### 3.9. 5-seed default CCA — multi-seed stability

```powershell
python scripts/run_seeds.py --model_id cca --protocol default --seeds 0,1,2,3,4 `
  --use_numbered_script --stats_after `
  --stats_models cca vlm_mlp mlgcn qformer_adapter
```

Outputs:
- `data/processed/experiments/cca/default/seeds_s{0..4}/metrics.json`
- `data/processed/experiments/cca/default/seeds_summary.parquet`
- `reports/comparison/stats.md` (bootstrap CI + paired AUROC)

### 3.10. Batch driver for all LoRA + frozen variants

The five rows in §3.1–3.5 are also covered by:

```powershell
python scripts/run_cca_lora_variants.py --gpu_id 0 --skip_existing
python scripts/run_cca_lora_variants.py --compare_only
```

Writes [`reports/comparison/cca_lora_variants.md`](../reports/comparison/cca_lora_variants.md).

---

## 4. Complete hyperparameter reference (`scripts/14_train_cca.py`)

All flags resolved from `scripts/cca_train_core.py::build_argparser`.

### Data / I/O

| Flag | Default | Description |
|------|---------|-------------|
| `--train_rows_json` | `data/processed/splits/train_rows.json` | Train rows JSON |
| `--val_rows_json` | `data/processed/splits/val_rows.json` | Val rows JSON |
| `--test_rows_json` | `data/processed/splits/test_rows.json` | Test rows JSON |
| `--calib_rows_json` | `None` | Optional calib split (for `calibrated4way`) |
| `--per_class_thresholds_json` | `data/processed/experiments/thresholds/per_class_thresholds.json` | Per-class thresholds for `@thr` metric |
| `--image_root` | `data/raw` | Root for image paths (used only when encoding patches) |
| `--protocol` | `default` | `default` or `calibrated4way` |
| `--run_id` | `""` | Identifier; empty → auto-timestamp |
| `--out_dir` | `""` | Override output dir (else `data/processed/experiments/<model_id>/<protocol>/<run_id>`) |
| `--model_id` | `cca` | Registry key |
| `--resume_from` | `""` | Resume from checkpoint path |

### Vision encoder / patch cache

| Flag | Default | Description |
|------|---------|-------------|
| `--clip_model` | `openai/clip-vit-base-patch16` | HF id of CLIP ViT |
| `--lora_rank` | `None` | One of `{4,8,16}`. Sets LoRA patch cache keys automatically. |
| `--patch_encoder_id` | `""` | Override FeatureCache encoder id (e.g. custom LoRA) |
| `--patch_cache_version` | `""` | Override FeatureCache version |
| `--embeddings_cache_dir` | `data/processed/embeddings` | Disk cache dir (~17.5 GB per encoder) |
| `--clip_cache_pt` | `""` | Legacy combined `.pt` cache (avoid for new runs) |
| `--clip_batch_size` | `16` | Batch size during patch encoding only |

### CCA architecture

| Flag | Default | Best (trial 27) | Description |
|------|---------|-----------------|-------------|
| `--num_primitives` | `30` | `30` | P, number of concept queries |
| `--query_dim` | `128` | `64` | Hidden dim for queries / cross-attn / readout |
| `--patch_dim` | `768` | `768` | ViT patch dim (auto-detected from cache) |
| `--n_heads` | `2` | `4` | Multi-head attention heads |
| `--n_cross_attn_layers` | `2` | `1` | Layer 1 cross-attn depth (queries ↔ patches) |
| `--n_self_attn_layers` | `2` | `2` | Layer 2 self-attn depth over primitives |
| `--alpha` | `1.0` | `0.5` | VLM residual gain in Layer 3 |
| `--dropout` | `0.1` | `0.1001` | Dropout in attention + MLPs |
| `--init_queries_from_text` / `--no-init_queries_from_text` | `True` | `True` | Init concept queries from CLIP text embeddings of `DEFAULT_CONCEPT_PHRASES` |
| `--use_gate_M` / `--no-use_gate_M` | `True` | `False` | Sparse `C×P` GumbelGate on primitive→finding edges |

### Faithfulness losses (Phase 2)

| Flag | Default | Used in `cca_faithful` |
|------|---------|------------------------|
| `--lambda_sparse` | `0.0` | `0.01` |
| `--lambda_faithful` | `0.0` | `0.1` |
| `--sparsity_target` | `0.10` | `0.10` |
| `--gumbel_tau_init` | `1.0` | `1.0` |
| `--gumbel_tau_min` | `0.5` | `0.5` |
| `--gumbel_anneal_epochs` | `10` | `10` |
| `--intervention_per_step` | `1` | `1` (0 disables intervention loss) |

### Concept prior (Phase 3)

| Flag | Default | Description |
|------|---------|-------------|
| `--radgraph_prior_json` | `""` | Path to P×P prior JSON (`scripts/build_concept_prior.py` output) |

### Optimisation

| Flag | Default | Best (trial 27) |
|------|---------|-----------------|
| `--epochs` | `60` | `60` |
| `--batch_size` | `16` | `8` |
| `--lr` | `3e-4` | `4.79e-4` |
| `--min_lr` | `1e-6` | `1e-6` |
| `--weight_decay` | `1e-4` | `1.11e-4` |
| `--grad_clip_norm` | `1.0` | `1.0` |
| `--pos_weight_max` | `100.0` | `100.0` |
| `--lr_scheduler` | `cosine` | `cosine` |
| `--plateau_factor` | `0.5` | (only used if scheduler=`plateau`) |
| `--plateau_patience` | `6` | — |
| `--warmup_epochs` | `2` | `2` |

### Checkpoint / early-stop

| Flag | Default | Recommended |
|------|---------|-------------|
| `--best_metric` | `val_bce` | **`val_macro_f1_05`** for reporting |
| `--early_stop_patience` | `16` | `16` |
| `--seed` | `42` | use `0..4` for multi-seed |
| `--num_workers` | `4` | `0` on Windows |
| `--gpu_id` | `0` | match your CUDA device |

### Output of every run

`data/processed/experiments/cca/<protocol>/<run_id>/metrics.json` contains:

- `trainable_params`, `epochs_ran`
- `val_macro_f1@0.5`, `test_macro_f1@0.5`
- `val_macro_f1@per_class_thr`, `test_macro_f1@per_class_thr`
- `val_subset_accuracy@0.5`, `test_subset_accuracy@0.5`
- `patch_encoder_id`, `patch_cache_version` (provenance)
- `best_metric`, `best_score`
- If faithfulness enabled: `gate_density_eval`, `faithfulness_f1_full`, `faithfulness_necessity_drop`, `faithfulness_sufficiency_f1`, `intervention_consistency`
- `hparams`: full `vars(args)` snapshot

---

## 5. LoRA fine-tune flags (`scripts/19_train_lora_clip_vision.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_rank` | `8` | One of `{4,8,16}`; sets `lora_alpha = 2*r`, targets `q_proj`/`v_proj` |
| `--epochs` | `3` | Proxy multi-label BCE training epochs |
| `--batch_size` | `8` | |
| `--lr` | `1e-4` | |
| `--gpu_id` | `0` | |
| `--clip_model` | `openai/clip-vit-base-patch16` | |
| `--train_rows_json` / `--val_rows_json` / `--test_rows_json` | default splits | |
| `--calib_rows_json` | `""` | Encode calib split too (4-way protocol) |
| `--adapter_out_dir` | `embeddings/lora_r{r}_adapter` | Where to save PEFT adapter |
| `--encode_only` | `False` | Skip fine-tune; load existing adapter and encode missing splits |
| `--embeddings_cache_dir` | `data/processed/embeddings` | |
| `--image_root` | `data/raw` | |

---

## 6. Optuna search (`scripts/tune_cca_optuna.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--n_trials` | `20` | New trials per invocation (resumes from DB) |
| `--tune_epochs` | `25` | Per-trial epoch budget |
| `--tune_early_stop_patience` | `8` | Per-trial early stop |
| `--final_epochs` | `60` | Final retrain after HPO |
| `--final_early_stop_patience` | `16` | Final early stop |
| `--skip_final_train` | `False` | Stop after HPO |
| `--study_name` | `cca_hpo` | Optuna study name |
| `--storage` | `sqlite:///data/processed/experiments/cca/optuna/study.db` | Optuna RDB |

Search space documented in [`docs/cca_optuna_hpo.md`](cca_optuna_hpo.md#search-space).

---

## 7. Comparison / reporting

```powershell
# Refresh LoRA variant table from metrics.json files
python scripts/run_cca_lora_variants.py --compare_only

# Multi-seed bootstrap (after run_seeds.py)
python scripts/stats_compare.py --protocol default `
  --models cca vlm_mlp mlgcn qformer_adapter --reference cca
```

Generated files:

| File | Source |
|------|--------|
| [`reports/comparison/cca_lora_variants.md`](../reports/comparison/cca_lora_variants.md) | `run_cca_lora_variants.py --compare_only` |
| [`reports/comparison/cca_optuna_summary.md`](../reports/comparison/cca_optuna_summary.md) | manual |
| [`reports/comparison/stats.md`](../reports/comparison/stats.md) | `stats_compare.py` |
| [`docs/cca_experiment_results.md`](cca_experiment_results.md) | manual aggregation |

---

## 8. Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `error: unrecognized arguments: --no-use-gate-M` | `argparse.BooleanOptionalAction` uses the **dest** name (underscore) | Use **`--no-use_gate_M`**, not `--no-use-gate-M` |
| `FileNotFoundError: shutil.disk_usage('')` | Old code on Windows with relative cache dir | Pull latest `feature_cache.py` (`_disk_usage_target` resolves parent) |
| `Missing LoRA patch cache for split(s): ['test']` | LoRA adapter created earlier without test encoding | `python scripts/19_train_lora_clip_vision.py --lora_rank 8 --encode_only` |
| `LoRA patch cache cannot be used with --clip_cache_pt` | Mixing legacy `.pt` cache with new encoder id | Drop `--clip_cache_pt`; the FeatureCache handles per-split files |
| `Patch cache (fp16) estimate: ~17.5 GB` warning | Need free disk | Move `--embeddings_cache_dir` to a larger drive |
| Optuna final test F1 lower than tuning trial | Final train uses `val_bce` checkpoint by default | Re-run with `--best_metric val_macro_f1_05` |
| Slow start, no progress | Encoding ViT patches for the first time | Wait ~10 min for `chexpert_default_*_patch_v2_fp16.pt` to materialize |
| `TF_CPP_MIN_LOG_LEVEL` noise from transformers | TensorFlow checkpoint hooks | `$env:TF_CPP_MIN_LOG_LEVEL = "2"` (PowerShell) before running |
| Drift across runs | Different seed / `--best_metric` | Pin `--seed 42 --best_metric val_macro_f1_05` |

---

## 9. Sanity-check after reproduction

After running §3.1 the output `metrics.json` should be **bit-close** to:

```json
{
  "trainable_params": 118891,
  "patch_encoder_id": "openai_clip-vit-base-patch16_lora_r8",
  "patch_cache_version": "patch_v2_fp16_lora_r8",
  "best_metric": "val_macro_f1_05",
  "best_score": 0.7044,
  "val_macro_f1@0.5": 0.7044,
  "test_macro_f1@0.5": 0.7012,
  "test_macro_f1@per_class_thr": 0.6816,
  "epochs_ran": 23
}
```

Small deviations (±0.005) are expected with non-deterministic CUDA kernels and different GPU drivers. The **patch_encoder_id** / **patch_cache_version** fields are exact and confirm you used the right cache.

---

## 10. Recommended reporting workflow

For a paper-grade report:

1. Run §3.1 (best LoRA + trial-27) → primary number.
2. Run §3.2 (frozen + trial-27) → ablation of LoRA value (≈ +0.7 pt).
3. Run §3.7 (default CCA) → ablation of HPO value.
4. Run §3.9 (5 seeds) → confidence intervals — **rerun with trial-27 hparams** for a fair CI:

   ```powershell
   python scripts/run_seeds.py --model_id cca --protocol default --seeds 0,1,2,3,4 `
     --use_numbered_script --run_id_prefix seeds_lora_trial27 -- `
     --lora_rank 8 --best_metric val_macro_f1_05 --epochs 60 `
     --num_primitives 30 --query_dim 64 --n_cross_attn_layers 1 --n_self_attn_layers 2 `
     --n_heads 4 --alpha 0.5 --dropout 0.1001 --lr 0.000479 --weight_decay 0.000111 `
     --batch_size 8 --no-use_gate_M --init_queries_from_text
   ```

5. (Optional) calibrated 4-way protocol for fair comparison to legacy gnn13 (0.689 calibrated test).
6. Update [`docs/cca_experiment_results.md`](cca_experiment_results.md) leaderboard.
