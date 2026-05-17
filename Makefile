# MBZAI pipeline entry points (Phase 0)
# Override: PY=python3 GPU=0 RUN_ID=my_run make reproduce

PY ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
endif

GPU ?= 0
RUN_ID ?= reproduce_$(shell date +%Y%m%d_%H%M%S)
export PYTHONPATH := $(CURDIR)/scripts$(if $(PYTHONPATH),:$(PYTHONPATH),)

SPLIT := data/processed/splits
SPLIT4 := data/processed/splits_4way
GRAPH := data/processed/graph
GRAPH4 := data/processed/graph_4way
CLIP_CACHE_DEFAULT := data/processed/embeddings/chexpert_default_clip_vitb32_v1.pt
CLIP_CACHE_4WAY := data/processed/embeddings/chexpert_calibrated4way_clip_vitb32_v1.pt

.PHONY: data baselines gnns cca calibrate report tables reproduce help

help:
	@echo "Targets: data, baselines, gnns, cca, calibrate, report, tables, reproduce"
	@echo "Windows (no make): powershell -File scripts/run.ps1 help"

data:
	$(PY) scripts/01_build_canonical_labels.py
	$(PY) scripts/02_align_vlm_outputs.py
	$(PY) scripts/03_make_multilabel_splits.py
	$(PY) scripts/03_make_multilabel_splits_4way.py
	$(PY) scripts/04_build_coerror_graph.py --train_rows_json $(SPLIT)/train_rows.json --out_dir $(GRAPH)
	$(PY) scripts/04_build_coerror_graph.py --train_rows_json $(SPLIT4)/train_fit_rows.json --out_dir $(GRAPH4)

baselines:
	$(PY) scripts/05_run_baseline_frozen_vlm.py --model_id vlm_zeroshot --protocol default --run_id $(RUN_ID) --gpu_id $(GPU) \
		--val_rows_json $(SPLIT)/val_rows.json --test_rows_json $(SPLIT)/test_rows.json
	$(PY) scripts/06_run_baseline_mlp.py --model_id vlm_mlp --protocol default --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT)/train_rows.json --val_rows_json $(SPLIT)/val_rows.json --test_rows_json $(SPLIT)/test_rows.json
	$(PY) scripts/06_run_baseline_mlp.py --model_id vlm_mlp --protocol calibrated4way --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT4)/train_fit_rows.json --calib_rows_json $(SPLIT4)/calib_rows.json \
		--val_rows_json $(SPLIT4)/val_rows.json --test_rows_json $(SPLIT4)/test_rows.json

gnns:
	$(PY) scripts/07_train_gnn_adapter.py --model_id gnn07_label_residual --protocol default --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT)/train_rows.json --val_rows_json $(SPLIT)/val_rows.json --test_rows_json $(SPLIT)/test_rows.json \
		--edge_index_json $(GRAPH)/edge_index.json --edge_weight_json $(GRAPH)/edge_weight.json
	$(PY) scripts/12_train_clip_vlm_gnn_adapter.py --model_id gnn12_clip_vlm_homo --protocol default --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT)/train_rows.json --val_rows_json $(SPLIT)/val_rows.json --test_rows_json $(SPLIT)/test_rows.json \
		--edge_index_json $(GRAPH)/edge_index.json --edge_weight_json $(GRAPH)/edge_weight.json \
		--clip_cache_pt $(CLIP_CACHE_DEFAULT)
	$(PY) scripts/13_train_bipartite_gnn_adapter.py --model_id gnn13_clip_bipartite --protocol default --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT)/train_rows.json --val_rows_json $(SPLIT)/val_rows.json --test_rows_json $(SPLIT)/test_rows.json \
		--clip_cache_pt $(CLIP_CACHE_DEFAULT)

cca:
	$(PY) scripts/14_train_cca.py --model_id cca --protocol default --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT)/train_rows.json --val_rows_json $(SPLIT)/val_rows.json \
		--test_rows_json $(SPLIT)/test_rows.json
	$(PY) scripts/14_train_cca.py --model_id cca --protocol calibrated4way --run_id $(RUN_ID) --gpu_id $(GPU) \
		--train_rows_json $(SPLIT4)/train_fit_rows.json --calib_rows_json $(SPLIT4)/calib_rows.json \
		--val_rows_json $(SPLIT4)/val_rows.json --test_rows_json $(SPLIT4)/test_rows.json

calibrate:
	@echo "Run scripts/reproduce_all_results.sh calibrated_eval_run for full 4-way calibration, or:"
	@echo "  $(PY) scripts/calibration.py (via 08/09 on each run_dir)"

report:
	$(PY) scripts/11_package_report.py

tables:
	$(PY) scripts/generate_tables.py --protocol calibrated4way

reproduce: data baselines gnns report tables
	@echo "Full GPU reproduction with calibration: bash scripts/reproduce_all_results.sh RUN_ID=$(RUN_ID) GPU=$(GPU)"
