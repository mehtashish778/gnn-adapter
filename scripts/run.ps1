# Windows pipeline runner (replaces `make` when GNU make is not installed).
# Usage: .\scripts\run.ps1 help
#        .\scripts\run.ps1 data
#        $env:GPU=0; $env:RUN_ID="my_run"; .\scripts\run.ps1 baselines

param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$env:PYTHONPATH = Join-Path $Repo "scripts"
if ($env:PYTHONPATH_EXTRA) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $env:PYTHONPATH_EXTRA }

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$Gpu = if ($env:GPU) { $env:GPU } else { "0" }
$RunId = if ($env:RUN_ID) { $env:RUN_ID } else { "reproduce_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }

$Split = "data/processed/splits"
$Split4 = "data/processed/splits_4way"
$Graph = "data/processed/graph"
$Graph4 = "data/processed/graph_4way"
$ClipDefault = "data/processed/embeddings/chexpert_default_clip_vitb32_v1.pt"
$Clip4way = "data/processed/embeddings/chexpert_calibrated4way_clip_vitb32_v1.pt"

function Invoke-Py {
    param([string[]]$Args)
    & $Py @Args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Py $($Args -join ' ')" }
}

switch ($Target) {
    "help" {
        Write-Host "Targets: data, baselines, gnns, cca, cca_optuna, report, tables, reproduce"
        Write-Host "Set env: GPU, RUN_ID"
    }
    "data" {
        Invoke-Py @("scripts/01_build_canonical_labels.py")
        Invoke-Py @("scripts/02_align_vlm_outputs.py")
        Invoke-Py @("scripts/03_make_multilabel_splits.py")
        Invoke-Py @("scripts/03_make_multilabel_splits_4way.py")
        Invoke-Py @("scripts/04_build_coerror_graph.py", "--train_rows_json", "$Split/train_rows.json", "--out_dir", $Graph)
        Invoke-Py @("scripts/04_build_coerror_graph.py", "--train_rows_json", "$Split4/train_fit_rows.json", "--out_dir", $Graph4)
    }
    "baselines" {
        Invoke-Py @("scripts/05_run_baseline_frozen_vlm.py", "--model_id", "vlm_zeroshot", "--protocol", "default", "--run_id", $RunId, "--gpu_id", $Gpu, "--val_rows_json", "$Split/val_rows.json", "--test_rows_json", "$Split/test_rows.json")
        Invoke-Py @("scripts/06_run_baseline_mlp.py", "--model_id", "vlm_mlp", "--protocol", "default", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split/train_rows.json", "--val_rows_json", "$Split/val_rows.json", "--test_rows_json", "$Split/test_rows.json")
        Invoke-Py @("scripts/06_run_baseline_mlp.py", "--model_id", "vlm_mlp", "--protocol", "calibrated4way", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split4/train_fit_rows.json", "--calib_rows_json", "$Split4/calib_rows.json", "--val_rows_json", "$Split4/val_rows.json", "--test_rows_json", "$Split4/test_rows.json")
    }
    "gnns" {
        Invoke-Py @("scripts/07_train_gnn_adapter.py", "--model_id", "gnn07_label_residual", "--protocol", "default", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split/train_rows.json", "--val_rows_json", "$Split/val_rows.json", "--test_rows_json", "$Split/test_rows.json", "--edge_index_json", "$Graph/edge_index.json", "--edge_weight_json", "$Graph/edge_weight.json")
        Invoke-Py @("scripts/12_train_clip_vlm_gnn_adapter.py", "--model_id", "gnn12_clip_vlm_homo", "--protocol", "default", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split/train_rows.json", "--val_rows_json", "$Split/val_rows.json", "--test_rows_json", "$Split/test_rows.json", "--edge_index_json", "$Graph/edge_index.json", "--edge_weight_json", "$Graph/edge_weight.json", "--clip_cache_pt", $ClipDefault)
        Invoke-Py @("scripts/13_train_bipartite_gnn_adapter.py", "--model_id", "gnn13_clip_bipartite", "--protocol", "default", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split/train_rows.json", "--val_rows_json", "$Split/val_rows.json", "--test_rows_json", "$Split/test_rows.json", "--clip_cache_pt", $ClipDefault)
    }
    "cca" {
        Invoke-Py @("scripts/14_train_cca.py", "--model_id", "cca", "--protocol", "default", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split/train_rows.json", "--val_rows_json", "$Split/val_rows.json", "--test_rows_json", "$Split/test_rows.json")
        if (Test-Path "$Split4/calib_rows.json") {
            Invoke-Py @("scripts/14_train_cca.py", "--model_id", "cca", "--protocol", "calibrated4way", "--run_id", $RunId, "--gpu_id", $Gpu, "--train_rows_json", "$Split4/train_fit_rows.json", "--calib_rows_json", "$Split4/calib_rows.json", "--val_rows_json", "$Split4/val_rows.json", "--test_rows_json", "$Split4/test_rows.json")
        }
    }
    "cca_optuna" {
        Invoke-Py @(
            "scripts/tune_cca_optuna.py",
            "--model_id", "cca",
            "--protocol", "default",
            "--gpu_id", $Gpu,
            "--num_workers", "0",
            "--train_rows_json", "$Split/train_rows.json",
            "--val_rows_json", "$Split/val_rows.json",
            "--test_rows_json", "$Split/test_rows.json",
            "--n_trials", "20",
            "--tune_epochs", "25",
            "--final_epochs", "60"
        )
    }
    "report" { Invoke-Py @("scripts/11_package_report.py") }
    "tables" { Invoke-Py @("scripts/generate_tables.py", "--protocol", "calibrated4way") }
    "reproduce" {
        & $PSScriptRoot\run.ps1 data
        & $PSScriptRoot\run.ps1 baselines
        & $PSScriptRoot\run.ps1 gnns
        & $PSScriptRoot\run.ps1 report
        & $PSScriptRoot\run.ps1 tables
        Write-Host "For full 4-way calibration: bash scripts/reproduce_all_results.sh"
    }
    default { throw "Unknown target '$Target'. Use: .\scripts\run.ps1 help" }
}
