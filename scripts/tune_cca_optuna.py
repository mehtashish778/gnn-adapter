#!/usr/bin/env python3
"""
Optuna hyperparameter search for CCA.

Loads patch caches once, runs many short training trials, then optionally
re-trains with the best config and full epoch budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from cca_train_core import build_argparser, load_cca_data, train_cca
from common_multilabel import require_cuda_device, set_seed, write_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    base = build_argparser()
    parser = argparse.ArgumentParser(
        description="Optuna HPO for CCA (see cca_train_core for data/model flags).",
        parents=[base],
        conflict_handler="resolve",
        add_help=True,
    )
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--study_name", default="cca_hpo", help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default="sqlite:///data/processed/experiments/cca/optuna/study.db",
        help="Optuna storage URL (sqlite recommended).",
    )
    parser.add_argument(
        "--tune_epochs",
        type=int,
        default=25,
        help="Epoch budget per trial (keep lower than --final_epochs).",
    )
    parser.add_argument(
        "--tune_early_stop_patience",
        type=int,
        default=8,
        help="Early-stop patience during tuning trials.",
    )
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=60,
        help="Epochs for final training with best hyperparameters.",
    )
    parser.add_argument(
        "--final_early_stop_patience",
        type=int,
        default=16,
        help="Early-stop patience for final training run.",
    )
    parser.add_argument(
        "--skip_final_train",
        action="store_true",
        help="Only run Optuna; do not train final model with best params.",
    )
    parser.add_argument(
        "--optuna_out_dir",
        default="data/processed/experiments/cca/optuna",
        help="Directory for study JSON and final run artifacts.",
    )
    parser.add_argument("--timeout_per_trial", type=float, default=None, help="Seconds per trial (optional).")
    return parser.parse_args(argv)


def apply_trial_params(trial: optuna.Trial, args: argparse.Namespace) -> None:
    args.num_primitives = trial.suggest_categorical("num_primitives", [15, 30, 50])
    args.query_dim = trial.suggest_categorical("query_dim", [64, 128, 192])
    args.n_cross_attn_layers = trial.suggest_int("n_cross_attn_layers", 1, 2)
    args.n_self_attn_layers = trial.suggest_int("n_self_attn_layers", 1, 2)
    args.n_heads = trial.suggest_categorical("n_heads", [2, 4])
    args.alpha = trial.suggest_categorical("alpha", [0.5, 1.0])
    args.dropout = trial.suggest_float("dropout", 0.05, 0.25)
    args.lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    args.use_gate_M = trial.suggest_categorical("use_gate_M", [True, False])
    args.init_queries_from_text = trial.suggest_categorical("init_queries_from_text", [True, False])


def objective(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
    data,
    device,
) -> float:
    args = copy.deepcopy(base_args)
    apply_trial_params(trial, args)
    args.epochs = base_args.tune_epochs
    args.early_stop_patience = base_args.tune_early_stop_patience
    args.run_id = f"trial_{trial.number:04d}"
    args.out_dir = str(Path(base_args.optuna_out_dir) / "trials" / args.run_id)

    try:
        metrics = train_cca(
            args,
            data,
            device,
            trial=trial,
            save_artifacts=False,
            verbose=False,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise optuna.TrialPruned(f"OOM: {exc}") from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(str(exc)) from exc
        raise

    val_f1 = float(metrics["val_macro_f1@0.5"])
    trial.set_user_attr("test_macro_f1@0.5", float(metrics["test_macro_f1@0.5"]))
    trial.set_user_attr("trainable_params", int(metrics["trainable_params"]))
    trial.set_user_attr("epochs_ran", int(metrics["epochs_ran"]))
    return val_f1


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    optuna_dir = Path(args.optuna_out_dir)
    optuna_dir.mkdir(parents=True, exist_ok=True)

    if args.storage.startswith("sqlite:///"):
        db_path = Path(args.storage.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    print("Loading patch caches (once for all trials)...")
    data = load_cca_data(args, device)
    print({"patch_cache_loaded": True, "train": data.n_train, "val": data.n_val})

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )

    def _objective(trial: optuna.Trial) -> float:
        return objective(trial, args, data, device)

    study.optimize(
        _objective,
        n_trials=args.n_trials,
        timeout=args.timeout_per_trial,
        show_progress_bar=True,
    )

    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "storage": args.storage,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_val_macro_f1@0.5": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }
    write_json(optuna_dir / "best_trial.json", summary)
    print(summary)

    if args.skip_final_train:
        return

    print("Training final model with best hyperparameters...")
    final_args = copy.deepcopy(args)
    for k, v in best.params.items():
        setattr(final_args, k, v)
    final_args.epochs = args.final_epochs
    final_args.early_stop_patience = args.final_early_stop_patience
    final_args.run_id = f"best_optuna_{args.study_name}"
    final_args.out_dir = ""
    final_args.model_id = args.model_id or "cca"
    final_args.protocol = args.protocol or "default"
    set_seed(final_args.seed)

    final_metrics = train_cca(
        final_args,
        data,
        device,
        trial=None,
        save_artifacts=True,
        verbose=True,
    )
    write_json(optuna_dir / "final_metrics.json", final_metrics)
    print({"final_training_done": True, **{k: final_metrics[k] for k in ("val_macro_f1@0.5", "test_macro_f1@0.5")}})


if __name__ == "__main__":
    main()
