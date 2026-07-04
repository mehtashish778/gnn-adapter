#!/usr/bin/env python3
"""Train VLM MLP baseline via unified training engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from common_multilabel import build_standard_argparser, load_rows, to_vlm_training_batch
from model_registry import resolve_experiment_dir
from models.architectures.vlm_mlp import VLMFeatureMLP
from training_engine import TrainingConfig, run_training_loop


def main():
    parser = build_standard_argparser("Run simple MLP residual baseline.")
    args = parser.parse_args()

    tr = load_rows(Path(args.train_rows_json))
    va = load_rows(Path(args.val_rows_json))
    te = load_rows(Path(args.test_rows_json))
    ca = load_rows(Path(args.calib_rows_json)) if args.calib_rows_json else None

    train_batch = to_vlm_training_batch(tr)
    val_batch = to_vlm_training_batch(va)
    test_batch = to_vlm_training_batch(te)
    calib_batch = to_vlm_training_batch(ca) if ca is not None else None

    c = train_batch[1].shape[1]
    d = train_batch[0].shape[1]
    model = VLMFeatureMLP(input_dim=d, num_labels=c)

    if args.resume_from:
        import torch

        state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state)

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or None,
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/baseline_mlp",
    )

    config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        gpu_id=args.gpu_id,
        seed=args.seed,
        run_dir=out_dir,
    )

    metrics = run_training_loop(
        model,
        train_batch=train_batch,
        val_batch=val_batch,
        test_batch=test_batch,
        calib_batch=calib_batch,
        forward_fn=lambda m, x: m(x),
        config=config,
        model_id=args.model_id or "",
        protocol=args.protocol or "",
        hparams={"epochs": args.epochs, "lr": args.lr, "seed": args.seed},
    )
    print(metrics)


if __name__ == "__main__":
    main()
