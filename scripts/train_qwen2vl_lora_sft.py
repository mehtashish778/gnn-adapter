#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL-2B-Instruct with LoRA (r=16) via generative JSON SFT (CheXpert labels).

Requires: pip install peft transformers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    build_standard_argparser,
    load_rows,
    masked_macro_f1,
    masked_subset_accuracy,
    probabilistic_metrics,
    require_cuda_device,
    set_seed,
    write_json,
)
from model_registry import auto_run_id, resolve_experiment_dir, update_run_registry
from qwen2vl_lora_common import (
    DEFAULT_MODEL_ROOT,
    GpuTimer,
    apply_lora,
    build_sft_batch,
    count_trainable_params,
    ensure_model_snapshot,
    generate_probs_from_rows,
    load_base_qwen_model,
    load_processor,
    peak_gpu_memory_mb,
)


def main():
    parser = build_standard_argparser("Train Qwen2-VL LoRA r=16 with generative JSON SFT.")
    parser.set_defaults(lr=2e-5, epochs=1)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 = all rows (debug with small N).")
    parser.add_argument("--eval_every", type=int, default=500, help="Val loss check every N optimizer steps.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Fail if local HF cache is incomplete (do not fetch from Hub).",
    )
    args = parser.parse_args()

    try:
        from peft import PeftModel  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install peft: pip install peft") from exc

    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)
    timer = GpuTimer()

    train_rows = load_rows(Path(args.train_rows_json))
    val_rows = load_rows(Path(args.val_rows_json))
    test_rows = load_rows(Path(args.test_rows_json))
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    model_dir = ensure_model_snapshot(
        Path(args.model_path),
        allow_download=not args.no_download,
    )
    processor = load_processor(model_dir, local_files_only=True)
    model = load_base_qwen_model(
        model_dir,
        device,
        local_files_only=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = apply_lora(model, rank=args.lora_rank, causal_lm=True)
    model.train()

    image_root = Path(args.image_root)
    run_id = args.run_id or auto_run_id("qwen2vl_lora_r16_sft")
    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or "qwen2vl_lora_r16_sft",
        protocol=args.protocol or "default",
        run_id=run_id,
        default_legacy_out_dir="data/processed/experiments/qwen2vl_lora_r16_sft",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    best_val_loss = float("inf")
    global_step = 0
    accum = 0
    opt.zero_grad()

    for epoch in range(args.epochs):
        pbar = tqdm(range(0, len(train_rows), args.batch_size), desc=f"sft epoch {epoch+1}")
        epoch_loss = 0.0
        n_batches = 0
        for start in pbar:
            batch_rows = train_rows[start : start + args.batch_size]
            batch, _ = build_sft_batch(processor, batch_rows, image_root, device)
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            if not torch.isfinite(loss):
                opt.zero_grad()
                accum = 0
                continue
            loss.backward()
            accum += 1
            epoch_loss += float(loss.item()) * args.grad_accum
            n_batches += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip_norm,
                )
                opt.step()
                opt.zero_grad()
                accum = 0
                global_step += 1
                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    model.eval()
                    vloss = 0.0
                    vn = 0
                    for vs in range(0, min(len(val_rows), 200), args.batch_size):
                        vb = val_rows[vs : vs + args.batch_size]
                        b, _ = build_sft_batch(processor, vb, image_root, device)
                        with torch.no_grad():
                            vloss += float(model(**b).loss.item())
                        vn += 1
                    vloss /= max(vn, 1)
                    print({"step": global_step, "val_loss": vloss})
                    if vloss < best_val_loss:
                        best_val_loss = vloss
                    model.train()
            pbar.set_postfix(loss=epoch_loss / max(n_batches, 1))

        if accum > 0:
            opt.step()
            opt.zero_grad()

        model.eval()
        vloss = 0.0
        vn = 0
        for vs in range(0, min(len(val_rows), 500), args.batch_size):
            vb = val_rows[vs : vs + args.batch_size]
            b, _ = build_sft_batch(processor, vb, image_root, device)
            with torch.no_grad():
                vloss += float(model(**b).loss.item())
            vn += 1
        vloss /= max(vn, 1)
        print({"epoch": epoch + 1, "val_loss": vloss})
        if vloss < best_val_loss:
            best_val_loss = vloss
        # Always keep the latest epoch adapter (avoids early-step "best val loss" checkpoints).
        model.save_pretrained(adapter_dir)

    if not adapter_dir.exists():
        model.save_pretrained(adapter_dir)

    trainable = count_trainable_params(model)

    from peft import PeftModel

    base_reload = load_base_qwen_model(model_dir, device, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base_reload, str(adapter_dir)).to(device)
    model.eval()

    val_prob, val_parse_fail = generate_probs_from_rows(
        model, processor, val_rows, image_root, device, args.batch_size
    )
    test_prob, test_parse_fail = generate_probs_from_rows(
        model, processor, test_rows, image_root, device, args.batch_size
    )
    va_y = torch.tensor([r["y_true"] for r in val_rows], dtype=torch.float32)
    va_m = torch.tensor([r["y_mask"] for r in val_rows], dtype=torch.float32)
    te_y = torch.tensor([r["y_true"] for r in test_rows], dtype=torch.float32)
    te_m = torch.tensor([r["y_mask"] for r in test_rows], dtype=torch.float32)

    val_f1 = masked_macro_f1(val_prob, va_y, va_m, threshold=0.5)
    test_f1 = masked_macro_f1(test_prob, te_y, te_m, threshold=0.5)
    val_pm = probabilistic_metrics(val_prob, va_y, va_m)
    test_pm = probabilistic_metrics(test_prob, te_y, te_m)
    gpu_hours = timer.stop() / 3600.0

    metrics = {
        "variant": "qwen2vl_lora_r16_sft",
        "trainable_params": trainable,
        "lora_rank": args.lora_rank,
        "gpu_hours": gpu_hours,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        "best_val_loss": best_val_loss,
        "val_parse_failures": val_parse_fail,
        "test_parse_failures": test_parse_fail,
        "val_macro_f1@0.5": val_f1,
        "test_macro_f1@0.5": test_f1,
        "val_subset_accuracy@0.5": masked_subset_accuracy(val_prob, va_y, va_m, threshold=0.5),
        "test_subset_accuracy@0.5": masked_subset_accuracy(test_prob, te_y, te_m, threshold=0.5),
        "val_macro_auroc": val_pm["macro_auroc"],
        "val_macro_auprc": val_pm["macro_auprc"],
        "val_macro_ece": val_pm["macro_ece"],
        "val_macro_brier": val_pm["macro_brier"],
        "test_macro_auroc": test_pm["macro_auroc"],
        "test_macro_auprc": test_pm["macro_auprc"],
        "test_macro_ece": test_pm["macro_ece"],
        "test_macro_brier": test_pm["macro_brier"],
        "hparams": vars(args),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "val_predictions.json",
        {"probs": val_prob.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
    )
    write_json(
        out_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )
    write_json(
        out_dir / "run_meta.json",
        {"mode": "gen", "adapter_dir": str(adapter_dir), "parse_failures_test": test_parse_fail},
    )

    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id or "qwen2vl_lora_r16_sft",
            protocol=args.protocol or "default",
            run_dir=out_dir,
            metrics={
                "val_macro_f1@0.5": val_f1,
                "test_macro_f1@0.5": test_f1,
                "test_macro_auroc": test_pm["macro_auroc"],
            },
            hparams=vars(args),
        )

    print(
        {
            "run_dir": str(out_dir),
            "test_macro_f1@0.5": test_f1,
            "test_parse_failures": test_parse_fail,
            "trainable_params": trainable,
        }
    )


if __name__ == "__main__":
    main()
