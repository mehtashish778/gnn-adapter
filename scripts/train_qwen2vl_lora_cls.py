#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL-2B-Instruct with LoRA (r=16) + linear classification head on CheXpert.

Requires: pip install peft transformers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    build_standard_argparser,
    compute_pos_weight,
    load_rows,
    masked_bce_with_logits,
    masked_macro_f1,
    masked_subset_accuracy,
    probabilistic_metrics,
    require_cuda_device,
    set_seed,
    write_json,
)
from model_registry import auto_run_id, resolve_experiment_dir, update_run_registry
from qwen2vl_lora_common import (
    CLS_PROMPT,
    DEFAULT_MODEL_ROOT,
    GpuTimer,
    apply_lora,
    build_user_messages,
    count_trainable_params,
    ensure_model_snapshot,
    load_base_qwen_model,
    load_processor,
    open_image,
    peak_gpu_memory_mb,
    pool_last_token_hidden,
    prepare_inputs,
)


def _sanitize_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


class Qwen2VLClassifier(nn.Module):
    """fp32 classification head on pooled backbone hidden states (fp32 pool)."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, num_labels, dtype=torch.float32)

    def forward(self, inputs: dict) -> torch.Tensor:
        pooled = pool_last_token_hidden(self.backbone, inputs)
        return self.head(pooled)


def _prepare_batch(
    batch_rows,
    processor,
    device,
    image_root: Path,
    prompt: str,
):
    images = [open_image(image_root, r) for r in batch_rows]
    if len(batch_rows) == 1:
        msgs = build_user_messages(images[0], prompt)
        inputs = prepare_inputs(processor, msgs, images, device)
    else:
        texts = [
            processor.apply_chat_template(
                build_user_messages(img, prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for img in images
        ]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    y = torch.tensor([r["y_true"] for r in batch_rows], dtype=torch.float32, device=device)
    m = torch.tensor([r["y_mask"] for r in batch_rows], dtype=torch.float32, device=device)
    return inputs, y, m


def evaluate_rows(model, rows, processor, device, image_root, prompt, batch_size) -> tuple:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs, _, _ = _prepare_batch(batch, processor, device, image_root, prompt)
            logits = model(inputs)
            probs = _sanitize_probs(torch.sigmoid(logits.float()))
            all_probs.append(probs.cpu())
    probs_t = torch.cat(all_probs, dim=0)
    y_t = torch.tensor([r["y_true"] for r in rows], dtype=torch.float32)
    m_t = torch.tensor([r["y_mask"] for r in rows], dtype=torch.float32)
    f1 = masked_macro_f1(probs_t, y_t, m_t, threshold=0.5)
    return probs_t, y_t, m_t, f1


def _clip_gradients(model: nn.Module, max_norm: float) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def main():
    parser = build_standard_argparser("Train Qwen2-VL LoRA r=16 + classification head.")
    parser.set_defaults(lr=2e-5, epochs=3)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay_lora", type=float, default=0.0)
    parser.add_argument("--weight_decay_head", type=float, default=1e-4)
    parser.add_argument("--pos_weight_max", type=float, default=100.0)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 = all train rows (debug subset).")
    parser.add_argument("--max_val_samples", type=int, default=0, help="0 = full val (cap for smoke tests).")
    parser.add_argument("--max_test_samples", type=int, default=0, help="0 = full test (cap for smoke tests).")
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
    if args.max_val_samples > 0:
        val_rows = val_rows[: args.max_val_samples]
    if args.max_test_samples > 0:
        test_rows = test_rows[: args.max_test_samples]

    model_dir = ensure_model_snapshot(
        Path(args.model_path),
        allow_download=not args.no_download,
    )
    processor = load_processor(model_dir, local_files_only=True)
    backbone = load_base_qwen_model(
        model_dir,
        device,
        local_files_only=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    backbone = apply_lora(backbone, rank=args.lora_rank, causal_lm=False)

    hidden_size = backbone.config.hidden_size
    num_labels = len(train_rows[0]["y_true"])
    model = Qwen2VLClassifier(backbone, hidden_size, num_labels).to(device)

    ytr = torch.tensor([r["y_true"] for r in train_rows], dtype=torch.float32)
    mtr = torch.tensor([r["y_mask"] for r in train_rows], dtype=torch.float32)
    pos_weight = compute_pos_weight(ytr, mtr, max_weight=args.pos_weight_max).to(device)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay_lora},
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay_head},
        ]
    )

    image_root = Path(args.image_root)
    run_id = args.run_id or auto_run_id("qwen2vl_lora_r16")
    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or "qwen2vl_lora_r16",
        protocol=args.protocol or "default",
        run_id=run_id,
        default_legacy_out_dir="data/processed/experiments/qwen2vl_lora_r16",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"

    best_val_f1 = -1.0
    best_head_state = None
    patience_left = args.early_stop_patience
    nan_steps = 0

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        accum_loss = 0.0
        step_in_accum = 0
        n_steps = 0
        pbar = tqdm(range(0, len(train_rows), args.batch_size), desc=f"epoch {epoch+1}")
        for start in pbar:
            batch_rows = train_rows[start : start + args.batch_size]
            inputs, y, m = _prepare_batch(batch_rows, processor, device, image_root, CLS_PROMPT)
            logits = model(inputs)
            loss = masked_bce_with_logits(
                logits.float(), y, m, pos_weight.float()
            ) / args.grad_accum

            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad()
                step_in_accum = 0
                pbar.set_postfix(loss="nan", nan_steps=nan_steps)
                continue

            loss.backward()
            accum_loss += float(loss.item()) * args.grad_accum
            step_in_accum += 1
            n_steps += 1
            if step_in_accum >= args.grad_accum:
                _clip_gradients(model, args.grad_clip_norm)
                opt.step()
                opt.zero_grad()
                step_in_accum = 0

            display_loss = accum_loss / max(n_steps, 1)
            pbar.set_postfix(loss=f"{display_loss:.4f}", nan_steps=nan_steps)

        if step_in_accum > 0:
            _clip_gradients(model, args.grad_clip_norm)
            opt.step()
            opt.zero_grad()

        _, _, _, val_f1 = evaluate_rows(
            model, val_rows, processor, device, image_root, CLS_PROMPT, args.batch_size
        )
        print({"epoch": epoch + 1, "val_macro_f1@0.5": val_f1, "nan_steps": nan_steps})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_head_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
            patience_left = args.early_stop_patience
            model.backbone.save_pretrained(adapter_dir)
            torch.save(best_head_state, out_dir / "classifier_head.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print({"early_stop": epoch + 1})
                break

    if best_head_state is None:
        raise RuntimeError(
            "Training produced no checkpoint (val F1 never improved). "
            f"nan_steps={nan_steps}; try lower --lr or --max_train_samples for a smoke test."
        )

    from peft import PeftModel

    base_reload = load_base_qwen_model(
        model_dir,
        device,
        gradient_checkpointing=False,
    )
    backbone = PeftModel.from_pretrained(base_reload, str(adapter_dir)).to(device)
    model = Qwen2VLClassifier(backbone, hidden_size, num_labels).to(device)
    model.head.load_state_dict(best_head_state)
    model.eval()

    val_prob, va_y, va_m, val_f1 = evaluate_rows(
        model, val_rows, processor, device, image_root, CLS_PROMPT, args.batch_size
    )
    test_prob, te_y, te_m, test_f1 = evaluate_rows(
        model, test_rows, processor, device, image_root, CLS_PROMPT, args.batch_size
    )

    val_pm = probabilistic_metrics(val_prob, va_y, va_m)
    test_pm = probabilistic_metrics(test_prob, te_y, te_m)
    trainable = count_trainable_params(model)
    gpu_hours = timer.stop() / 3600.0

    metrics = {
        "variant": "qwen2vl_lora_r16_cls",
        "trainable_params": trainable,
        "lora_rank": args.lora_rank,
        "gpu_hours": gpu_hours,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        "nan_steps": nan_steps,
        "best_val_macro_f1@0.5": best_val_f1,
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
    write_json(out_dir / "run_meta.json", {"mode": "cls", "adapter_dir": str(adapter_dir)})

    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id or "qwen2vl_lora_r16",
            protocol=args.protocol or "default",
            run_dir=out_dir,
            metrics={
                "val_macro_f1@0.5": val_f1,
                "test_macro_f1@0.5": test_f1,
                "test_macro_auroc": test_pm["macro_auroc"],
            },
            hparams=vars(args),
        )

    print({"run_dir": str(out_dir), "test_macro_f1@0.5": test_f1, "trainable_params": trainable})


if __name__ == "__main__":
    main()
