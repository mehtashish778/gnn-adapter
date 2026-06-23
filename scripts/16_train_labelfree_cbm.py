#!/usr/bin/env python3
"""Label-free CBM: CLIP concept scores + linear label head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    build_standard_argparser,
    clip_image_embeds_tensor,
    clip_text_embeds_tensor,
    load_rows,
    masked_macro_f1,
    require_cuda_device,
    resolve_dataset_image_path,
    set_seed,
    to_label_tensors,
    write_json,
)
from model_registry import resolve_experiment_dir, update_run_registry
from models.architectures.cca import DEFAULT_CONCEPT_PHRASES


@torch.no_grad()
def clip_concept_features(rows, image_root, clip_model, processor, device, phrases, batch_size):
    text_in = processor(text=phrases, return_tensors="pt", padding=True, truncation=True)
    text_in = {k: v.to(device) for k, v in text_in.items()}
    txt = clip_text_embeds_tensor(clip_model, **text_in)
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    chunks = []
    for start in tqdm(range(0, len(rows), batch_size), desc="CLIP concept encode"):
        from PIL import Image

        batch = rows[start : start + batch_size]
        images = []
        for r in batch:
            with Image.open(resolve_dataset_image_path(image_root, r["path"])) as im:
                images.append(im.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        img_emb = clip_image_embeds_tensor(clip_model, pv)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scores = (img_emb @ txt.T).float()
        chunks.append(scores.cpu())
    return torch.cat(chunks, dim=0)


def main():
    parser = build_standard_argparser("Train label-free CBM.")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--num_concepts", type=int, default=30)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--image_root", default="data/raw")
    args = parser.parse_args()
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    tr = load_rows(Path(args.train_rows_json))
    va = load_rows(Path(args.val_rows_json))
    te = load_rows(Path(args.test_rows_json))
    _, _, ytr, mtr = to_label_tensors(tr)
    _, _, yva, mva = to_label_tensors(va)
    _, _, yte, mte = to_label_tensors(te)
    phrases = DEFAULT_CONCEPT_PHRASES[: args.num_concepts]

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    xtr = clip_concept_features(tr, Path(args.image_root), clip_model, processor, device, phrases, args.clip_batch_size)
    xva = clip_concept_features(va, Path(args.image_root), clip_model, processor, device, phrases, args.clip_batch_size)
    xte = clip_concept_features(te, Path(args.image_root), clip_model, processor, device, phrases, args.clip_batch_size)

    head = nn.Linear(args.num_concepts, ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    pos = (ytr * mtr).sum(0)
    neg = ((1 - ytr) * mtr).sum(0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).to(device)
    loader = DataLoader(TensorDataset(xtr, ytr, mtr), batch_size=64, shuffle=True)

    for _ in range(args.epochs):
        head.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            logits = head(xb)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight, reduction="none"
            )
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            loss.backward()
            opt.step()

    head.eval()
    from common_multilabel import probabilistic_metrics

    with torch.no_grad():
        va_prob = torch.sigmoid(head(xva.to(device)))
        te_prob = torch.sigmoid(head(xte.to(device)))
        va_f1 = masked_macro_f1(va_prob, yva.to(device), mva.to(device))
        te_f1 = masked_macro_f1(te_prob, yte.to(device), mte.to(device))
        va_pm = probabilistic_metrics(va_prob, yva, mva)
        te_pm = probabilistic_metrics(te_prob, yte, mte)

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or "cbm_labelfree",
        protocol=args.protocol or "default",
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/cbm_labelfree",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "variant": "cbm_labelfree",
        "trainable_params": sum(p.numel() for p in head.parameters() if p.requires_grad),
        "val_macro_f1@0.5": float(va_f1),
        "test_macro_f1@0.5": float(te_f1),
        "val_macro_auroc": va_pm["macro_auroc"],
        "test_macro_auroc": te_pm["macro_auroc"],
        "val_macro_auprc": va_pm["macro_auprc"],
        "test_macro_auprc": te_pm["macro_auprc"],
        "val_macro_ece": va_pm["macro_ece"],
        "test_macro_ece": te_pm["macro_ece"],
        "val_macro_brier": va_pm["macro_brier"],
        "test_macro_brier": te_pm["macro_brier"],
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "test_predictions.json",
        {"probs": te_prob.cpu().tolist(), "y_true": yte.tolist(), "y_mask": mte.tolist()},
    )
    write_json(
        out_dir / "val_predictions.json",
        {"probs": va_prob.cpu().tolist(), "y_true": yva.tolist(), "y_mask": mva.tolist()},
    )
    torch.save(head.state_dict(), out_dir / "best_checkpoint.pt")
    update_run_registry(
        model_id=args.model_id or "cbm_labelfree",
        protocol=args.protocol or "default",
        run_dir=out_dir,
        metrics={"val_macro_f1@0.5": float(va_f1), "test_macro_f1@0.5": float(te_f1)},
        hparams={"epochs": args.epochs, "lr": args.lr, "num_concepts": args.num_concepts},
    )
    print(metrics)


if __name__ == "__main__":
    main()
