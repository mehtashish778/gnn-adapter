#!/usr/bin/env python3
"""
Fine-tune CLIP vision tower with LoRA (r in {4,8,16}), then cache patch tokens for CCA.

Requires: pip install peft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import load_rows, require_cuda_device, resolve_dataset_image_path, set_seed, write_json
from feature_cache import FeatureCache, PATCH_CACHE_VERSION, clip_cache_dataset_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--lora_rank", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--embeddings_cache_dir", default="data/processed/embeddings")
    parser.add_argument("--image_root", default="data/raw")
    args = parser.parse_args()

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit("Install peft: pip install peft") from exc

    device = require_cuda_device(args.gpu_id)
    set_seed(42)
    train_rows = load_rows(Path(args.train_rows_json))
    val_rows = load_rows(Path(args.val_rows_json))

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, target_modules=["q_proj", "v_proj"])
    clip_model.vision_model = get_peft_model(clip_model.vision_model, lora_cfg)

    # Proxy task: multi-label BCE on linear probe over pooled vision features
    c = len(train_rows[0]["y_true"])
    probe = torch.nn.Linear(clip_model.vision_model.config.hidden_size, c).to(device)
    opt = torch.optim.AdamW(list(clip_model.vision_model.parameters()) + list(probe.parameters()), lr=args.lr)

    from PIL import Image

    def batch_tensors(rows, start, bs):
        batch = rows[start : start + bs]
        images = []
        ys, ms = [], []
        for r in batch:
            with Image.open(resolve_dataset_image_path(Path(args.image_root), r["path"])) as im:
                images.append(im.convert("RGB"))
            ys.append(r["y_true"])
            ms.append(r["y_mask"])
        inputs = processor(images=images, return_tensors="pt")
        y = torch.tensor(ys, dtype=torch.float32)
        m = torch.tensor(ms, dtype=torch.float32)
        return inputs["pixel_values"].to(device), y.to(device), m.to(device)

    for epoch in range(args.epochs):
        clip_model.vision_model.train()
        for start in range(0, len(train_rows), args.batch_size):
            pv, y, m = batch_tensors(train_rows, start, args.batch_size)
            opt.zero_grad()
            out = clip_model.vision_model(pixel_values=pv).pooler_output
            logits = probe(out)
            raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (raw * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            opt.step()
        print({"lora_epoch": epoch + 1, "train_loss": float(loss.item())})

    clip_model.vision_model.eval()
    cache_ver = f"{PATCH_CACHE_VERSION}_lora_r{args.lora_rank}"
    fc = FeatureCache(args.embeddings_cache_dir)
    encoder_id = args.clip_model.replace("/", "_") + f"_lora_r{args.lora_rank}"

    @torch.no_grad()
    def encode_rows(rows, split_name):
        chunks = []
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"encode {split_name}"):
            pv, _, _ = batch_tensors(rows, start, args.batch_size)
            hidden = clip_model.vision_model(pixel_values=pv).last_hidden_state
            patches = hidden[:, 1:, :].detach().cpu().to(torch.float16)
            chunks.append(patches)
        return torch.cat(chunks, dim=0)

    for split_name, rows in [("train", train_rows), ("val", val_rows)]:

        def _compute(split=split_name, split_rows=rows):
            return encode_rows(split_rows, split)

        tensor = fc.get_or_compute(
            dataset_id=f"{clip_cache_dataset_id(args.protocol)}_{split_name}",
            encoder_id=encoder_id,
            version=cache_ver,
            row_ids=[r["path"] for r in rows],
            compute_fn=_compute,
            storage_dtype="float16",
        )
        print({"cached": split_name, "shape": list(tensor.shape)})

    meta = {
        "clip_model": args.clip_model,
        "lora_rank": args.lora_rank,
        "cache_version": cache_ver,
        "encoder_id": encoder_id,
    }
    write_json(Path(args.embeddings_cache_dir) / f"lora_r{args.lora_rank}_meta.json", meta)
    print(meta)


if __name__ == "__main__":
    main()
