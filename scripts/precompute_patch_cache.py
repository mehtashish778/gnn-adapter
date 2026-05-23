#!/usr/bin/env python3
"""Precompute CLIP ViT patch caches for a row JSON (e.g. NIH cross-site test set)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cca_train_core import extract_patch_tokens, load_split_patches, resolve_patch_cache_config
from common_multilabel import load_rows, require_cuda_device, row_ids
from feature_cache import FeatureCache, clip_cache_dataset_id
from transformers import CLIPModel, CLIPProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_json", default="data/processed/splits/nih/test_rows.json")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--protocol", default="nih")
    parser.add_argument("--split_name", default="test")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--embeddings_cache_dir", default="data/processed/embeddings")
    parser.add_argument("--lora_rank", type=int, default=None, choices=[4, 8, 16])
    parser.add_argument(
        "--lora_adapter_dir",
        default="",
        help="PEFT adapter dir for LoRA vision (required when --lora_rank is set).",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = require_cuda_device(args.gpu_id)
    rows = load_rows(Path(args.rows_json))
    feature_cache = FeatureCache(args.embeddings_cache_dir)

    ns = SimpleNamespace(
        protocol=args.protocol,
        clip_model=args.clip_model,
        clip_batch_size=args.clip_batch_size,
        lora_rank=args.lora_rank,
        patch_encoder_id="",
        patch_cache_version="",
    )
    encoder_id, cache_version = resolve_patch_cache_config(ns)

    clip_model = None
    processor = None
    if args.lora_rank:
        from peft import PeftModel

        if not args.lora_adapter_dir:
            raise ValueError("--lora_adapter_dir required with --lora_rank")
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
        clip_model.vision_model = PeftModel.from_pretrained(
            clip_model.vision_model, args.lora_adapter_dir
        )
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()
    else:
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()

    dataset_id = f"{clip_cache_dataset_id(args.protocol)}_{args.split_name}"
    tensor = load_split_patches(
        rows,
        args.split_name,
        ns,
        Path(args.image_root),
        clip_model,
        processor,
        device,
        None,
        feature_cache,
    )
    print(
        {
            "dataset_id": dataset_id,
            "encoder_id": encoder_id,
            "version": cache_version,
            "shape": list(tensor.shape),
            "n_rows": len(rows),
            "row_hash": row_ids(rows)[:3],
        }
    )


if __name__ == "__main__":
    main()
