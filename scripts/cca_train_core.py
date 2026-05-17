#!/usr/bin/env python3
"""
Train Concept-Evidence Adapter (CCA): frozen ViT patch tokens + frozen VLM (z, p).

Layer 1: concept queries cross-attend over patch tokens.
Layer 2: self-attention compositional reasoning (optional RadGraph bias).
Layer 3: findings readout with VLM residual gating.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import (
    load_per_class_thresholds,
    load_rows,
    masked_bce_with_logits,
    masked_macro_f1,
    masked_subset_accuracy,
    require_cuda_device,
    resolve_dataset_image_path,
    row_ids,
    set_seed,
    to_label_tensors,
    write_json,
)
from feature_cache import FeatureCache, PATCH_CACHE_VERSION, atomic_torch_save, clip_cache_dataset_id
from model_registry import resolve_experiment_dir, update_run_registry
from faithfulness_metrics import (
    gate_density,
    intervention_consistency,
    intervention_faithfulness_loss,
    necessity_sufficiency_scores,
    sparsity_target_loss,
)
from models.architectures.cca import CCAModel, DEFAULT_CONCEPT_PHRASES


class PatchRowDataset(Dataset):
    def __init__(
        self,
        patch_tokens: torch.Tensor,
        logits: torch.Tensor,
        probs: torch.Tensor,
        y_true: torch.Tensor,
        y_mask: torch.Tensor,
    ):
        self.patch_tokens = patch_tokens
        self.logits = logits
        self.probs = probs
        self.y_true = y_true
        self.y_mask = y_mask

    def __len__(self) -> int:
        return self.patch_tokens.shape[0]

    def __getitem__(self, i: int):
        return (
            self.patch_tokens[i],
            self.logits[i],
            self.probs[i],
            self.y_true[i],
            self.y_mask[i],
        )


@torch.no_grad()
def extract_patch_tokens(
    rows: List[dict],
    image_root: Path,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return (N, num_patches, patch_dim) from frozen CLIP vision tower."""
    vision = clip_model.vision_model
    vision.eval()
    chunks = []
    for start in tqdm(range(0, len(rows), batch_size), desc="ViT patch encode"):
        batch_rows = rows[start : start + batch_size]
        images = []
        for r in batch_rows:
            p = resolve_dataset_image_path(image_root, r["path"])
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=vision.dtype if hasattr(vision, "dtype") else torch.float32)
        out = vision(pixel_values=pv)
        hidden = out.last_hidden_state
        patch_feats = hidden[:, 1:, :].detach().cpu().to(torch.float16)
        chunks.append(patch_feats)
    return torch.cat(chunks, dim=0)


def load_radgraph_prior(path: str, num_primitives: int, device: torch.device) -> Optional[torch.Tensor]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: radgraph_prior_json not found: {path}")
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mat = data.get("matrix") or data.get("prior") or data
    prior = torch.tensor(mat, dtype=torch.float32, device=device)
    if prior.shape[0] != num_primitives or prior.shape[1] != num_primitives:
        print(
            f"Warning: RadGraph prior shape {tuple(prior.shape)} != ({num_primitives},{num_primitives}); ignoring."
        )
        return None
    return prior


@torch.no_grad()
def init_concept_queries_from_text(
    model: CCAModel,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    phrases: List[str],
    device: torch.device,
) -> None:
    n_use = min(model.num_primitives, len(phrases))
    phrases = phrases[:n_use]
    inputs = processor(text=phrases, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_feats = clip_model.get_text_features(**inputs).float()
    text_dim = text_feats.shape[-1]
    if text_dim != model.query_dim:
        proj = nn.Linear(text_dim, model.query_dim).to(device)
        text_feats = proj(text_feats)
    model.init_concept_queries_from_text(text_feats)


def maybe_load_patch_cache(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def save_patch_cache(
    path: Path,
    clip_model_name: str,
    train_paths: List[str],
    train_patch: torch.Tensor,
    val_paths: List[str],
    val_patch: torch.Tensor,
    test_paths: List[str],
    test_patch: torch.Tensor,
    calib_paths: Optional[List[str]] = None,
    calib_patch: Optional[torch.Tensor] = None,
) -> None:
    payload = {
        "clip_model_name": clip_model_name,
        "train_paths": train_paths,
        "train_patch": train_patch,
        "val_paths": val_paths,
        "val_patch": val_patch,
        "test_paths": test_paths,
        "test_patch": test_patch,
        "calib_paths": calib_paths,
        "calib_patch": calib_patch,
    }
    for k in ("train_patch", "val_patch", "test_patch", "calib_patch"):
        t = payload.get(k)
        if t is not None:
            payload[k] = t.detach().cpu().to(torch.float16)
    atomic_torch_save(payload, path)


def verify_paths_order(stored: List[str], rows: List[dict]) -> None:
    for a, b in zip(stored, rows):
        if a != b["path"]:
            raise ValueError("Patch cache path order mismatch vs rows JSON; delete cache and re-encode.")


def load_split_patches(
    rows: List[dict],
    split_name: str,
    args: argparse.Namespace,
    image_root: Path,
    clip_model: Optional[CLIPModel],
    processor: Optional[CLIPProcessor],
    device: torch.device,
    cache: Optional[Dict[str, Any]],
    feature_cache: FeatureCache,
) -> torch.Tensor:
    """Load or compute patch tokens for one split."""
    if cache is not None:
        key = f"{split_name}_patch"
        if key in cache:
            return cache[key].float()

    protocol = args.protocol or "default"
    dataset_id = f"{clip_cache_dataset_id(protocol)}_{split_name}"
    encoder_id = args.clip_model.replace("/", "_")

    def compute():
        if clip_model is None or processor is None:
            raise RuntimeError("CLIP model required to compute patch tokens.")
        return extract_patch_tokens(rows, image_root, clip_model, processor, device, args.clip_batch_size)

    return feature_cache.get_or_compute(
        dataset_id=dataset_id,
        encoder_id=encoder_id,
        version=PATCH_CACHE_VERSION,
        row_ids=row_ids(rows),
        compute_fn=compute,
        storage_dtype="float16",
    ).float()


@dataclass
class CCADataBundle:
    train_rows: List[dict]
    val_rows: List[dict]
    test_rows: List[dict]
    calib_rows: Optional[List[dict]]
    n_train: int
    n_val: int
    n_test: int
    n_calib: int
    c: int
    tr_patch: torch.Tensor
    va_patch: torch.Tensor
    te_patch: torch.Tensor
    ca_patch: Optional[torch.Tensor]
    tr_logits: torch.Tensor
    tr_probs: torch.Tensor
    tr_y: torch.Tensor
    tr_m: torch.Tensor
    va_logits: torch.Tensor
    va_probs: torch.Tensor
    va_y: torch.Tensor
    va_m: torch.Tensor
    te_logits: torch.Tensor
    te_probs: torch.Tensor
    te_y: torch.Tensor
    te_m: torch.Tensor
    ca_logits: Optional[torch.Tensor]
    ca_probs: Optional[torch.Tensor]
    ca_y: Optional[torch.Tensor]
    ca_m: Optional[torch.Tensor]
    thr_list: Optional[List[float]]
    patch_dim: int


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Concept-Evidence Adapter (CCA).")
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--calib_rows_json", default=None, help="Optional calibration rows JSON.")
    parser.add_argument("--per_class_thresholds_json", default="data/processed/experiments/thresholds/per_class_thresholds.json")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--clip_cache_pt", default="", help="Legacy combined patch cache .pt")
    parser.add_argument(
        "--embeddings_cache_dir",
        default="data/processed/embeddings",
        help="Directory for per-split patch caches (use a drive with ~20GB+ free).",
    )
    parser.add_argument("--num_primitives", type=int, default=30)
    parser.add_argument("--query_dim", type=int, default=128)
    parser.add_argument("--patch_dim", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_cross_attn_layers", type=int, default=2)
    parser.add_argument("--n_self_attn_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init_queries_from_text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_gate_M", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_sparse", type=float, default=0.0, help="Weight on sparsity target loss for gate M.")
    parser.add_argument("--lambda_faithful", type=float, default=0.0, help="Weight on intervention faithfulness loss.")
    parser.add_argument("--sparsity_target", type=float, default=0.10, help="Target gate density (5-15% band center).")
    parser.add_argument("--gumbel_tau_init", type=float, default=1.0)
    parser.add_argument("--gumbel_tau_min", type=float, default=0.5)
    parser.add_argument("--gumbel_anneal_epochs", type=int, default=10)
    parser.add_argument("--intervention_per_step", type=int, default=1, help="Interventions per batch (0=off).")
    parser.add_argument("--radgraph_prior_json", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--pos_weight_max", type=float, default=100.0)
    parser.add_argument("--lr_scheduler", choices=("none", "cosine", "plateau"), default="cosine")
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=6)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--best_metric", choices=("val_bce", "val_macro_f1_thr", "val_macro_f1_05"), default="val_bce")
    parser.add_argument("--early_stop_patience", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_id", default="cca")
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser


def gumbel_tau_at_epoch(args: argparse.Namespace, epoch: int) -> float:
    w = max(1, args.gumbel_anneal_epochs)
    t = min(1.0, epoch / w)
    return args.gumbel_tau_init + t * (args.gumbel_tau_min - args.gumbel_tau_init)


def load_cca_data(args: argparse.Namespace, device: torch.device) -> CCADataBundle:
    train_rows = load_rows(Path(args.train_rows_json))
    val_rows = load_rows(Path(args.val_rows_json))
    test_rows = load_rows(Path(args.test_rows_json))
    calib_rows = load_rows(Path(args.calib_rows_json)) if args.calib_rows_json else None
    n_train, n_val, n_test = len(train_rows), len(val_rows), len(test_rows)
    n_calib = len(calib_rows) if calib_rows is not None else 0
    print({"dataset_sizes": {"train": n_train, "val": n_val, "calib": n_calib, "test": n_test}, "variant": "cca"})

    c = len(train_rows[0]["x_probs"])
    tr_logits, tr_probs, tr_y, tr_m = to_label_tensors(train_rows)
    va_logits, va_probs, va_y, va_m = to_label_tensors(val_rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(test_rows)
    if calib_rows is not None:
        ca_logits, ca_probs, ca_y, ca_m = to_label_tensors(calib_rows)
    else:
        ca_logits = ca_probs = ca_y = ca_m = None

    thr_path = Path(args.per_class_thresholds_json)
    thr_list = load_per_class_thresholds(thr_path)
    if thr_list is not None and len(thr_list) != c:
        thr_list = None

    image_root = Path(args.image_root)
    cache_path = Path(args.clip_cache_pt) if args.clip_cache_pt else None
    legacy_cache = maybe_load_patch_cache(cache_path)
    feature_cache = FeatureCache(args.embeddings_cache_dir)

    clip_model: Optional[CLIPModel] = None
    processor: Optional[CLIPProcessor] = None

    if legacy_cache is not None and legacy_cache.get("clip_model_name") != args.clip_model:
        print("Patch cache model mismatch; re-encoding.")
        legacy_cache = None

    need_encode = legacy_cache is None
    if legacy_cache is None:
        # Check if all FeatureCache splits exist
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            p = feature_cache.cache_path(
                f"{clip_cache_dataset_id(args.protocol)}_{split_name}",
                args.clip_model.replace("/", "_"),
                PATCH_CACHE_VERSION,
            )
            if not p.exists():
                need_encode = True
                break

    if need_encode:
        n_total = len(train_rows) + len(val_rows) + len(test_rows) + (len(calib_rows) if calib_rows else 0)
        est_gb = n_total * 196 * 768 * 2 / (1024**3)
        print(
            f"Patch cache (fp16) estimate: ~{est_gb:.1f} GB under {feature_cache.cache_dir}. "
            "Use --embeddings_cache_dir if C: is low on space."
        )
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()

    if legacy_cache is not None:
        verify_paths_order(legacy_cache["train_paths"], train_rows)
        verify_paths_order(legacy_cache["val_paths"], val_rows)
        verify_paths_order(legacy_cache["test_paths"], test_rows)
        tr_patch = legacy_cache["train_patch"].float()
        va_patch = legacy_cache["val_patch"].float()
        te_patch = legacy_cache["test_patch"].float()
        ca_patch = None
        if calib_rows is not None:
            if legacy_cache.get("calib_patch") is not None and legacy_cache.get("calib_paths"):
                verify_paths_order(legacy_cache["calib_paths"], calib_rows)
                ca_patch = legacy_cache["calib_patch"].float()
            elif clip_model is not None:
                ca_patch = extract_patch_tokens(
                    calib_rows, image_root, clip_model, processor, device, args.clip_batch_size
                )
    else:
        tr_patch = load_split_patches(
            train_rows, "train", args, image_root, clip_model, processor, device, None, feature_cache
        )
        va_patch = load_split_patches(
            val_rows, "val", args, image_root, clip_model, processor, device, None, feature_cache
        )
        te_patch = load_split_patches(
            test_rows, "test", args, image_root, clip_model, processor, device, None, feature_cache
        )
        ca_patch = None
        if calib_rows is not None:
            ca_patch = load_split_patches(
                calib_rows, "calib", args, image_root, clip_model, processor, device, None, feature_cache
            )
        if cache_path and clip_model is not None:
            save_patch_cache(
                cache_path,
                args.clip_model,
                [r["path"] for r in train_rows],
                tr_patch,
                [r["path"] for r in val_rows],
                va_patch,
                [r["path"] for r in test_rows],
                te_patch,
                calib_paths=[r["path"] for r in calib_rows] if calib_rows is not None else None,
                calib_patch=ca_patch,
            )
            print({"saved_patch_cache": str(cache_path)})

    if clip_model is not None:
        del clip_model
        del processor
        torch.cuda.empty_cache()
        clip_model = None
        processor = None

    patch_dim = tr_patch.shape[-1]
    if patch_dim != args.patch_dim:
        print(f"Note: detected patch_dim={patch_dim} (CLI default was {args.patch_dim})")
        args.patch_dim = patch_dim

    return CCADataBundle(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        calib_rows=calib_rows,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_calib=n_calib,
        c=c,
        tr_patch=tr_patch,
        va_patch=va_patch,
        te_patch=te_patch,
        ca_patch=ca_patch,
        tr_logits=tr_logits,
        tr_probs=tr_probs,
        tr_y=tr_y,
        tr_m=tr_m,
        va_logits=va_logits,
        va_probs=va_probs,
        va_y=va_y,
        va_m=va_m,
        te_logits=te_logits,
        te_probs=te_probs,
        te_y=te_y,
        te_m=te_m,
        ca_logits=ca_logits,
        ca_probs=ca_probs,
        ca_y=ca_y,
        ca_m=ca_m,
        thr_list=thr_list,
        patch_dim=patch_dim,
    )


def train_cca(
    args: argparse.Namespace,
    data: CCADataBundle,
    device: torch.device,
    *,
    trial: Optional["optuna.Trial"] = None,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    n_train, n_val, n_test, n_calib = data.n_train, data.n_val, data.n_test, data.n_calib
    c = data.c
    calib_rows = data.calib_rows
    tr_patch, va_patch, te_patch, ca_patch = data.tr_patch, data.va_patch, data.te_patch, data.ca_patch
    tr_logits, tr_probs, tr_y, tr_m = data.tr_logits, data.tr_probs, data.tr_y, data.tr_m
    va_logits, va_probs, va_y, va_m = data.va_logits, data.va_probs, data.va_y, data.va_m
    te_logits, te_probs, te_y, te_m = data.te_logits, data.te_probs, data.te_y, data.te_m
    ca_logits, ca_probs, ca_y, ca_m = data.ca_logits, data.ca_probs, data.ca_y, data.ca_m
    thr_list = data.thr_list
    patch_dim = data.patch_dim

    best_metric = args.best_metric
    if best_metric == "val_macro_f1_thr" and thr_list is None:
        if verbose:
            print("Warning: val_macro_f1_thr not available; using val_bce.")
        best_metric = "val_bce"

    radgraph_prior = load_radgraph_prior(args.radgraph_prior_json, args.num_primitives, device)

    model = CCAModel(
        patch_dim=patch_dim,
        query_dim=args.query_dim,
        num_primitives=args.num_primitives,
        num_findings=c,
        n_heads=args.n_heads,
        n_cross_attn_layers=args.n_cross_attn_layers,
        n_self_attn_layers=args.n_self_attn_layers,
        alpha=args.alpha,
        dropout=args.dropout,
        use_gate_M=args.use_gate_M,
        gumbel_tau=args.gumbel_tau_init,
    ).to(device)

    if radgraph_prior is not None:
        model.layer2.set_radgraph_prior(radgraph_prior)

    if args.init_queries_from_text:
        proc = CLIPProcessor.from_pretrained(args.clip_model)
        clip_for_text = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
        for p in clip_for_text.parameters():
            p.requires_grad = False
        init_concept_queries_from_text(model, clip_for_text, proc, DEFAULT_CONCEPT_PHRASES, device)
        del clip_for_text, proc
        torch.cuda.empty_cache()

    n_params = model.count_trainable_params()
    if verbose:
        print({"trainable_params": n_params})
    if n_params >= 1_000_000:
        if trial is not None:
            import optuna

            raise optuna.TrialPruned(f"param_count {n_params} >= 1_000_000")
        raise RuntimeError(f"Param count {n_params} exceeds 1M limit")

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        state = ckpt.get("adapter_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    plateau_sched = None
    if args.lr_scheduler == "plateau":
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr
        )

    pos = (tr_y * tr_m).sum(dim=0)
    neg = ((1 - tr_y) * tr_m).sum(dim=0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=args.pos_weight_max).to(device)

    va_y_d = va_y.to(device)
    va_m_d = va_m.to(device)
    te_y_d = te_y.to(device)
    te_m_d = te_m.to(device)
    if calib_rows is not None:
        ca_y_d = ca_y.to(device)
        ca_m_d = ca_m.to(device)

    train_loader = DataLoader(
        PatchRowDataset(tr_patch, tr_logits, tr_probs, tr_y, tr_m),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    base_lr = args.lr

    def lr_at_epoch(epoch: int) -> float:
        if args.lr_scheduler != "cosine":
            return base_lr
        w = max(0, args.warmup_epochs)
        if w > 0 and epoch <= w:
            return base_lr * (epoch / w)
        t = (epoch - w) / max(1, args.epochs - w)
        return args.min_lr + (base_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    best: Dict[str, Any] = {"score": None, "state_dict": None}
    history: List[dict] = []
    epochs_no_improve = 0

    def is_better(metric_name, val_bce, f1_05, f1_thr):
        if metric_name == "val_bce":
            if best["score"] is None:
                return True
            return val_bce < best["score"]
        if metric_name == "val_macro_f1_05":
            if best["score"] is None:
                return True
            return f1_05 > best["score"]
        if best["score"] is None:
            return True
        return f1_thr > best["score"]

    for epoch in range(1, args.epochs + 1):
        if args.lr_scheduler == "cosine":
            lr_now = lr_at_epoch(epoch)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

        model.train()
        tau = gumbel_tau_at_epoch(args, epoch)
        model.gumbel_tau = tau
        epoch_losses = []
        epoch_bce = []
        epoch_sparse = []
        epoch_faith = []
        epoch_density = []
        for patches, ll, pp, yt, ym in train_loader:
            patches = patches.to(device, non_blocking=True)
            ll = ll.to(device, non_blocking=True)
            pp = pp.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
            ym = ym.to(device, non_blocking=True)
            opt.zero_grad()
            out, _attn, gate_aux = model(patches, ll, pp, radgraph_prior=radgraph_prior, gumbel_tau=tau)
            loss_bce = masked_bce_with_logits(out, yt, ym, pos_weight)
            loss = loss_bce
            if args.lambda_sparse > 0 and gate_aux.get("M_tilde") is not None:
                l_sparse = sparsity_target_loss(gate_aux["M_tilde"], target=args.sparsity_target)
                loss = loss + args.lambda_sparse * l_sparse
                epoch_sparse.append(float(l_sparse.item()))
                epoch_density.append(float(gate_density(gate_aux["M_tilde"]).item()))
            if args.lambda_faithful > 0 and args.intervention_per_step > 0 and model.use_gate_M:
                p_idx = torch.randint(0, model.num_primitives, (patches.shape[0],), device=device)
                _, out_int, gate_aux_i = model.forward_with_intervention(
                    patches, ll, pp, p_idx, radgraph_prior=radgraph_prior, gumbel_tau=tau
                )
                m_tilde = gate_aux_i.get("M_tilde", gate_aux.get("M_tilde"))
                if m_tilde is not None:
                    l_faith = intervention_faithfulness_loss(out, out_int, m_tilde, p_idx)
                    loss = loss + args.lambda_faithful * l_faith
                    epoch_faith.append(float(l_faith.item()))
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
            epoch_losses.append(float(loss.item()))
            epoch_bce.append(float(loss_bce.item()))

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        train_bce = sum(epoch_bce) / max(1, len(epoch_bce))
        train_sparse = sum(epoch_sparse) / max(1, len(epoch_sparse)) if epoch_sparse else 0.0
        train_faith = sum(epoch_faith) / max(1, len(epoch_faith)) if epoch_faith else 0.0
        train_gate_density = sum(epoch_density) / max(1, len(epoch_density)) if epoch_density else 0.0

        model.eval()
        val_out_parts = []
        with torch.no_grad():
            nv = va_patch.shape[0]
            for start in range(0, nv, args.batch_size):
                patches = va_patch[start : start + args.batch_size].to(device, non_blocking=True)
                ll = va_logits[start : start + args.batch_size].to(device, non_blocking=True)
                pp = va_probs[start : start + args.batch_size].to(device, non_blocking=True)
                out, _, _ = model(patches, ll, pp, radgraph_prior=radgraph_prior, gumbel_tau=args.gumbel_tau_min)
                val_out_parts.append(out)
            val_out = torch.cat(val_out_parts, dim=0)
            val_prob = torch.sigmoid(val_out)
            val_bce = float(masked_bce_with_logits(val_out, va_y_d, va_m_d, pos_weight).item())
            val_f1_05 = masked_macro_f1(val_prob, va_y_d, va_m_d, threshold=0.5)
            val_f1_thr = (
                masked_macro_f1(val_prob, va_y_d, va_m_d, threshold=thr_list) if thr_list is not None else val_f1_05
            )

        history.append(
            {
                "epoch": epoch,
                "lr": float(opt.param_groups[0]["lr"]),
                "gumbel_tau": tau,
                "train_loss": train_loss,
                "train_l_bce": train_bce,
                "train_l_sparse": train_sparse,
                "train_l_faithful": train_faith,
                "gate_density": train_gate_density,
                "val_bce": val_bce,
                "val_macro_f1@0.5": val_f1_05,
                "val_macro_f1@thr": val_f1_thr,
            }
        )
        if plateau_sched is not None:
            plateau_sched.step(val_bce)

        if is_better(best_metric, val_bce, val_f1_05, val_f1_thr):
            if best_metric == "val_bce":
                best["score"] = val_bce
            elif best_metric == "val_macro_f1_05":
                best["score"] = val_f1_05
            else:
                best["score"] = val_f1_thr
            best["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if trial is not None:
            trial.report(float(val_f1_05), epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

        if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
            if verbose:
                print({"early_stop": True, "epoch": epoch})
            break

    if best["state_dict"] is None:
        raise RuntimeError("No checkpoint saved.")

    model.load_state_dict(best["state_dict"])
    model.eval()

    def logits_for(t_patch: torch.Tensor, t_logits: torch.Tensor, t_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = []
        attn_parts = []
        with torch.no_grad():
            n = t_patch.shape[0]
            for start in range(0, n, args.batch_size):
                patches = t_patch[start : start + args.batch_size].to(device, non_blocking=True)
                ll = t_logits[start : start + args.batch_size].to(device, non_blocking=True)
                pp = t_probs[start : start + args.batch_size].to(device, non_blocking=True)
                out, attn, _ = model(patches, ll, pp, radgraph_prior=radgraph_prior, gumbel_tau=args.gumbel_tau_min)
                parts.append(out)
                attn_parts.append(attn.cpu())
        return torch.cat(parts, dim=0), torch.cat(attn_parts, dim=0)

    val_out, val_attn = logits_for(va_patch, va_logits, va_probs)
    test_out, test_attn = logits_for(te_patch, te_logits, te_probs)
    val_prob = torch.sigmoid(val_out).cpu()
    test_prob = torch.sigmoid(test_out).cpu()

    val_f1 = masked_macro_f1(val_prob.to(device), va_y_d, va_m_d, threshold=0.5)
    test_f1 = masked_macro_f1(test_prob.to(device), te_y_d, te_m_d, threshold=0.5)
    calib_prob = None
    calib_f1 = None
    calib_f1_thr_eval = None
    if calib_rows is not None:
        calib_out, _ = logits_for(ca_patch, ca_logits, ca_probs)
        calib_prob = torch.sigmoid(calib_out).cpu()
        calib_f1 = masked_macro_f1(calib_prob.to(device), ca_y_d, ca_m_d, threshold=0.5)

    val_f1_thr_eval = (
        masked_macro_f1(val_prob.to(device), va_y_d, va_m_d, threshold=thr_list) if thr_list is not None else val_f1
    )
    test_f1_thr_eval = (
        masked_macro_f1(test_prob.to(device), te_y_d, te_m_d, threshold=thr_list) if thr_list is not None else test_f1
    )
    if calib_rows is not None:
        calib_f1_thr_eval = (
            masked_macro_f1(calib_prob.to(device), ca_y_d, ca_m_d, threshold=thr_list)
            if thr_list is not None
            else calib_f1
        )

    val_sub = masked_subset_accuracy(val_prob.to(device), va_y_d, va_m_d, threshold=0.5)
    test_sub = masked_subset_accuracy(test_prob.to(device), te_y_d, te_m_d, threshold=0.5)
    calib_sub = calib_sub_thr = None
    if calib_rows is not None:
        calib_sub = masked_subset_accuracy(calib_prob.to(device), ca_y_d, ca_m_d, threshold=0.5)
        calib_sub_thr = (
            masked_subset_accuracy(calib_prob.to(device), ca_y_d, ca_m_d, threshold=thr_list)
            if thr_list is not None
            else calib_sub
        )
    val_sub_thr = masked_subset_accuracy(val_prob.to(device), va_y_d, va_m_d, threshold=thr_list) if thr_list else val_sub
    test_sub_thr = (
        masked_subset_accuracy(test_prob.to(device), te_y_d, te_m_d, threshold=thr_list) if thr_list else test_sub
    )

    metrics_out: Dict[str, Any] = {
        "variant": "cca",
        "trainable_params": n_params,
        "dataset_sizes": {"train": n_train, "val": n_val, "calib": n_calib, "test": n_test},
        "clip_model": args.clip_model,
        "best_metric": best_metric,
        "best_score": best["score"],
        "seed": args.seed,
        "hparams": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "val_macro_f1@0.5": val_f1,
        "test_macro_f1@0.5": test_f1,
        "val_macro_f1@per_class_thr": val_f1_thr_eval,
        "test_macro_f1@per_class_thr": test_f1_thr_eval,
        "val_subset_accuracy@0.5": val_sub,
        "test_subset_accuracy@0.5": test_sub,
        "val_subset_accuracy@per_class_thr": val_sub_thr,
        "test_subset_accuracy@per_class_thr": test_sub_thr,
        "calib_macro_f1@0.5": calib_f1,
        "calib_macro_f1@per_class_thr": calib_f1_thr_eval,
        "calib_subset_accuracy@0.5": calib_sub,
        "calib_subset_accuracy@per_class_thr": calib_sub_thr,
        "epochs_ran": len(history),
    }

    if model.use_gate_M and model.gate is not None:
        m_hard = model.gate.hard_gate()
        metrics_out["gate_density_eval"] = float(gate_density(m_hard).item())
        n_faith = min(256, va_patch.shape[0])
        vp = va_patch[:n_faith].to(device)
        vl = va_logits[:n_faith].to(device)
        vpp = va_probs[:n_faith].to(device)
        vy = va_y[:n_faith].to(device)
        vm = va_m[:n_faith].to(device)
        faith_extra = necessity_sufficiency_scores(
            model, vp, vl, vpp, vy, vm, m_hard, radgraph_prior=radgraph_prior
        )
        metrics_out.update({f"faithfulness_{k}": v for k, v in faith_extra.items()})
        with torch.no_grad():
            p_idx = torch.randint(0, model.num_primitives, (n_faith,), device=device)
            y0, y1, _ = model.forward_with_intervention(vp, vl, vpp, p_idx, radgraph_prior=radgraph_prior)
            metrics_out["intervention_consistency"] = intervention_consistency(y0, y1, m_hard, p_idx)

    if not save_artifacts:
        if verbose:
            print(
                {
                    "variant": "cca",
                    "trainable_params": n_params,
                    "best_metric": best_metric,
                    "best_score": best["score"],
                    "val_macro_f1@0.5": val_f1,
                    "test_macro_f1@0.5": test_f1,
                }
            )
        return metrics_out

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or None,
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/cca",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": best["state_dict"],
            "adapter_hparams": {
                "patch_dim": patch_dim,
                "query_dim": args.query_dim,
                "num_primitives": args.num_primitives,
                "num_labels": c,
                "n_heads": args.n_heads,
                "n_cross_attn_layers": args.n_cross_attn_layers,
                "n_self_attn_layers": args.n_self_attn_layers,
                "alpha": args.alpha,
                "use_gate_M": args.use_gate_M,
                "lambda_sparse": args.lambda_sparse,
                "lambda_faithful": args.lambda_faithful,
            },
            "trainable_params": n_params,
        },
        out_dir / "best_checkpoint.pt",
    )
    write_json(out_dir / "metrics.json", metrics_out)
    write_json(out_dir / "history.json", history)
    write_json(
        out_dir / "val_predictions.json",
        {"probs": val_prob.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
    )
    write_json(
        out_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )
    if calib_rows is not None:
        write_json(
            out_dir / "calib_predictions.json",
            {"probs": calib_prob.tolist(), "y_true": ca_y.tolist(), "y_mask": ca_m.tolist()},
        )
    torch.save({"val": val_attn, "test": test_attn}, out_dir / "attention_maps.pt")

    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id,
            protocol=args.protocol,
            run_dir=out_dir,
            metrics={
                "val_macro_f1@0.5": val_f1,
                "test_macro_f1@0.5": test_f1,
                "val_macro_f1@per_class_thr": val_f1_thr_eval,
                "test_macro_f1@per_class_thr": test_f1_thr_eval,
                "val_subset_accuracy@0.5": val_sub,
                "test_subset_accuracy@0.5": test_sub,
            },
            hparams={"epochs": len(history), "lr": args.lr, "num_primitives": args.num_primitives},
        )
    if verbose:
        print(
            {
                "variant": "cca",
                "trainable_params": n_params,
                "best_metric": best_metric,
                "best_score": best["score"],
                "val_macro_f1@0.5": val_f1,
                "test_macro_f1@0.5": test_f1,
            }
        )
    return metrics_out


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_argparser().parse_args(argv)
    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)
    data = load_cca_data(args, device)
    return train_cca(args, data, device)


if __name__ == "__main__":
    main()
