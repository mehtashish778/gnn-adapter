#!/usr/bin/env python3
"""
Train bipartite NativeGNN (attribute → object, weighted-mean messages) with frozen CLIP object features +
VLM (logit, prob) as attribute node features. No fixed C×C adjacency.

Contrast:
  07: homogeneous label graph, residual MLP + single adj matmul
  12: homogeneous label graph, K rounds of adj @ H + CLIP per label node
  13: bipartite (C attribute nodes → 1 object node), stacked bipartite layers + CLIP as object init
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    clip_image_embeds_tensor,
    load_per_class_thresholds,
    load_rows,
    masked_bce_with_logits,
    masked_macro_f1,
    masked_subset_accuracy,
    require_cuda_device,
    resolve_dataset_image_path,
    set_seed,
    to_label_tensors,
    write_json,
)
from gnn_bipartite import build_bipartite_edge_weights
from model_registry import resolve_experiment_dir, update_run_registry
from models.architectures.gnn13_clip_bipartite import ClipObjectBipartiteGNN


class RowTensorDataset(Dataset):
    def __init__(
        self,
        clip_emb: torch.Tensor,
        logits: torch.Tensor,
        probs: torch.Tensor,
        y_true: torch.Tensor,
        y_mask: torch.Tensor,
    ):
        self.clip_emb = clip_emb
        self.logits = logits
        self.probs = probs
        self.y_true = y_true
        self.y_mask = y_mask

    def __len__(self) -> int:
        return self.clip_emb.shape[0]

    def __getitem__(self, i: int):
        return (
            self.clip_emb[i],
            self.logits[i],
            self.probs[i],
            self.y_true[i],
            self.y_mask[i],
        )


@torch.no_grad()
def compute_clip_embeddings(
    rows: List[dict],
    image_root: Path,
    clip: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    clip.eval()
    chunks = []
    for start in tqdm(range(0, len(rows), batch_size), desc="CLIP encode"):
        batch_rows = rows[start : start + batch_size]
        images = []
        for r in batch_rows:
            p = resolve_dataset_image_path(image_root, r["path"])
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=torch.float32)
        feat = clip_image_embeds_tensor(clip, pv)
        chunks.append(feat.detach().cpu())
    return torch.cat(chunks, dim=0)


def maybe_load_clip_cache(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def save_clip_cache(
    path: Path,
    clip_model_name: str,
    train_paths: List[str],
    train_e: torch.Tensor,
    val_paths: List[str],
    val_e: torch.Tensor,
    test_paths: List[str],
    test_e: torch.Tensor,
    calib_paths: Optional[List[str]] = None,
    calib_e: Optional[torch.Tensor] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clip_model_name": clip_model_name,
            "train_paths": train_paths,
            "train_emb": train_e,
            "val_paths": val_paths,
            "val_emb": val_e,
            "test_paths": test_paths,
            "test_emb": test_e,
            "calib_paths": calib_paths,
            "calib_emb": calib_e,
        },
        path,
    )


def verify_paths_order(stored: List[str], rows: List[dict]) -> None:
    for a, b in zip(stored, rows):
        if a != b["path"]:
            raise ValueError("CLIP cache path order mismatch vs rows JSON; delete cache and re-encode.")


def main():
    parser = argparse.ArgumentParser(description="Train CLIP object + bipartite attribute GNN.")
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--calib_rows_json", default=None, help="Optional calibration rows JSON (threshold tuning).")
    parser.add_argument("--per_class_thresholds_json", default="data/processed/experiments/thresholds/per_class_thresholds.json")
    parser.add_argument(
        "--image_root",
        default="data/raw",
        help="Image root; CheXpert-v1.0-small/... in row paths maps to <root>/train/...",
    )
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_cache_pt", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--object_feature_dim", type=int, default=512, help="CLIP projected dim (= first object hidden).")
    parser.add_argument("--gnn_hidden_dims", default="512,256", help="Comma-separated out_dims per bipartite layer.")
    parser.add_argument("--gnn_mid_dim", type=int, default=0, help="Attribute path mid dim; 0 = use each layer out_dim.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5, help="Residual scale on frozen VLM logits.")
    parser.add_argument("--edge_mode", choices=("all", "vlm_positive"), default="all")
    parser.add_argument("--vlm_tau", type=float, default=0.5, help="For vlm_positive edge mask.")
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
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--resume_from", default="", help="Optional checkpoint path to initialize adapter weights.")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    hidden_dims = [int(x.strip()) for x in args.gnn_hidden_dims.split(",") if x.strip()]
    if not hidden_dims:
        raise ValueError("--gnn_hidden_dims must list at least one dim, e.g. 512,256")
    mid_dim: Optional[int] = None if args.gnn_mid_dim <= 0 else args.gnn_mid_dim

    device = require_cuda_device(args.gpu_id)
    set_seed(args.seed)

    train_rows = load_rows(Path(args.train_rows_json))
    val_rows = load_rows(Path(args.val_rows_json))
    test_rows = load_rows(Path(args.test_rows_json))
    calib_rows = load_rows(Path(args.calib_rows_json)) if args.calib_rows_json else None
    n_train, n_val, n_test = len(train_rows), len(val_rows), len(test_rows)
    n_calib = len(calib_rows) if calib_rows is not None else 0
    print({"dataset_sizes": {"train": n_train, "val": n_val, "calib": n_calib, "test": n_test}, "variant": "13_bipartite_gnn"})

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

    best_metric = args.best_metric
    if best_metric == "val_macro_f1_thr" and thr_list is None:
        print("Warning: val_macro_f1_thr not available; using val_bce.")
        best_metric = "val_bce"

    cache_path = Path(args.clip_cache_pt) if args.clip_cache_pt else None
    cache = maybe_load_clip_cache(cache_path)
    image_root = Path(args.image_root)

    if cache is not None and cache.get("clip_model_name") != args.clip_model:
        print("CLIP cache model mismatch; re-encoding.")
        cache = None

    if cache is None:
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()
        tr_e = compute_clip_embeddings(train_rows, image_root, clip_model, processor, device, args.clip_batch_size)
        va_e = compute_clip_embeddings(val_rows, image_root, clip_model, processor, device, args.clip_batch_size)
        te_e = compute_clip_embeddings(test_rows, image_root, clip_model, processor, device, args.clip_batch_size)
        ca_e = None
        if calib_rows is not None:
            print("Encoding calib split with CLIP...")
            ca_e = compute_clip_embeddings(
                calib_rows, image_root, clip_model, processor, device, args.clip_batch_size
            )
        clip_dim = tr_e.shape[1]
        if cache_path:
            save_clip_cache(
                cache_path,
                args.clip_model,
                [r["path"] for r in train_rows],
                tr_e,
                [r["path"] for r in val_rows],
                va_e,
                [r["path"] for r in test_rows],
                te_e,
                calib_paths=[r["path"] for r in calib_rows] if calib_rows is not None else None,
                calib_e=ca_e,
            )
            print({"saved_clip_cache": str(cache_path)})
        del clip_model, processor
        torch.cuda.empty_cache()
    else:
        verify_paths_order(cache["train_paths"], train_rows)
        verify_paths_order(cache["val_paths"], val_rows)
        verify_paths_order(cache["test_paths"], test_rows)
        tr_e = cache["train_emb"].float()
        va_e = cache["val_emb"].float()
        te_e = cache["test_emb"].float()
        clip_dim = tr_e.shape[1]
        ca_e = None
        if calib_rows is not None:
            stored_calib_paths = cache.get("calib_paths")
            stored_calib_emb = cache.get("calib_emb")
            if stored_calib_paths is not None and stored_calib_emb is not None:
                verify_paths_order(stored_calib_paths, calib_rows)
                ca_e = stored_calib_emb.float()
            else:
                processor = CLIPProcessor.from_pretrained(args.clip_model)
                clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
                for p in clip_model.parameters():
                    p.requires_grad = False
                clip_model.eval()
                print("Encoding calib split with CLIP (cache missing calib)...")
                ca_e = compute_clip_embeddings(
                    calib_rows, image_root, clip_model, processor, device, args.clip_batch_size
                )
                if cache_path:
                    save_clip_cache(
                        cache_path,
                        args.clip_model,
                        [r["path"] for r in train_rows],
                        tr_e,
                        [r["path"] for r in val_rows],
                        va_e,
                        [r["path"] for r in test_rows],
                        te_e,
                        calib_paths=[r["path"] for r in calib_rows],
                        calib_e=ca_e,
                    )
                    print({"saved_clip_cache": str(cache_path)})
                del clip_model, processor
                torch.cuda.empty_cache()

    adapter = ClipObjectBipartiteGNN(
        clip_dim=clip_dim,
        object_feature_dim=args.object_feature_dim,
        num_attributes=c,
        hidden_dims=hidden_dims,
        mid_dim=mid_dim,
        dropout=args.dropout,
        alpha=args.alpha,
    ).to(device)
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        state = ckpt.get("adapter_state_dict", ckpt)
        adapter.load_state_dict(state)

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    else:
        ca_y_d = None
        ca_m_d = None

    train_loader = DataLoader(
        RowTensorDataset(tr_e, tr_logits, tr_probs, tr_y, tr_m),
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

        adapter.train()
        epoch_losses = []
        for ce, ll, pp, yt, ym in train_loader:
            ce = ce.to(device, non_blocking=True)
            ll = ll.to(device, non_blocking=True)
            pp = pp.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
            ym = ym.to(device, non_blocking=True)
            edge_w = build_bipartite_edge_weights(pp, args.edge_mode, args.vlm_tau).to(device)
            opt.zero_grad()
            out = adapter(ce, ll, pp, edge_w)
            raw = F.binary_cross_entropy_with_logits(out, yt, pos_weight=pos_weight, reduction="none")
            loss = (raw * ym).sum() / ym.sum().clamp(min=1.0)
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip_norm)
            opt.step()
            epoch_losses.append(float(loss.item()))

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        adapter.eval()
        val_out_parts = []
        with torch.no_grad():
            nv = va_e.shape[0]
            for start in range(0, nv, args.batch_size):
                ce = va_e[start : start + args.batch_size].to(device, non_blocking=True)
                ll = va_logits[start : start + args.batch_size].to(device, non_blocking=True)
                pp = va_probs[start : start + args.batch_size].to(device, non_blocking=True)
                edge_w = build_bipartite_edge_weights(pp, args.edge_mode, args.vlm_tau).to(device)
                val_out_parts.append(adapter(ce, ll, pp, edge_w))
            val_out = torch.cat(val_out_parts, dim=0)
            val_prob = torch.sigmoid(val_out)
            val_bce = float(masked_bce_with_logits(val_out, va_y_d, va_m_d, pos_weight).item())
            val_f1_05 = masked_macro_f1(val_prob, va_y_d, va_m_d, threshold=0.5)
            val_f1_thr = (
                masked_macro_f1(val_prob, va_y_d, va_m_d, threshold=thr_list) if thr_list is not None else val_f1_05
            )

        row = {
            "epoch": epoch,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_bce": val_bce,
            "val_macro_f1@0.5": val_f1_05,
            "val_macro_f1@thr": val_f1_thr,
        }
        history.append(row)
        if plateau_sched is not None:
            plateau_sched.step(val_bce)

        if is_better(best_metric, val_bce, val_f1_05, val_f1_thr):
            if best_metric == "val_bce":
                best["score"] = val_bce
            elif best_metric == "val_macro_f1_05":
                best["score"] = val_f1_05
            else:
                best["score"] = val_f1_thr
            best["state_dict"] = {k: v.cpu() for k, v in adapter.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
            print({"early_stop": True, "epoch": epoch})
            break

    if best["state_dict"] is None:
        raise RuntimeError("No checkpoint saved.")

    adapter.load_state_dict(best["state_dict"])
    adapter.eval()

    def logits_for(t_clip: torch.Tensor, t_logits: torch.Tensor, t_probs: torch.Tensor) -> torch.Tensor:
        parts = []
        with torch.no_grad():
            n = t_clip.shape[0]
            for start in range(0, n, args.batch_size):
                ce = t_clip[start : start + args.batch_size].to(device, non_blocking=True)
                ll = t_logits[start : start + args.batch_size].to(device, non_blocking=True)
                pp = t_probs[start : start + args.batch_size].to(device, non_blocking=True)
                edge_w = build_bipartite_edge_weights(pp, args.edge_mode, args.vlm_tau).to(device)
                parts.append(adapter(ce, ll, pp, edge_w))
        return torch.cat(parts, dim=0)

    val_out = logits_for(va_e, va_logits, va_probs)
    test_out = logits_for(te_e, te_logits, te_probs)
    val_prob = torch.sigmoid(val_out).cpu()
    test_prob = torch.sigmoid(test_out).cpu()
    val_f1 = masked_macro_f1(val_prob.to(device), va_y_d, va_m_d, threshold=0.5)
    test_f1 = masked_macro_f1(test_prob.to(device), te_y_d, te_m_d, threshold=0.5)
    calib_prob = None
    calib_f1 = None
    calib_f1_thr_eval = None
    if calib_rows is not None:
        calib_out = logits_for(ca_e, ca_logits, ca_probs)
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
            masked_macro_f1(calib_prob.to(device), ca_y_d, ca_m_d, threshold=thr_list) if thr_list is not None else calib_f1
        )

    val_vp = val_prob.to(device)
    val_sub = masked_subset_accuracy(val_vp, va_y_d, va_m_d, threshold=0.5)
    test_sub = masked_subset_accuracy(test_prob.to(device), te_y_d, te_m_d, threshold=0.5)
    calib_sub = None
    calib_sub_thr = None
    if calib_rows is not None:
        calib_sub = masked_subset_accuracy(calib_prob.to(device), ca_y_d, ca_m_d, threshold=0.5)
        calib_sub_thr = masked_subset_accuracy(calib_prob.to(device), ca_y_d, ca_m_d, threshold=thr_list) if thr_list is not None else calib_sub
    val_sub_thr = masked_subset_accuracy(val_vp, va_y_d, va_m_d, threshold=thr_list) if thr_list is not None else val_sub
    test_sub_thr = masked_subset_accuracy(test_prob.to(device), te_y_d, te_m_d, threshold=thr_list) if thr_list is not None else test_sub

    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or None,
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/bipartite_clip_gnn_adapter",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": best["state_dict"],
            "adapter_hparams": {
                "clip_dim": clip_dim,
                "object_feature_dim": args.object_feature_dim,
                "num_labels": c,
                "gnn_hidden_dims": hidden_dims,
                "gnn_mid_dim": mid_dim,
                "alpha": args.alpha,
                "edge_mode": args.edge_mode,
                "vlm_tau": args.vlm_tau,
            },
        },
        out_dir / "best_checkpoint.pt",
    )
    write_json(
        out_dir / "metrics.json",
        {
            "variant": "bipartite_clip_gnn",
            "dataset_sizes": {"train": n_train, "val": n_val, "calib": n_calib, "test": n_test},
            "clip_model": args.clip_model,
            "best_metric": best_metric,
            "best_score": best["score"],
            "seed": args.seed,
            "hparams": vars(args),
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
        },
    )
    write_json(out_dir / "history.json", history)
    write_json(
        out_dir / "val_predictions.json",
        {"probs": val_prob.tolist(), "y_true": va_y.tolist(), "y_mask": va_m.tolist()},
    )
    if calib_rows is not None:
        write_json(
            out_dir / "calib_predictions.json",
            {"probs": calib_prob.tolist(), "y_true": ca_y.tolist(), "y_mask": ca_m.tolist()},
        )
    write_json(
        out_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": te_y.tolist(), "y_mask": te_m.tolist()},
    )
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
                "val_subset_accuracy@per_class_thr": val_sub_thr,
                "test_subset_accuracy@per_class_thr": test_sub_thr,
            },
            hparams={
                "epochs": len(history),
                "lr": args.lr,
                "best_metric": best_metric,
                "gnn_hidden_dims": args.gnn_hidden_dims,
            },
        )
    print(
        {
            "variant": "bipartite_clip_gnn",
            "best_metric": best_metric,
            "best_score": best["score"],
            "val_macro_f1@0.5": val_f1,
            "test_macro_f1@0.5": test_f1,
            "test_subset_accuracy@0.5": test_sub,
            "test_subset_accuracy@per_class_thr": test_sub_thr,
            "val_macro_f1@per_class_thr": val_f1_thr_eval,
            "test_macro_f1@per_class_thr": test_f1_thr_eval,
        }
    )


if __name__ == "__main__":
    main()
