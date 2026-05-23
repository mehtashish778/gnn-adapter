#!/usr/bin/env python3
"""Evaluate a CheXpert-trained model on NIH (or other cross-site) rows — inference only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cca_train_core import (
    extract_patch_tokens,
    load_radgraph_prior,
    load_split_patches,
    resolve_patch_cache_config,
)
from common_multilabel import (
    build_adj,
    clip_image_embeds_tensor,
    load_rows,
    masked_macro_f1,
    require_cuda_device,
    to_label_tensors,
    to_vlm_feature_tensors,
)
from crosssite_common import (
    load_checkpoint_state,
    load_metrics_hparams,
    resolve_chexpert_run_dir,
    require_vlm_scores,
    write_crosssite_eval,
)
from feature_cache import FeatureCache
from models.architectures.cca import CCAModel, DEFAULT_CONCEPT_PHRASES
from models.architectures.gnn07_label_residual import ResidualLabelGNNModel
from models.architectures.gnn12_clip_vlm_homo import ClipVlmGraphAdapter
from models.architectures.gnn13_clip_bipartite import ClipObjectBipartiteGNN
from models.architectures.mlgcn import MLGCN
from models.architectures.posthoc_cbm import PostHocCBM
from models.architectures.qformer_adapter import QFormerAdapter
from models.architectures.vlm_mlp import VLMFeatureMLP
from gnn_bipartite import build_bipartite_edge_weights


@torch.no_grad()
def _batched_probs(logits_fn, n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    parts = []
    for start in range(0, n, batch_size):
        parts.append(torch.sigmoid(logits_fn(start, start + batch_size).float()))
    return torch.cat(parts, dim=0)


def eval_vlm_mlp(rows, ckpt_dir: Path, device, batch_size: int) -> torch.Tensor:
    require_vlm_scores(rows)
    x, _, _, y, m = to_vlm_feature_tensors(rows)
    state = load_checkpoint_state(ckpt_dir)
    in_dim = state["0.weight"].shape[1]
    out_dim = state["3.weight"].shape[0]
    hidden = state["0.weight"].shape[0]
    model = VLMFeatureMLP(in_dim, out_dim, hidden_dim=hidden).to(device)
    model.load_state_dict(state)
    model.eval()

    def fn(s, e):
        return model(x[s:e].to(device))

    return _batched_probs(fn, len(rows), batch_size, device), y, m


def eval_cbm_posthoc(rows, ckpt_dir: Path, device, batch_size: int) -> tuple:
    require_vlm_scores(rows)
    x, _, _, y, m = to_vlm_feature_tensors(rows)
    hp = load_metrics_hparams(ckpt_dir)
    num_concepts = int(hp.get("num_concepts", 30))
    state = load_checkpoint_state(ckpt_dir)
    in_dim = state["concept_proj.weight"].shape[1]
    c = y.shape[1]
    model = PostHocCBM(in_dim, num_concepts, c).to(device)
    model.load_state_dict(state)
    model.eval()

    def fn(s, e):
        out, _ = model(x[s:e].to(device))
        return out

    return _batched_probs(fn, len(rows), batch_size, device), y, m


def eval_mlgcn(rows, ckpt_dir: Path, device, batch_size: int, chexpert_train_json: Path) -> tuple:
    require_vlm_scores(rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(rows)
    tr_rows = load_rows(chexpert_train_json)
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_mlgcn", _SCRIPT_DIR / "18_train_mlgcn.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    adj = mod.build_label_adj(tr_rows, te_y.shape[1]).to(device)

    state = load_checkpoint_state(ckpt_dir)
    model = MLGCN(te_y.shape[1]).to(device)
    model.load_state_dict(state, strict=False)
    model.set_adjacency(adj)
    model.eval()

    def fn(s, e):
        return model(te_logits[s:e].to(device), te_probs[s:e].to(device))

    return _batched_probs(fn, len(rows), batch_size, device), te_y, te_m


def eval_gnn07(rows, ckpt_dir: Path, device, batch_size: int) -> tuple:
    require_vlm_scores(rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(rows)
    edge_index_path = _SCRIPT_DIR.parent / "data/processed/graph/edge_index.json"
    edge_weight_path = _SCRIPT_DIR.parent / "data/processed/graph/edge_weight.json"
    with edge_index_path.open("r", encoding="utf-8") as f:
        edge_index = json.load(f)
    with edge_weight_path.open("r", encoding="utf-8") as f:
        edge_weight = json.load(f)
    adj = build_adj(edge_index, edge_weight, te_y.shape[1]).to(device)

    state = load_checkpoint_state(ckpt_dir)
    hidden = int(state["fc1.weight"].shape[0])
    alpha = float(load_metrics_hparams(ckpt_dir).get("alpha", 0.5))
    model = ResidualLabelGNNModel(hidden_dim=hidden, alpha=alpha).to(device)
    model.load_state_dict(state)
    model.eval()

    def fn(s, e):
        return model(te_logits[s:e].to(device), te_probs[s:e].to(device), adj)

    return _batched_probs(fn, len(rows), batch_size, device), te_y, te_m


@torch.no_grad()
def compute_clip_pooled(
    rows: List[dict],
    image_root: Path,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    from common_multilabel import resolve_dataset_image_path
    from PIL import Image
    from tqdm import tqdm

    chunks = []
    clip_model.eval()
    for start in tqdm(range(0, len(rows), batch_size), desc="CLIP pooled"):
        batch = rows[start : start + batch_size]
        images = []
        for r in batch:
            with Image.open(resolve_dataset_image_path(image_root, r["path"])) as im:
                images.append(im.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=torch.float32)
        feat = clip_image_embeds_tensor(clip_model, pv)
        chunks.append(feat.cpu())
    return torch.cat(chunks, dim=0)


def eval_gnn12(rows, ckpt_dir: Path, device, batch_size: int, image_root: Path, args_ns) -> tuple:
    require_vlm_scores(rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(rows)
    hp = load_metrics_hparams(ckpt_dir)
    clip_name = hp.get("clip_model", "openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name, use_safetensors=True).to(device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_emb = compute_clip_pooled(rows, image_root, clip_model, processor, device, args_ns.clip_batch_size)

    edge_index_path = Path(hp.get("edge_index_json", "data/processed/graph/edge_index.json"))
    edge_weight_path = Path(hp.get("edge_weight_json", "data/processed/graph/edge_weight.json"))
    if not edge_index_path.is_absolute():
        edge_index_path = _SCRIPT_DIR.parent / edge_index_path
    if not edge_weight_path.is_absolute():
        edge_weight_path = _SCRIPT_DIR.parent / edge_weight_path
    with edge_index_path.open("r", encoding="utf-8") as f:
        edge_index = json.load(f)
    with edge_weight_path.open("r", encoding="utf-8") as f:
        edge_weight = json.load(f)
    adj = build_adj(edge_index, edge_weight, te_y.shape[1]).to(device)

    state = load_checkpoint_state(ckpt_dir)
    clip_dim = int(state["clip_to_h.weight"].shape[1])
    hidden = int(state["clip_to_h.weight"].shape[0])
    gnn_layers = sum(1 for k in state if k.startswith("gnn_layers.")) // 2
    gnn_layers = max(1, gnn_layers)
    alpha = float(hp.get("alpha", 0.5))
    model = ClipVlmGraphAdapter(clip_dim, te_y.shape[1], hidden, gnn_layers, alpha).to(device)
    model.load_state_dict(state)
    model.eval()

    probs_parts = []
    for start in range(0, len(rows), batch_size):
        ce = clip_emb[start : start + batch_size].to(device)
        ll = te_logits[start : start + batch_size].to(device)
        pp = te_probs[start : start + batch_size].to(device)
        out = model(ce, ll, pp, adj)
        probs_parts.append(torch.sigmoid(out).cpu())
    return torch.cat(probs_parts, dim=0), te_y, te_m


def eval_gnn13(rows, ckpt_dir: Path, device, batch_size: int, image_root: Path) -> tuple:
    require_vlm_scores(rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(rows)
    hp = load_metrics_hparams(ckpt_dir)
    clip_name = hp.get("clip_model", "openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name, use_safetensors=True).to(device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_emb = compute_clip_pooled(rows, image_root, clip_model, processor, device, batch_size)

    state = load_checkpoint_state(ckpt_dir)
    clip_dim = int(state["clip_proj.weight"].shape[1])
    obj_dim = int(state["clip_proj.weight"].shape[0])
    hidden_dims = []
    i = 0
    while f"gnn.layers.{i}.attr_to_mid.weight" in state:
        hidden_dims.append(int(state[f"gnn.layers.{i}.update.0.weight"].shape[0]))
        i += 1
    if not hidden_dims:
        hidden_dims = [int(state["gnn.classifier.weight"].shape[1])]
    alpha = float(hp.get("alpha", 0.5))
    dropout = float(hp.get("dropout", 0.2))
    model = ClipObjectBipartiteGNN(
        clip_dim=clip_dim,
        object_feature_dim=obj_dim,
        num_attributes=te_y.shape[1],
        hidden_dims=hidden_dims,
        mid_dim=None,
        dropout=dropout,
        alpha=alpha,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    edge_mode = hp.get("edge_mode", "all")
    vlm_tau = float(hp.get("vlm_tau", 0.5))
    probs_parts = []
    for start in range(0, len(rows), batch_size):
        ce = clip_emb[start : start + batch_size].to(device)
        ll = te_logits[start : start + batch_size].to(device)
        pp = te_probs[start : start + batch_size].to(device)
        ew = build_bipartite_edge_weights(pp, edge_mode, vlm_tau)
        out = model(ce, ll, pp, ew)
        probs_parts.append(torch.sigmoid(out).cpu())
    return torch.cat(probs_parts, dim=0), te_y, te_m


def eval_cbm_labelfree(rows, ckpt_dir: Path, device, batch_size: int, image_root: Path) -> tuple:
    _, _, y, m = to_label_tensors(rows)
    hp = load_metrics_hparams(ckpt_dir)
    num_concepts = int(hp.get("num_concepts", 30))
    clip_name = hp.get("clip_model", "openai/clip-vit-base-patch16")
    phrases = DEFAULT_CONCEPT_PHRASES[:num_concepts]
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name, use_safetensors=True).to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    import importlib.util

    spec = importlib.util.spec_from_file_location("labelfree", _SCRIPT_DIR / "16_train_labelfree_cbm.py")
    lf = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(lf)
    x = lf.clip_concept_features(rows, image_root, clip_model, processor, device, phrases, batch_size)

    state = load_checkpoint_state(ckpt_dir)
    head = nn.Linear(num_concepts, y.shape[1])
    head.load_state_dict(state)
    head.to(device).eval()

    def fn(s, e):
        return head(x[s:e].to(device))

    return _batched_probs(fn, len(rows), batch_size, device), y, m


def eval_qformer(rows, ckpt_dir: Path, device, batch_size: int, image_root: Path, protocol: str) -> tuple:
    require_vlm_scores(rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(rows)
    hp = load_metrics_hparams(ckpt_dir)
    ns = SimpleNamespace(
        protocol=protocol,
        clip_model=hp.get("clip_model", "openai/clip-vit-base-patch16"),
        clip_batch_size=int(hp.get("clip_batch_size", 16)),
        lora_rank=None,
        patch_encoder_id="",
        patch_cache_version="",
    )
    feature_cache = FeatureCache(hp.get("embeddings_cache_dir", "data/processed/embeddings"))
    processor = CLIPProcessor.from_pretrained(ns.clip_model)
    clip_model = CLIPModel.from_pretrained(ns.clip_model, use_safetensors=True).to(device)
    for p in clip_model.parameters():
        p.requires_grad = False
    patches = load_split_patches(
        rows, "test", ns, image_root, clip_model, processor, device, None, feature_cache
    )

    state = load_checkpoint_state(ckpt_dir)
    patch_dim = patches.shape[-1]
    query_dim = int(state["queries"].shape[-1]) if "queries" in state else int(hp.get("query_dim", 128))
    num_queries = int(state["queries"].shape[0]) if "queries" in state else int(hp.get("num_queries", 32))
    n_heads = int(hp.get("n_heads", 4))
    n_layers = int(hp.get("n_cross_attn_layers", 2))
    dropout = float(hp.get("dropout", 0.1))
    model = QFormerAdapter(
        patch_dim=patch_dim,
        query_dim=query_dim,
        num_queries=num_queries,
        num_labels=te_y.shape[1],
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    probs_parts = []
    for start in range(0, len(rows), batch_size):
        out = model(patches[start : start + batch_size].to(device))
        probs_parts.append(torch.sigmoid(out).cpu())
    return torch.cat(probs_parts, dim=0), te_y, te_m


def eval_cca(rows, ckpt_dir: Path, device, batch_size: int, image_root: Path, protocol: str) -> tuple:
    require_vlm_scores(rows)
    te_logits, te_probs, te_y, te_m = to_label_tensors(rows)
    hp = load_metrics_hparams(ckpt_dir)
    lora_rank = hp.get("lora_rank")
    if lora_rank is not None:
        lora_rank = int(lora_rank)

    ns = SimpleNamespace(
        protocol=protocol,
        clip_model=hp.get("clip_model", "openai/clip-vit-base-patch16"),
        clip_batch_size=int(hp.get("clip_batch_size", 16)),
        lora_rank=lora_rank,
        patch_encoder_id=hp.get("patch_encoder_id", ""),
        patch_cache_version=hp.get("patch_cache_version", ""),
    )
    feature_cache = FeatureCache(hp.get("embeddings_cache_dir", "data/processed/embeddings"))

    clip_model = None
    processor = None
    if lora_rank:
        from peft import PeftModel

        default_ad = _SCRIPT_DIR.parent / "data/processed/embeddings" / f"lora_r{lora_rank}_adapter"
        adapter_dir = hp.get("lora_adapter_dir") or str(default_ad)
        if not Path(adapter_dir).is_dir():
            raise FileNotFoundError(
                f"LoRA CLIP adapter not found at {adapter_dir}. "
                f"Run scripts/19_train_lora_clip_vision.py --lora_rank {lora_rank} on CheXpert first."
            )
        processor = CLIPProcessor.from_pretrained(ns.clip_model)
        clip_model = CLIPModel.from_pretrained(ns.clip_model, use_safetensors=True).to(device)
        clip_model.vision_model = PeftModel.from_pretrained(clip_model.vision_model, adapter_dir)
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()
    else:
        processor = CLIPProcessor.from_pretrained(ns.clip_model)
        clip_model = CLIPModel.from_pretrained(ns.clip_model, use_safetensors=True).to(device)
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()

    patches = load_split_patches(
        rows, "test", ns, image_root, clip_model, processor, device, None, feature_cache
    )
    patch_dim = patches.shape[-1]

    model = CCAModel(
        patch_dim=patch_dim,
        query_dim=int(hp.get("query_dim", 64)),
        num_primitives=int(hp.get("num_primitives", 30)),
        num_findings=te_y.shape[1],
        n_heads=int(hp.get("n_heads", 2)),
        n_cross_attn_layers=int(hp.get("n_cross_attn_layers", 1)),
        n_self_attn_layers=int(hp.get("n_self_attn_layers", 2)),
        alpha=float(hp.get("alpha", 0.5)),
        dropout=float(hp.get("dropout", 0.1)),
        use_gate_M=bool(hp.get("use_gate_M", False)),
        gumbel_tau=float(hp.get("gumbel_tau_init", 0.5)),
    ).to(device)
    state = load_checkpoint_state(ckpt_dir)
    model.load_state_dict(state, strict=False)
    rad = load_radgraph_prior(hp.get("radgraph_prior_json", ""), model.num_primitives, device)
    if rad is not None:
        model.layer2.set_radgraph_prior(rad)
    model.eval()
    gumbel_tau = float(hp.get("gumbel_tau_min", 0.5))

    probs_parts = []
    for start in range(0, len(rows), batch_size):
        p = patches[start : start + batch_size].to(device)
        ll = te_logits[start : start + batch_size].to(device)
        pp = te_probs[start : start + batch_size].to(device)
        out, _, _ = model(p, ll, pp, radgraph_prior=rad, gumbel_tau=gumbel_tau)
        probs_parts.append(torch.sigmoid(out).cpu())
    return torch.cat(probs_parts, dim=0), te_y, te_m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--chexpert_run_dir", default="")
    parser.add_argument("--test_rows_json", default="data/processed/splits/nih/test_rows.json")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument(
        "--chexpert_train_rows_json",
        default="data/processed/splits/train_rows.json",
        help="For MLGCN adjacency built on CheXpert train.",
    )
    parser.add_argument("--protocol", default="nih")
    parser.add_argument("--run_id", default="crosssite_eval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = require_cuda_device(args.gpu_id)
    rows = load_rows(Path(args.test_rows_json))
    ckpt_dir = resolve_chexpert_run_dir(
        args.model_id, args.chexpert_run_dir or None
    )
    image_root = Path(args.image_root)

    metrics_path = ckpt_dir / "metrics.json"
    trainable = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            trainable = json.load(f).get("trainable_params")

    ns = SimpleNamespace(clip_batch_size=args.clip_batch_size)
    model_id = args.model_id

    if model_id == "vlm_mlp":
        probs, y, m = eval_vlm_mlp(rows, ckpt_dir, device, args.batch_size)
    elif model_id == "cbm_posthoc":
        probs, y, m = eval_cbm_posthoc(rows, ckpt_dir, device, args.batch_size)
    elif model_id == "mlgcn":
        probs, y, m = eval_mlgcn(
            rows, ckpt_dir, device, args.batch_size, Path(args.chexpert_train_rows_json)
        )
    elif model_id == "gnn07_label_residual":
        probs, y, m = eval_gnn07(rows, ckpt_dir, device, args.batch_size)
    elif model_id == "gnn12_clip_vlm_homo":
        probs, y, m = eval_gnn12(rows, ckpt_dir, device, args.batch_size, image_root, ns)
    elif model_id == "gnn13_clip_bipartite":
        probs, y, m = eval_gnn13(rows, ckpt_dir, device, args.batch_size, image_root)
    elif model_id == "cbm_labelfree":
        probs, y, m = eval_cbm_labelfree(rows, ckpt_dir, device, args.clip_batch_size, image_root)
    elif model_id == "qformer_adapter":
        probs, y, m = eval_qformer(rows, ckpt_dir, device, args.batch_size, image_root, args.protocol)
    elif model_id == "cca":
        probs, y, m = eval_cca(rows, ckpt_dir, device, args.batch_size, image_root, args.protocol)
    else:
        raise ValueError(
            f"model_id={model_id} not handled here. Use 05_run_baseline_frozen_vlm (vlm_zeroshot) "
            "or score_qwen2vl_lora (qwen2vl_lora_r16)."
        )

    out_dir = write_crosssite_eval(
        model_id=model_id,
        protocol=args.protocol,
        run_id=args.run_id,
        probs=probs,
        y_true=y,
        y_mask=m,
        trainable_params=trainable,
        chexpert_run_dir=ckpt_dir,
    )
    print({"wrote": str(out_dir), "test_macro_f1@0.5": float(masked_macro_f1(probs, y, m))})


if __name__ == "__main__":
    main()
