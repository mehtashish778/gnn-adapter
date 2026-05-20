from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from common_multilabel import write_json


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    description: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "vlm_zeroshot": ModelSpec(
        model_id="vlm_zeroshot",
        display_name="VLMZeroShot",
        description="Frozen VLM outputs used directly (no adapter head).",
    ),
    "vlm_mlp": ModelSpec(
        model_id="vlm_mlp",
        display_name="VLMFeatureMLP",
        description="MLP adapter over frozen VLM logits/probabilities.",
    ),
    "gnn07_label_residual": ModelSpec(
        model_id="gnn07_label_residual",
        display_name="LabelGraphResidualGNN",
        description="Residual label-graph GNN over (logit, prob) node features.",
    ),
    "gnn12_clip_vlm_homo": ModelSpec(
        model_id="gnn12_clip_vlm_homo",
        display_name="ClipVlmHomogeneousGNN",
        description="CLIP+VLM features on homogeneous label graph.",
    ),
    "gnn13_clip_bipartite": ModelSpec(
        model_id="gnn13_clip_bipartite",
        display_name="ClipBipartiteAttributeGNN",
        description="CLIP object node + VLM attribute nodes with bipartite message passing.",
    ),
    "cca": ModelSpec(
        model_id="cca",
        display_name="CompositionalConceptAdapter",
        description="Concept-evidence adapter: patch cross-attention → compositional self-attn → gated findings readout.",
    ),
    "cbm_posthoc": ModelSpec(
        model_id="cbm_posthoc",
        display_name="PostHocCBM",
        description="Post-hoc concept bottleneck on VLM [logits; probs].",
    ),
    "cbm_labelfree": ModelSpec(
        model_id="cbm_labelfree",
        display_name="LabelFreeCBM",
        description="Label-free CBM with CLIP concept similarities.",
    ),
    "qformer_adapter": ModelSpec(
        model_id="qformer_adapter",
        display_name="QFormerAdapter",
        description="Learnable query cross-attention over ViT patches.",
    ),
    "mlgcn": ModelSpec(
        model_id="mlgcn",
        display_name="MLGCN",
        description="ML-GCN style label-graph message passing.",
    ),
    "qwen2vl_lora_r16": ModelSpec(
        model_id="qwen2vl_lora_r16",
        display_name="Qwen2VLLoRA16Cls",
        description="Qwen2-VL-2B-Instruct + LoRA r=16, classification head, masked BCE.",
    ),
    "qwen2vl_lora_r16_sft": ModelSpec(
        model_id="qwen2vl_lora_r16_sft",
        display_name="Qwen2VLLoRA16SFT",
        description="Qwen2-VL-2B-Instruct + LoRA r=16, generative JSON SFT.",
    ),
}


def auto_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def resolve_experiment_dir(
    *,
    out_dir: Optional[str],
    model_id: Optional[str],
    protocol: Optional[str],
    run_id: Optional[str],
    default_legacy_out_dir: str,
) -> Path:
    # Backward-compatible path behavior:
    # - If out_dir is explicit, use it unchanged.
    # - Else if model_id+protocol are provided, use organized path.
    # - Else fallback to legacy default out_dir.
    if out_dir:
        return Path(out_dir)
    if model_id and protocol:
        rid = run_id or auto_run_id()
        return Path("data/processed/experiments") / model_id / protocol / rid
    return Path(default_legacy_out_dir)


def update_run_registry(
    *,
    model_id: str,
    protocol: str,
    run_dir: Path,
    metrics: dict,
    hparams: Optional[dict] = None,
) -> None:
    base = Path("data/processed/experiments") / model_id / protocol
    base.mkdir(parents=True, exist_ok=True)
    index_path = base / "runs_index.json"
    payload = {"model_id": model_id, "protocol": protocol, "runs": []}
    if index_path.exists():
        import json

        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    record = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "hparams": hparams or {},
    }
    payload["runs"] = [r for r in payload.get("runs", []) if r.get("run_id") != run_dir.name]
    payload["runs"].append(record)
    write_json(index_path, payload)

    # latest pointer
    write_json(base / "latest.json", {"run_id": run_dir.name, "run_dir": str(run_dir)})

    # best pointer by test macro F1 if available, else val macro F1.
    def score(rec: dict) -> float:
        m = rec.get("metrics", {}) or {}
        for k in [
            "test_macro_f1_calibrated",
            "test_macro_f1@per_class_thr",
            "test_macro_f1@0.5",
            "test_macro_f1",
            "val_macro_f1_calibrated",
            "val_macro_f1@per_class_thr",
            "val_macro_f1@0.5",
            "val_macro_f1",
        ]:
            v = m.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return float("-inf")

    best = max(payload["runs"], key=score) if payload["runs"] else None
    if best is not None:
        write_json(base / "best.json", {"run_id": best["run_id"], "run_dir": best["run_dir"]})

