#!/usr/bin/env python3
import json
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, Qwen2VLForConditionalGeneration


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Edema",
    "Consolidation",
    "No Finding",
]

MODEL_PATH = Path("data/hf_cache/models--Qwen--Qwen2-VL-2B-Instruct")
EDGE_INDEX_PATH = Path("data/processed/graph/edge_index.json")
EDGE_WEIGHT_PATH = Path("data/processed/graph/edge_weight.json")
SCRIPT_DIR = Path(__file__).resolve().parent


def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


def soft_clamp_prob(p: float) -> float:
    return max(0.0, min(1.0, float(p)))


def load_thresholds_for_model(model_id: str) -> List[float]:
    thresh_path = resolve_thresholds_path(model_id)
    if thresh_path is None or not thresh_path.exists():
        return [0.5] * len(LABELS)
    with thresh_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    thresholds = payload.get("thresholds", [0.5] * len(LABELS))
    if len(thresholds) != len(LABELS):
        return [0.5] * len(LABELS)
    return [float(x) for x in thresholds]


def resolve_hf_model_dir(model_root: Path) -> Path:
    # Hugging Face cache layout: models--*/snapshots/<revision>/...
    if (model_root / "config.json").exists():
        return model_root
    snapshots_dir = model_root / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Model path missing config and snapshots: {model_root}")
    snapshot_dirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshots found under: {snapshots_dir}")
    resolved = snapshot_dirs[-1]
    if not (resolved / "config.json").exists():
        raise FileNotFoundError(f"Snapshot missing config.json: {resolved}")
    return resolved


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(model_id: str, protocol: str) -> Optional[Path]:
    base = Path("data/processed/experiments") / model_id / protocol
    best_ptr = read_json(base / "best.json")
    if best_ptr and best_ptr.get("run_dir"):
        p = Path(best_ptr["run_dir"])
        if p.exists():
            return p
    latest_ptr = read_json(base / "latest.json")
    if latest_ptr and latest_ptr.get("run_dir"):
        p = Path(latest_ptr["run_dir"])
        if p.exists():
            return p
    return None


def resolve_checkpoint(
    model_id: str,
    protocol: str,
    legacy_candidates: List[Path],
    *,
    filename: str = "best_checkpoint.pt",
) -> Optional[Path]:
    run_dir = resolve_run_dir(model_id, protocol)
    if run_dir is not None:
        p = run_dir / filename
        if p.exists():
            return p
    for c in legacy_candidates:
        if c.exists():
            return c
    return None


def resolve_thresholds_path(model_id: str) -> Optional[Path]:
    # Prefer calibrated thresholds from the same model.
    run_dir = resolve_run_dir(model_id, "calibrated4way")
    if run_dir is not None:
        p = run_dir / "per_class_thresholds.json"
        if p.exists():
            return p
    legacy_map = {
        "vlm_mlp": [Path("data/processed/experiments/mlp_calibrated/per_class_thresholds.json")],
        "gnn07_label_residual": [Path("data/processed/experiments/gnn_calibrated/per_class_thresholds.json")],
        "gnn12_clip_vlm_homo": [Path("data/processed/experiments/clip_vlm_gnn_calibrated4way/per_class_thresholds.json")],
        "gnn13_clip_bipartite": [Path("data/processed/experiments/bipartite_clip_gnn_calibrated4way/per_class_thresholds.json")],
    }
    legacy = legacy_map.get(model_id, []) + [Path("data/processed/experiments/thresholds/per_class_thresholds.json")]
    for p in legacy:
        if p.exists():
            return p
    return None


def load_script_module(module_name: str, file_name: str):
    path = SCRIPT_DIR / "scripts" / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    if str(SCRIPT_DIR / "scripts") not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def build_adj(num_nodes: int, edge_index: List[List[int]], edge_weight: List[float]) -> torch.Tensor:
    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for s, t, w in zip(edge_index[0], edge_index[1], edge_weight):
        a[int(s), int(t)] = float(w)
    a = a + torch.eye(num_nodes, dtype=torch.float32)
    deg = a.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return a / deg


def extract_json_dict(text: str) -> Dict[str, float]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON object.")
    payload = json.loads(match.group(0))
    out = {}
    for label in LABELS:
        out[label] = soft_clamp_prob(payload.get(label, 0.0))
    return out


class MLPResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualLabelGNN(nn.Module):
    def __init__(self, hidden_dim: int, alpha: float):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, probs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.stack([logits, probs], dim=-1)  # B,C,2
        x = torch.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)  # B,C
        x = torch.matmul(x, adj.T)
        return logits + self.alpha * x


class InferenceEngine:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

        self.thresholds = {
            "plain": [0.5] * len(LABELS),
            "vlm_mlp": load_thresholds_for_model("vlm_mlp"),
            "gnn07_label_residual": load_thresholds_for_model("gnn07_label_residual"),
            "gnn12_clip_vlm_homo": load_thresholds_for_model("gnn12_clip_vlm_homo"),
            "gnn13_clip_bipartite": load_thresholds_for_model("gnn13_clip_bipartite"),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = resolve_hf_model_dir(MODEL_PATH)
        self.mlp_ckpt_path = resolve_checkpoint(
            "vlm_mlp",
            "default",
            legacy_candidates=[
                Path("data/processed/experiments/baseline_mlp/best_checkpoint.pt"),
                Path("data/processed/experiments/multilabel_adapter/mlp_residual_best.pt"),
            ],
        )
        self.gnn_ckpt_path = resolve_checkpoint(
            "gnn07_label_residual",
            "default",
            legacy_candidates=[
                Path("data/processed/experiments/gnn_adapter/best_checkpoint.pt"),
            ],
        )
        self.gnn12_ckpt_path = resolve_checkpoint(
            "gnn12_clip_vlm_homo",
            "default",
            legacy_candidates=[Path("data/processed/experiments/clip_vlm_gnn_adapter/best_checkpoint.pt")],
        )
        self.gnn13_ckpt_path = resolve_checkpoint(
            "gnn13_clip_bipartite",
            "default",
            legacy_candidates=[Path("data/processed/experiments/bipartite_clip_gnn_adapter/best_checkpoint.pt")],
        )

        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=model_dtype,
            local_files_only=True,
        ).to(self.device)
        self.base_model.eval()

        self.mlp_input_dim = None
        self.mlp_model = self._load_mlp_model()
        self.gnn_model, self.adj = self._load_gnn_model_and_adj()
        self.clip_processor = None
        self.clip_model = None
        self.gnn12_model = None
        self.gnn12_adj = None
        self.gnn13_model = None
        self.gnn13_edge_mode = "all"
        self.gnn13_vlm_tau = 0.5
        self._load_clip_adapters()

    def _ensure_clip_backbone(self):
        if self.clip_model is not None and self.clip_processor is not None:
            return
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()

    def _compute_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        self._ensure_clip_backbone()
        inputs = self.clip_processor(images=[image], return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, dtype=torch.float32)
        with torch.no_grad():
            feat = self.clip_model.get_image_features(pixel_values=pv)
        if not torch.is_tensor(feat):
            feat = feat.pooler_output
        return feat

    def _load_mlp_model(self):
        if self.mlp_ckpt_path is None or not self.mlp_ckpt_path.exists():
            return None
        state = torch.load(self.mlp_ckpt_path, map_location="cpu")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        k0 = "net.0.weight" if "net.0.weight" in state else "0.weight"
        k3 = "net.3.weight" if "net.3.weight" in state else "3.weight"
        hidden_dim = int(state[k0].shape[0])
        input_dim = int(state[k0].shape[1])
        output_dim = int(state[k3].shape[0])
        self.mlp_input_dim = input_dim
        model = MLPResidual(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model

    def _load_gnn_model_and_adj(self):
        if self.gnn_ckpt_path is None or not self.gnn_ckpt_path.exists() or not EDGE_INDEX_PATH.exists() or not EDGE_WEIGHT_PATH.exists():
            return None, None

        with EDGE_INDEX_PATH.open("r", encoding="utf-8") as f:
            edge_index = json.load(f)
        with EDGE_WEIGHT_PATH.open("r", encoding="utf-8") as f:
            edge_weight = json.load(f)

        state = torch.load(self.gnn_ckpt_path, map_location="cpu")
        hidden_dim = int(state["fc1.weight"].shape[0])
        alpha = 0.5
        model = ResidualLabelGNN(hidden_dim=hidden_dim, alpha=alpha)
        model.load_state_dict(state)
        model.to(self.device).eval()

        adj = build_adj(len(LABELS), edge_index, edge_weight).to(self.device)
        return model, adj

    def _load_clip_adapters(self):
        # GNN12
        if self.gnn12_ckpt_path is not None and self.gnn12_ckpt_path.exists() and EDGE_INDEX_PATH.exists() and EDGE_WEIGHT_PATH.exists():
            mod12 = load_script_module("train_clip_vlm_gnn_adapter", "12_train_clip_vlm_gnn_adapter.py")
            ckpt = torch.load(self.gnn12_ckpt_path, map_location="cpu")
            state = ckpt.get("adapter_state_dict", ckpt)
            hp = ckpt.get("adapter_hparams", {})
            hidden_dim = int(hp.get("hidden_dim", 64))
            gnn_layers = int(hp.get("gnn_layers", 2))
            alpha = float(hp.get("alpha", 0.5))
            clip_dim = int(hp.get("clip_dim", 512))
            model = mod12.ClipVlmGraphAdapter(
                clip_dim=clip_dim,
                num_labels=len(LABELS),
                hidden_dim=hidden_dim,
                gnn_layers=gnn_layers,
                alpha=alpha,
            )
            model.load_state_dict(state)
            self.gnn12_model = model.to(self.device).eval()
            with EDGE_INDEX_PATH.open("r", encoding="utf-8") as f:
                edge_index = json.load(f)
            with EDGE_WEIGHT_PATH.open("r", encoding="utf-8") as f:
                edge_weight = json.load(f)
            self.gnn12_adj = build_adj(len(LABELS), edge_index, edge_weight).to(self.device)

        # GNN13
        if self.gnn13_ckpt_path is not None and self.gnn13_ckpt_path.exists():
            mod13 = load_script_module("train_bipartite_gnn_adapter", "13_train_bipartite_gnn_adapter.py")
            ckpt = torch.load(self.gnn13_ckpt_path, map_location="cpu")
            state = ckpt.get("adapter_state_dict", ckpt)
            hp = ckpt.get("adapter_hparams", {})
            clip_dim = int(hp.get("clip_dim", 512))
            object_feature_dim = int(hp.get("object_feature_dim", 512))
            hidden_dims = hp.get("gnn_hidden_dims", [512, 256])
            if isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(",") if x.strip()]
            mid_dim = hp.get("gnn_mid_dim", None)
            alpha = float(hp.get("alpha", 0.5))
            edge_mode = hp.get("edge_mode", "all")
            vlm_tau = float(hp.get("vlm_tau", 0.5))
            model = mod13.ClipObjectBipartiteGNN(
                clip_dim=clip_dim,
                object_feature_dim=object_feature_dim,
                num_attributes=len(LABELS),
                hidden_dims=hidden_dims,
                mid_dim=mid_dim,
                dropout=0.0,
                alpha=alpha,
            )
            model.load_state_dict(state)
            self.gnn13_model = model.to(self.device).eval()
            self.gnn13_edge_mode = edge_mode
            self.gnn13_vlm_tau = vlm_tau
            self._build_bipartite_edge_weights = mod13.build_bipartite_edge_weights

    def run_plain(self, image: Image.Image) -> Dict[str, float]:
        prompt = (
            "You are a chest X-ray classifier. Return ONLY valid JSON with these exact keys: "
            + ", ".join(LABELS)
            + ". Values must be probabilities between 0 and 1."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.base_model.generate(**inputs, max_new_tokens=192, do_sample=False)
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return extract_json_dict(decoded)

    def _apply_mlp(self, probs_plain: List[float]) -> Optional[List[float]]:
        if self.mlp_model is None:
            return None
        logits = [safe_logit(x) for x in probs_plain]
        if self.mlp_input_dim == len(LABELS):
            features = probs_plain
        elif self.mlp_input_dim == 2 * len(LABELS):
            features = logits + probs_plain
        else:
            # Fallback for unknown adapter feature shape: use probability-only features.
            features = probs_plain
        x = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.mlp_model(x)
            probs = torch.sigmoid(out).squeeze(0).cpu().tolist()
        return [soft_clamp_prob(v) for v in probs]

    def _apply_gnn(self, probs_plain: List[float]) -> Optional[List[float]]:
        if self.gnn_model is None or self.adj is None:
            return None
        logits = [safe_logit(x) for x in probs_plain]
        logits_t = torch.tensor([logits], dtype=torch.float32, device=self.device)
        probs_t = torch.tensor([probs_plain], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.gnn_model(logits_t, probs_t, self.adj)
            probs = torch.sigmoid(out).squeeze(0).cpu().tolist()
        return [soft_clamp_prob(v) for v in probs]

    def _apply_gnn12(self, image: Image.Image, probs_plain: List[float]) -> Optional[List[float]]:
        if self.gnn12_model is None or self.gnn12_adj is None:
            return None
        clip_emb = self._compute_clip_embedding(image)
        logits = [safe_logit(x) for x in probs_plain]
        logits_t = torch.tensor([logits], dtype=torch.float32, device=self.device)
        probs_t = torch.tensor([probs_plain], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.gnn12_model(clip_emb, logits_t, probs_t, self.gnn12_adj)
            probs = torch.sigmoid(out).squeeze(0).cpu().tolist()
        return [soft_clamp_prob(v) for v in probs]

    def _apply_gnn13(self, image: Image.Image, probs_plain: List[float]) -> Optional[List[float]]:
        if self.gnn13_model is None:
            return None
        clip_emb = self._compute_clip_embedding(image)
        logits = [safe_logit(x) for x in probs_plain]
        logits_t = torch.tensor([logits], dtype=torch.float32, device=self.device)
        probs_t = torch.tensor([probs_plain], dtype=torch.float32, device=self.device)
        edge_w = self._build_bipartite_edge_weights(probs_t, self.gnn13_edge_mode, self.gnn13_vlm_tau).to(self.device)
        with torch.no_grad():
            out = self.gnn13_model(clip_emb, logits_t, probs_t, edge_w)
            probs = torch.sigmoid(out).squeeze(0).cpu().tolist()
        return [soft_clamp_prob(v) for v in probs]

    def infer_all(self, image: Image.Image):
        if image is None:
            raise gr.Error("Please upload an image.")

        image = image.convert("RGB")
        plain_dict = self.run_plain(image)
        plain_probs = [plain_dict[lbl] for lbl in LABELS]
        mlp_probs = self._apply_mlp(plain_probs)
        gnn07_probs = self._apply_gnn(plain_probs)
        gnn12_probs = self._apply_gnn12(image, plain_probs)
        gnn13_probs = self._apply_gnn13(image, plain_probs)

        rows = []
        for i, label in enumerate(LABELS):
            thr_plain = self.thresholds["plain"][i]
            thr_mlp = self.thresholds["vlm_mlp"][i]
            thr_gnn07 = self.thresholds["gnn07_label_residual"][i]
            thr_gnn12 = self.thresholds["gnn12_clip_vlm_homo"][i]
            thr_gnn13 = self.thresholds["gnn13_clip_bipartite"][i]

            def fmt_prob(v):
                return None if v is None else round(v, 4)

            def fmt_pred(v, thr):
                return None if v is None else int(v >= thr)

            rows.append(
                [
                    label,
                    round(plain_probs[i], 4),
                    int(plain_probs[i] >= thr_plain),
                    fmt_prob(None if mlp_probs is None else mlp_probs[i]),
                    fmt_pred(None if mlp_probs is None else mlp_probs[i], thr_mlp),
                    fmt_prob(None if gnn07_probs is None else gnn07_probs[i]),
                    fmt_pred(None if gnn07_probs is None else gnn07_probs[i], thr_gnn07),
                    fmt_prob(None if gnn12_probs is None else gnn12_probs[i]),
                    fmt_pred(None if gnn12_probs is None else gnn12_probs[i], thr_gnn12),
                    fmt_prob(None if gnn13_probs is None else gnn13_probs[i]),
                    fmt_pred(None if gnn13_probs is None else gnn13_probs[i], thr_gnn13),
                ]
            )

        plain_json = json.dumps(dict(zip(LABELS, [round(x, 6) for x in plain_probs])), indent=2)
        mlp_json = json.dumps(dict(zip(LABELS, [round(x, 6) for x in mlp_probs])), indent=2) if mlp_probs is not None else "{}"
        gnn07_json = json.dumps(dict(zip(LABELS, [round(x, 6) for x in gnn07_probs])), indent=2) if gnn07_probs is not None else "{}"
        gnn12_json = json.dumps(dict(zip(LABELS, [round(x, 6) for x in gnn12_probs])), indent=2) if gnn12_probs is not None else "{}"
        gnn13_json = json.dumps(dict(zip(LABELS, [round(x, 6) for x in gnn13_probs])), indent=2) if gnn13_probs is not None else "{}"
        return rows, plain_json, mlp_json, gnn07_json, gnn12_json, gnn13_json


def build_app():
    engine = InferenceEngine()
    with gr.Blocks(title="Qwen2-VL + Adapters Inference") as demo:
        gr.Markdown(
            "## Qwen2-VL Inference Comparison\n"
            "Upload one image to compare:\n"
            "- Plain `Qwen2-VL-2B-Instruct`\n"
            "- `VLMFeatureMLP` (adapter over VLM logits/probs)\n"
            "- `LabelGraphResidualGNN` (adapter over label graph)\n"
            "- `ClipVlmHomogeneousGNN`\n"
            "- `ClipBipartiteAttributeGNN`\n\n"
            "Note: each model uses its own calibrated thresholds when available."
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            table_output = gr.Dataframe(
                headers=[
                    "label",
                    "plain_prob",
                    "plain_pred",
                    "mlp_prob",
                    "mlp_pred",
                    "gnn07_prob",
                    "gnn07_pred",
                    "gnn12_prob",
                    "gnn12_pred",
                    "gnn13_prob",
                    "gnn13_pred",
                ],
                datatype=["str", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
                label="Per-label comparison",
                interactive=False,
            )

        with gr.Row():
            plain_json = gr.Code(label="Plain probabilities (JSON)", language="json")
            mlp_json = gr.Code(label="MLP probabilities (JSON)", language="json")
            gnn07_json = gr.Code(label="GNN07 probabilities (JSON)", language="json")
        with gr.Row():
            gnn12_json = gr.Code(label="GNN12 probabilities (JSON)", language="json")
            gnn13_json = gr.Code(label="GNN13 probabilities (JSON)", language="json")

        run_btn = gr.Button("Run Inference", variant="primary")
        run_btn.click(
            fn=engine.infer_all,
            inputs=[image_input],
            outputs=[table_output, plain_json, mlp_json, gnn07_json, gnn12_json, gnn13_json],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
