"""
Shared utilities for Qwen2-VL LoRA training and scoring (CheXpert multi-label).
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from common_multilabel import VLM_LABELS, load_rows, resolve_dataset_image_path

DEFAULT_HUB_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MODEL_ROOT = Path("data/hf_cache/models--Qwen--Qwen2-VL-2B-Instruct")
DEFAULT_HF_CACHE = Path("data/hf_cache")

CLS_PROMPT = (
    "You are a chest X-ray classifier. Analyze the image and output a multi-label prediction."
)

JSON_PROMPT = (
    "You are a chest X-ray classifier. Return ONLY valid JSON with these exact keys: "
    + ", ".join(VLM_LABELS)
    + ". Values must be probabilities between 0 and 1."
)

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_hf_model_dir(model_root: Path) -> Path:
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


def _shard_paths_from_index(model_dir: Path) -> List[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.stat().st_size == 0:
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    names = sorted(set(index.get("weight_map", {}).values()))
    return [model_dir / n for n in names]


def validate_model_weights(model_dir: Path) -> bool:
    """Return True when all required weight shards are present and non-empty."""
    import os

    model_dir = Path(model_dir)
    single = model_dir / "model.safetensors"
    if single.is_file() and single.stat().st_size > 0:
        return True

    index_shards = _shard_paths_from_index(model_dir)
    if index_shards:
        if not all(p.is_file() and os.path.getsize(p) > 0 for p in index_shards):
            return False
        expected_total = _expected_sharded_bytes(model_dir)
        if expected_total:
            on_disk = sum(os.path.getsize(p) for p in index_shards)
            return on_disk >= int(expected_total * 0.9)
        return True

    glob_shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not glob_shards:
        return False

    match = re.search(r"-of-(\d+)\.safetensors$", glob_shards[0].name)
    if match:
        expected_n = int(match.group(1))
        if len(glob_shards) < expected_n:
            return False
    if not all(p.is_file() and os.path.getsize(p) > 0 for p in glob_shards):
        return False
    expected_total = _expected_sharded_bytes(model_dir)
    if expected_total:
        return sum(os.path.getsize(p) for p in glob_shards) >= int(expected_total * 0.9)
    return True


def _expected_sharded_bytes(model_dir: Path) -> Optional[int]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file() and index_path.stat().st_size > 0:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            return int(index.get("metadata", {}).get("total_size", 0)) or None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def remove_corrupt_weight_files(model_dir: Path) -> List[str]:
    """Delete empty or truncated weight/index files so hub download can re-fetch them."""
    import os

    removed: List[str] = []
    model_dir = Path(model_dir)
    expected_total = _expected_sharded_bytes(model_dir)

    for pattern in ("model*.safetensors", "model.safetensors.index.json", "vocab.json"):
        for p in model_dir.glob(pattern):
            if not p.is_file():
                continue
            size = os.path.getsize(p)
            if size == 0:
                p.unlink(missing_ok=True)
                removed.append(p.name)

    glob_shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if glob_shards and expected_total:
        on_disk = sum(os.path.getsize(p) for p in glob_shards)
        # Allow small overhead; truncated caches are far below total_size.
        if on_disk < int(expected_total * 0.9):
            for p in glob_shards:
                p.unlink(missing_ok=True)
                removed.append(f"{p.name} (truncated)")
            index_path = model_dir / "model.safetensors.index.json"
            if index_path.is_file():
                index_path.unlink(missing_ok=True)
                removed.append(index_path.name)

    return removed


def ensure_model_snapshot(
    model_path: str | Path = DEFAULT_MODEL_ROOT,
    *,
    hub_id: str = DEFAULT_HUB_ID,
    cache_dir: Path | None = None,
    allow_download: bool = True,
) -> Path:
    """
    Resolve a local snapshot with valid weights, downloading/repairing from the Hub if needed.

    Handles incomplete HF caches (e.g. empty model.safetensors.index.json) that cause
    JSONDecodeError in transformers.
    """
    root = Path(model_path)
    cache_dir = cache_dir or DEFAULT_HF_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    if root.exists():
        resolved = resolve_hf_model_dir(root)
        if validate_model_weights(resolved):
            return resolved
        removed = remove_corrupt_weight_files(resolved)
        if removed:
            print({"removed_corrupt_cache_files": removed, "dir": str(resolved)})

    if not allow_download:
        raise FileNotFoundError(
            f"Qwen2-VL weights missing or corrupt under {root}. "
            f"Re-download: huggingface-cli download {hub_id} --cache-dir {cache_dir}"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub") from exc

    print({"downloading_or_repairing": hub_id, "cache_dir": str(cache_dir)})
    snapshot_path = snapshot_download(
        repo_id=hub_id,
        cache_dir=str(cache_dir),
        resume_download=True,
        # Re-fetch any files removed after a partial/corrupt download.
        force_download=False,
    )
    resolved = Path(snapshot_path)
    if not validate_model_weights(resolved):
        raise RuntimeError(
            f"Model download finished but weights are still invalid at {resolved}. "
            f"Try deleting {root} and re-running."
        )
    print({"model_snapshot": str(resolved)})
    return resolved


def soft_clamp_prob(p: float) -> float:
    return max(0.0, min(1.0, float(p)))


def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


def extract_json_dict(text: str) -> Dict[str, float]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON object.")
    payload = json.loads(match.group(0))
    out: Dict[str, float] = {}
    for label in VLM_LABELS:
        out[label] = soft_clamp_prob(payload.get(label, 0.0))
    return out


def target_json_from_row(row: dict) -> str:
    """Ground-truth JSON for SFT: supervised labels as 0/1; mask=0 -> 0.0."""
    y = row["y_true"]
    m = row["y_mask"]
    payload = {lbl: float(y[i]) if m[i] else 0.0 for i, lbl in enumerate(VLM_LABELS)}
    return json.dumps(payload, separators=(",", ": "))


def build_sft_batch(processor, rows: Sequence[dict], image_root: Path, device) -> Tuple[dict, List[Image.Image]]:
    """
    Build a causal-LM SFT batch with labels masked on the user+vision prefix only.

    Prompt length must come from processor(text, images=...) — not text-only tokenization —
    so vision tokens are included in the masked prefix.
    """
    images = [open_image(image_root, r) for r in rows]
    full_texts: List[str] = []
    prompt_lens: List[int] = []
    for img, row in zip(images, rows):
        target = target_json_from_row(row)
        user_msgs = build_user_messages(img, JSON_PROMPT)
        conv = user_msgs + [{"role": "assistant", "content": target}]
        full_texts.append(processor.apply_chat_template(conv, tokenize=False))
        prompt_text = processor.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
        prompt_lens.append(int(prompt_inputs["input_ids"].shape[1]))

    if len(rows) == 1:
        batch = processor(text=[full_texts[0]], images=[images[0]], return_tensors="pt", padding=True)
    else:
        batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch, images


def decode_generated_json(processor, input_ids, output_ids) -> str:
    """Decode only tokens generated after the prompt (exclude vision+user prefix)."""
    prompt_len = int(input_ids.shape[1])
    new_ids = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_row_json_probs(model, processor, row: dict, image_root: Path, device) -> Tuple[List[float], bool]:
    """
    Run greedy JSON generation for one row. Returns (prob_list, parse_failed).
    """
    img = open_image(image_root, row)
    msgs = build_user_messages(img, JSON_PROMPT)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False)
    decoded = decode_generated_json(processor, inputs["input_ids"], out_ids)
    try:
        parsed = extract_json_dict(decoded)
        return [parsed[lbl] for lbl in VLM_LABELS], False
    except (ValueError, json.JSONDecodeError):
        return [0.5] * len(VLM_LABELS), True


def generate_probs_from_rows(
    model,
    processor,
    rows: Sequence[dict],
    image_root: Path,
    device,
    batch_size: int = 1,
) -> Tuple["torch.Tensor", int]:
    """Batch wrapper (batch_size>1 still generates one image at a time)."""
    import torch

    probs_list: List[List[float]] = []
    parse_failures = 0
    model.eval()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        for row in chunk:
            probs, failed = generate_row_json_probs(model, processor, row, image_root, device)
            probs_list.append(probs)
            if failed:
                parse_failures += 1
    return torch.tensor(probs_list, dtype=torch.float32), parse_failures


def build_user_messages(image: Image.Image, prompt: str) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lora_config(*, rank: int, causal_lm: bool = False):
    from peft import LoraConfig

    kwargs: Dict[str, Any] = {
        "r": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": LORA_TARGET_MODULES,
    }
    if causal_lm:
        kwargs["task_type"] = "CAUSAL_LM"
    return LoraConfig(**kwargs)


def load_processor(model_dir: Path, *, local_files_only: bool = True):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir), local_files_only=local_files_only)


def load_base_qwen_model(
    model_dir: Path,
    device,
    dtype=None,
    *,
    local_files_only: bool = True,
    gradient_checkpointing: bool = False,
):
    import torch
    from transformers import Qwen2VLForConditionalGeneration

    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    if not validate_model_weights(model_dir):
        raise FileNotFoundError(
            f"Incomplete weights at {model_dir}. Call ensure_model_snapshot() first."
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model.to(device)


def apply_lora(model, *, rank: int, causal_lm: bool = False):
    from peft import get_peft_model

    cfg = get_lora_config(rank=rank, causal_lm=causal_lm)
    return get_peft_model(model, cfg)


def load_lora_model(
    model_dir: Path,
    adapter_dir: Path,
    device,
    *,
    causal_lm: bool = False,
    merge: bool = False,
    allow_download: bool = True,
):
    from peft import PeftModel

    model_dir = ensure_model_snapshot(model_dir, allow_download=allow_download)
    base = load_base_qwen_model(model_dir, device)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model


def prepare_inputs(processor, messages: List[dict], images: List[Image.Image], device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def pool_last_token_hidden(model, inputs: dict) -> "torch.Tensor":
    import torch

    out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = out.hidden_states[-1].float()
    attn = inputs.get("attention_mask")
    if attn is None:
        return hidden[:, -1, :]
    last_idx = attn.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, last_idx, :]


def open_image(image_root: Path, row: dict) -> Image.Image:
    path = resolve_dataset_image_path(image_root, row["path"])
    with Image.open(path) as im:
        return im.convert("RGB")


class GpuTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.elapsed = 0.0

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.t0
        return self.elapsed


def peak_gpu_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        pass
    return 0.0


def load_split_rows(
    train_json: str,
    val_json: str,
    test_json: str,
) -> Tuple[List[dict], List[dict], List[dict]]:
    return (
        load_rows(Path(train_json)),
        load_rows(Path(val_json)),
        load_rows(Path(test_json)),
    )
