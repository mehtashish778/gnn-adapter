"""
Shared utilities for Qwen3.5-2B / Qwen3.5-4B LoRA training and scoring (CheXpert multi-label).

Re-exports model-agnostic helpers from qwen2vl_lora_common; overrides model class and hub paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from qwen2vl_lora_common import (
    CLS_PROMPT,
    DEFAULT_HF_CACHE,
    JSON_PROMPT,
    LORA_TARGET_MODULES,
    GpuTimer,
    apply_lora,
    build_sft_batch,
    build_user_messages,
    count_cls_trainable_params,
    count_lora_adapter_params,
    count_trainable_params,
    decode_generated_json,
    ensure_model_snapshot,
    extract_json_dict,
    generate_probs_from_rows,
    generate_row_json_probs,
    get_lora_config,
    load_processor,
    load_split_rows,
    open_image,
    peak_gpu_memory_mb,
    pool_last_token_hidden,
    prepare_inputs,
    qwen_hidden_size,
    remove_corrupt_weight_files,
    resolve_hf_model_dir,
    safe_logit,
    soft_clamp_prob,
    target_json_from_row,
    validate_model_weights,
)

QN35_VARIANTS: Dict[str, Dict[str, str]] = {
    "2b": {
        "hub_id": "Qwen/Qwen3.5-2B",
        "cache_name": "models--Qwen--Qwen3.5-2B",
    },
    "4b": {
        "hub_id": "Qwen/Qwen3.5-4B",
        "cache_name": "models--Qwen--Qwen3.5-4B",
    },
}

DEFAULT_VARIANT = "2b"
DEFAULT_HUB_ID = QN35_VARIANTS[DEFAULT_VARIANT]["hub_id"]
DEFAULT_MODEL_ROOT = Path("data/hf_cache") / QN35_VARIANTS[DEFAULT_VARIANT]["cache_name"]


def normalize_variant(variant: str) -> str:
    v = (variant or DEFAULT_VARIANT).strip().lower()
    if v not in QN35_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(QN35_VARIANTS)}")
    return v


def hub_id_for_variant(variant: str) -> str:
    return QN35_VARIANTS[normalize_variant(variant)]["hub_id"]


def model_root_for_variant(variant: str) -> Path:
    cache_name = QN35_VARIANTS[normalize_variant(variant)]["cache_name"]
    return Path("data/hf_cache") / cache_name


def vlm_output_dir_for_variant(variant: str) -> Path:
    return Path(f"data/outputs_vlm_qwen35_{normalize_variant(variant)}")


def default_lora_cls_model_id(variant: str) -> str:
    return f"qwen35_{normalize_variant(variant)}_lora_r16"


def default_lora_sft_model_id(variant: str) -> str:
    return f"qwen35_{normalize_variant(variant)}_lora_r16_sft"


def ensure_qwen35_snapshot(
    model_path: str | Path | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    cache_dir: Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Download/repair Qwen3.5 weights for the given variant (2b or 4b)."""
    v = normalize_variant(variant)
    root = Path(model_path) if model_path is not None else model_root_for_variant(v)
    hub_id = hub_id_for_variant(v)
    return ensure_model_snapshot(
        root,
        hub_id=hub_id,
        cache_dir=cache_dir or DEFAULT_HF_CACHE,
        allow_download=allow_download,
    )


def load_base_qwen35_model(
    model_dir: Path,
    device,
    dtype=None,
    *,
    local_files_only: bool = True,
    gradient_checkpointing: bool = False,
    attn_implementation: Optional[str] = None,
):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    if dtype is None:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if device.type == "cuda" else torch.float32
    if not validate_model_weights(model_dir):
        raise FileNotFoundError(
            f"Incomplete weights at {model_dir}. Call ensure_qwen35_snapshot() first."
        )
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_dir), **load_kwargs)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model.to(device)


def load_lora_model(
    model_dir: Path,
    adapter_dir: Path,
    device,
    *,
    causal_lm: bool = False,
    merge: bool = False,
    allow_download: bool = True,
    variant: str = DEFAULT_VARIANT,
):
    from peft import PeftModel

    model_dir = ensure_qwen35_snapshot(model_dir, variant=variant, allow_download=allow_download)
    base = load_base_qwen35_model(model_dir, device)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model


__all__ = [
    "CLS_PROMPT",
    "DEFAULT_HF_CACHE",
    "DEFAULT_HUB_ID",
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_VARIANT",
    "JSON_PROMPT",
    "LORA_TARGET_MODULES",
    "QN35_VARIANTS",
    "GpuTimer",
    "apply_lora",
    "build_sft_batch",
    "build_user_messages",
    "count_cls_trainable_params",
    "count_lora_adapter_params",
    "count_trainable_params",
    "decode_generated_json",
    "default_lora_cls_model_id",
    "default_lora_sft_model_id",
    "ensure_qwen35_snapshot",
    "extract_json_dict",
    "generate_probs_from_rows",
    "generate_row_json_probs",
    "get_lora_config",
    "hub_id_for_variant",
    "load_base_qwen35_model",
    "load_lora_model",
    "load_processor",
    "load_split_rows",
    "model_root_for_variant",
    "normalize_variant",
    "open_image",
    "peak_gpu_memory_mb",
    "pool_last_token_hidden",
    "prepare_inputs",
    "qwen_hidden_size",
    "remove_corrupt_weight_files",
    "resolve_hf_model_dir",
    "safe_logit",
    "soft_clamp_prob",
    "target_json_from_row",
    "validate_model_weights",
    "vlm_output_dir_for_variant",
]
