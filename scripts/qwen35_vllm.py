"""vLLM inference helpers for frozen Qwen3.5 JSON scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common_multilabel import VLM_LABELS, resolve_dataset_image_path
from qwen2vl_lora_common import (
    JSON_PROMPT,
    apply_chat_template_for_generation,
    extract_json_dict,
    load_processor,
)


def parse_generation_probs(text: str) -> tuple[list[float], bool]:
    try:
        parsed = extract_json_dict(text)
        return [parsed[lbl] for lbl in VLM_LABELS], False
    except (ValueError, json.JSONDecodeError):
        return [0.5] * len(VLM_LABELS), True


def build_vllm_input(processor, image_path: Path, prompt: str = JSON_PROMPT) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = apply_chat_template_for_generation(processor, messages, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    inp: dict[str, Any] = {"prompt": text}
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    if mm_data:
        inp["multi_modal_data"] = mm_data
    if video_kwargs:
        inp["mm_processor_kwargs"] = video_kwargs
    return inp


class Qwen35VllmScorer:
    def __init__(
        self,
        model_dir: Path,
        *,
        max_new_tokens: int = 128,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
    ):
        """
        tensor_parallel_size:
          - 1  → single-GPU engine (default; works for 2B on 12 GB)
          - 2+ → model sharded across multiple GPUs (e.g., 4B on 2×12 GB)
        """
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Avoid inductor calling nvcc during TP worker inference (WSL permission issues).
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

        from vllm import LLM, SamplingParams

        # Prefix cache costs extra VRAM; disable for TP>1 on tight 12 GB cards.
        enable_prefix_caching = tensor_parallel_size <= 1

        self.llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"video": 0},
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=tensor_parallel_size > 1,
        )
        self.processor = load_processor(model_dir, local_files_only=True)
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def score_batch(
        self,
        rows: list[dict],
        image_root: Path,
    ) -> list[tuple[list[float], bool]]:
        inputs = [
            build_vllm_input(
                self.processor,
                resolve_dataset_image_path(image_root, row["path"]),
            )
            for row in rows
        ]
        outputs = self.llm.generate(inputs, self.sampling)
        return [parse_generation_probs(out.outputs[0].text) for out in outputs]

    def shutdown(self) -> None:
        del self.llm
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
