#!/usr/bin/env python3
"""Benchmark frozen Qwen3.5 JSON scoring: HuggingFace vs vLLM on N NIH images."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import VLM_LABELS, require_cuda_device
from qwen2vl_lora_common import (
    JSON_PROMPT,
    apply_chat_template_for_generation,
    build_user_messages,
    extract_json_dict,
    generate_row_json_probs,
    load_processor,
    open_image,
)
from qwen35_common import (
    DEFAULT_VARIANT,
    ensure_qwen35_snapshot,
    load_base_qwen35_model,
    model_root_for_variant,
    normalize_variant,
)


def load_rows(canonical_json: Path, max_samples: int) -> list[dict]:
    with canonical_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    return rows[:max_samples] if max_samples > 0 else rows


def parse_probs(text: str) -> tuple[list[float], bool]:
    try:
        parsed = extract_json_dict(text)
        return [parsed[lbl] for lbl in VLM_LABELS], False
    except (ValueError, json.JSONDecodeError):
        return [0.5] * len(VLM_LABELS), True


def resolve_attn_implementation(use_flash_attn: bool) -> str | None:
    if not use_flash_attn:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def bench_hf(
    rows: list[dict],
    image_root: Path,
    model_dir: Path,
    device,
    *,
    max_new_tokens: int,
    use_flash_attn: bool,
    warmup: int,
) -> dict[str, Any]:
    attn_impl = resolve_attn_implementation(use_flash_attn)
    t0 = time.perf_counter()
    model = load_base_qwen35_model(
        model_dir,
        device,
        gradient_checkpointing=False,
        attn_implementation=attn_impl,
    )
    processor = load_processor(model_dir, local_files_only=True)
    model.eval()
    load_s = time.perf_counter() - t0

    for row in rows[:warmup]:
        generate_row_json_probs(
            model,
            processor,
            {"path": row["path"]},
            image_root,
            device,
            max_new_tokens=max_new_tokens,
        )

    probs_out: list[list[float]] = []
    parse_failures = 0
    t1 = time.perf_counter()
    for row in rows:
        probs, failed = generate_row_json_probs(
            model,
            processor,
            {"path": row["path"]},
            image_root,
            device,
            max_new_tokens=max_new_tokens,
        )
        probs_out.append(probs)
        if failed:
            parse_failures += 1
    infer_s = time.perf_counter() - t1

    del model
    try:
        import torch

        if device.type == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass

    n = len(rows)
    return {
        "backend": "hf",
        "load_s": load_s,
        "infer_s": infer_s,
        "n": n,
        "images_per_s": n / infer_s if infer_s > 0 else 0.0,
        "s_per_image": infer_s / n if n else 0.0,
        "parse_failures": parse_failures,
        "probs": probs_out,
    }


def build_vllm_input(processor, image_path: Path, prompt: str) -> dict[str, Any]:
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


def bench_vllm(
    rows: list[dict],
    image_root: Path,
    model_dir: Path,
    *,
    max_new_tokens: int,
    request_batch_size: int,
    gpu_memory_utilization: float,
    warmup: int,
) -> dict[str, Any]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    from common_multilabel import resolve_dataset_image_path

    t0 = time.perf_counter()
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        limit_mm_per_prompt={"video": 0},
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
    )
    processor = load_processor(model_dir, local_files_only=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    load_s = time.perf_counter() - t0

    def run_batch(batch_rows: list[dict]) -> tuple[list[list[float]], int]:
        inputs = []
        for row in batch_rows:
            img_path = resolve_dataset_image_path(image_root, row["path"])
            inputs.append(build_vllm_input(processor, img_path, JSON_PROMPT))
        outputs = llm.generate(inputs, sampling)
        probs_batch: list[list[float]] = []
        failures = 0
        for out in outputs:
            probs, failed = parse_probs(out.outputs[0].text)
            probs_batch.append(probs)
            if failed:
                failures += 1
        return probs_batch, failures

    if warmup > 0:
        run_batch(rows[: min(warmup, len(rows))])

    probs_out: list[list[float]] = []
    parse_failures = 0
    t1 = time.perf_counter()
    for start in range(0, len(rows), request_batch_size):
        chunk = rows[start : start + request_batch_size]
        probs_chunk, failures = run_batch(chunk)
        probs_out.extend(probs_chunk)
        parse_failures += failures
    infer_s = time.perf_counter() - t1

    del llm
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    n = len(rows)
    return {
        "backend": "vllm",
        "load_s": load_s,
        "infer_s": infer_s,
        "n": n,
        "request_batch_size": request_batch_size,
        "images_per_s": n / infer_s if infer_s > 0 else 0.0,
        "s_per_image": infer_s / n if n else 0.0,
        "parse_failures": parse_failures,
        "probs": probs_out,
    }


def compare_probs(hf_probs: list[list[float]], vllm_probs: list[list[float]]) -> dict[str, float]:
    if not hf_probs or not vllm_probs:
        return {}
    n = min(len(hf_probs), len(vllm_probs))
    abs_diffs: list[float] = []
    for i in range(n):
        for j in range(len(VLM_LABELS)):
            abs_diffs.append(abs(hf_probs[i][j] - vllm_probs[i][j]))
    return {
        "mean_abs_prob_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
        "max_abs_prob_diff": max(abs_diffs) if abs_diffs else 0.0,
        "n_compared": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical_json",
        default="data/processed/multilabel/nih/canonical_labels.json",
    )
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--variant", choices=["2b", "4b"], default=DEFAULT_VARIANT)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--request_batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--backend",
        choices=["both", "hf", "vllm"],
        default="both",
        help="Run HF, vLLM, or both sequentially.",
    )
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    variant = normalize_variant(args.variant)
    device = require_cuda_device(args.gpu_id)
    image_root = Path(args.image_root)
    rows = load_rows(Path(args.canonical_json), args.max_samples)
    if not rows:
        raise SystemExit("No rows to benchmark.")

    model_dir = ensure_qwen35_snapshot(model_root_for_variant(variant), variant=variant)
    print({"variant": variant, "n": len(rows), "model_dir": str(model_dir), "gpu_id": args.gpu_id})

    results: dict[str, Any] = {"variant": variant, "n": len(rows), "backends": {}}

    if args.backend in ("both", "hf"):
        print("\n=== HuggingFace ===")
        hf = bench_hf(
            rows,
            image_root,
            model_dir,
            device,
            max_new_tokens=args.max_new_tokens,
            use_flash_attn=args.use_flash_attn,
            warmup=args.warmup,
        )
        hf_report = {k: v for k, v in hf.items() if k != "probs"}
        print(hf_report)
        results["backends"]["hf"] = hf_report

    if args.backend in ("both", "vllm"):
        print("\n=== vLLM ===")
        vllm = bench_vllm(
            rows,
            image_root,
            model_dir,
            max_new_tokens=args.max_new_tokens,
            request_batch_size=args.request_batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            warmup=args.warmup,
        )
        vllm_report = {k: v for k, v in vllm.items() if k != "probs"}
        print(vllm_report)
        results["backends"]["vllm"] = vllm_report

    if args.backend == "both" and "hf" in results["backends"] and "vllm" in results["backends"]:
        parity = compare_probs(hf["probs"], vllm["probs"])
        hf_ips = results["backends"]["hf"]["images_per_s"]
        vllm_ips = results["backends"]["vllm"]["images_per_s"]
        speedup = vllm_ips / hf_ips if hf_ips > 0 else 0.0
        results["parity"] = parity
        results["speedup_vllm_over_hf"] = speedup
        print("\n=== Comparison ===")
        print({"speedup_vllm_over_hf": round(speedup, 2), **parity})

    out_path = Path(args.out_json) if args.out_json else Path(
        f"data/outputs_vlm_qwen35_{variant}/benchmark_vllm_vs_hf_{len(rows)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print({"wrote": str(out_path)})


if __name__ == "__main__":
    main()
