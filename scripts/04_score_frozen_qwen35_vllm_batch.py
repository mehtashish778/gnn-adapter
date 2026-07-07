#!/usr/bin/env python3
"""Batch frozen Qwen3.5 JSON scoring via vLLM for NIH (or any canonical label JSON)."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import VLM_LABELS, normalize_path, read_jsonl, require_cuda_device, write_json
from qwen35_common import (
    DEFAULT_VARIANT,
    ensure_qwen35_snapshot,
    model_root_for_variant,
    normalize_variant,
    vlm_output_dir_for_variant,
)
from qwen35_vllm import Qwen35VllmScorer


def load_canonical_rows(path: Path, max_samples: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def path_shard(path: str, num_shards: int) -> int:
    key = normalize_path(path).encode("utf-8")
    return zlib.adler32(key) % num_shards


def shard_path(out_dir: Path, worker_id: int, shard_id: int) -> Path:
    return out_dir / f"nih_vlm_shard_w{worker_id:02d}_{shard_id:04d}.jsonl"


def load_completed_paths(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for p in sorted(out_dir.glob("*.jsonl")):
        for row in read_jsonl(p):
            path = normalize_path(row.get("path", ""))
            if path and "error" not in row:
                done.add(path)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical_json",
        default="data/processed/multilabel/nih/canonical_labels.json",
    )
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output JSONL dir (default: data/outputs_vlm_qwen35_{variant}).",
    )
    parser.add_argument("--variant", choices=["2b", "4b"], default=DEFAULT_VARIANT)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Logical GPU index (use 0 when CUDA_VISIBLE_DEVICES pins one physical GPU).",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--request_batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size (set 2 for Qwen3.5-4B across 2 GPUs).",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"--shard_idx must be in [0, {args.num_shards})")

    variant = normalize_variant(args.variant)
    require_cuda_device(args.gpu_id)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir) if args.out_dir else vlm_output_dir_for_variant(variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_canonical_rows(Path(args.canonical_json), args.max_samples)
    done = load_completed_paths(out_dir) if args.resume else set()
    pending = [r for r in rows if normalize_path(r["path"]) not in done]
    if args.num_shards > 1:
        pending = [r for r in pending if path_shard(r["path"], args.num_shards) == args.shard_idx]

    print(
        {
            "backend": "vllm",
            "variant": variant,
            "total": len(rows),
            "pending": len(pending),
            "already_scored": len(done),
            "out_dir": str(out_dir),
            "gpu_id": args.gpu_id,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "request_batch_size": args.request_batch_size,
        }
    )

    if not pending:
        return

    model_path = args.model_dir if args.model_dir is not None else model_root_for_variant(variant)
    model_dir = ensure_qwen35_snapshot(model_path, variant=variant)

    # For 4B on 2×12 GB GPUs, recommend:
    # CUDA_VISIBLE_DEVICES=0,1 --tensor_parallel_size 2 --request_batch_size 1
    scorer = Qwen35VllmScorer(
        model_dir,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    worker_id = args.shard_idx if args.num_shards > 1 else 0
    shard_id = max(
        [
            int(p.stem.split("_")[-1])
            for p in out_dir.glob(f"nih_vlm_shard_w{worker_id:02d}_*.jsonl")
        ],
        default=-1,
    ) + 1
    buffer: list[dict] = []
    errors = 0
    desc = f"vLLM Qwen3.5-{variant} w{args.shard_idx}/{args.num_shards}"

    try:
        for start in tqdm(range(0, len(pending), args.request_batch_size), desc=desc):
            batch_rows = pending[start : start + args.request_batch_size]
            try:
                results = scorer.score_batch(batch_rows, image_root)
                for row, (probs, failed) in zip(batch_rows, results):
                    if failed:
                        errors += 1
                        buffer.append(
                            {
                                "path": row["path"],
                                "image_id": row.get("image_id"),
                                "error": "parse_failed",
                                "scores": {lbl: 0.5 for lbl in VLM_LABELS},
                            }
                        )
                    else:
                        buffer.append(
                            {
                                "path": row["path"],
                                "image_id": row.get("image_id"),
                                "scores": {lbl: float(probs[i]) for i, lbl in enumerate(VLM_LABELS)},
                            }
                        )
            except Exception as exc:
                for row in batch_rows:
                    errors += 1
                    buffer.append(
                        {
                            "path": row["path"],
                            "image_id": row.get("image_id"),
                            "error": str(exc),
                        }
                    )

            if len(buffer) >= args.shard_size:
                outp = shard_path(out_dir, worker_id, shard_id)
                with outp.open("w", encoding="utf-8") as f:
                    for rec in buffer:
                        f.write(json.dumps(rec) + "\n")
                print({"wrote_shard": str(outp), "n": len(buffer)})
                shard_id += 1
                buffer = []
    finally:
        scorer.shutdown()

    if buffer:
        outp = shard_path(out_dir, worker_id, shard_id)
        with outp.open("w", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec) + "\n")
        print({"wrote_shard": str(outp), "n": len(buffer)})

    write_json(
        out_dir / f"scoring_report_w{worker_id:02d}.json",
        {
            "backend": "vllm",
            "variant": variant,
            "pending_start": len(pending),
            "errors": errors,
            "label_order": VLM_LABELS,
            "gpu_id": args.gpu_id,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "request_batch_size": args.request_batch_size,
        },
    )
    print({"errors": errors, "out_dir": str(out_dir), "variant": variant, "shard_idx": args.shard_idx})


if __name__ == "__main__":
    main()
