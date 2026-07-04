#!/usr/bin/env python3
"""Batch frozen Qwen3.5 JSON scoring for NIH (or any canonical label JSON)."""

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
    generate_row_json_probs,
    load_base_qwen35_model,
    load_processor,
    model_root_for_variant,
    normalize_variant,
    vlm_output_dir_for_variant,
)


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


def resolve_attn_implementation(use_flash_attn: bool) -> str | None:
    if not use_flash_attn:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        print({"flash_attn": "not installed; falling back to sdpa"})
        return "sdpa"


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
    parser.add_argument(
        "--variant",
        choices=["2b", "4b"],
        default=DEFAULT_VARIANT,
        help="Qwen3.5 model size: 2b or 4b.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Local HF cache dir (default: data/hf_cache/models--Qwen--Qwen3.5-{variant}).",
    )
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split work across N workers (e.g. 2 = half on GPU0, half on GPU1).",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="This worker's shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Cap JSON generation length (Qwen3.5 pretty-JSON needs ~90-100 tokens).",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash_attention_2 if flash-attn is installed; else sdpa.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"--shard_idx must be in [0, {args.num_shards})")

    variant = normalize_variant(args.variant)
    device = require_cuda_device(args.gpu_id)
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
            "variant": variant,
            "total": len(rows),
            "pending": len(pending),
            "already_scored": len(done),
            "out_dir": str(out_dir),
            "gpu_id": args.gpu_id,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "max_new_tokens": args.max_new_tokens,
        }
    )

    if not pending:
        return

    model_path = args.model_dir if args.model_dir is not None else model_root_for_variant(variant)
    model_dir = ensure_qwen35_snapshot(model_path, variant=variant)
    attn_impl = resolve_attn_implementation(args.use_flash_attn)
    model = load_base_qwen35_model(
        model_dir,
        device,
        gradient_checkpointing=False,
        attn_implementation=attn_impl,
    )
    processor = load_processor(model_dir, local_files_only=True)
    model.eval()

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

    desc = f"Qwen3.5-{variant} NIH w{args.shard_idx}/{args.num_shards}"
    for row in tqdm(pending, desc=desc):
        try:
            pseudo = {
                "path": row["path"],
                "image_id": row.get("image_id"),
            }
            probs, failed = generate_row_json_probs(
                model,
                processor,
                pseudo,
                image_root,
                device,
                max_new_tokens=args.max_new_tokens,
            )
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

    if buffer:
        outp = shard_path(out_dir, worker_id, shard_id)
        with outp.open("w", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec) + "\n")
        print({"wrote_shard": str(outp), "n": len(buffer)})

    write_json(
        out_dir / f"scoring_report_w{worker_id:02d}.json",
        {
            "variant": variant,
            "pending_start": len(pending),
            "errors": errors,
            "label_order": VLM_LABELS,
            "gpu_id": args.gpu_id,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        },
    )
    print({"errors": errors, "out_dir": str(out_dir), "variant": variant, "shard_idx": args.shard_idx})


if __name__ == "__main__":
    main()
