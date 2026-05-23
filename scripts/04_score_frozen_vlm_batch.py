#!/usr/bin/env python3
"""Batch frozen Qwen2-VL JSON scoring for NIH (or any canonical label JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_multilabel import VLM_LABELS, normalize_path, read_jsonl, require_cuda_device, write_json
from qwen2vl_lora_common import (
    DEFAULT_MODEL_ROOT,
    ensure_model_snapshot,
    generate_row_json_probs,
    load_base_qwen_model,
    load_processor,
)


def load_canonical_rows(path: Path, max_samples: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def shard_path(out_dir: Path, shard_id: int) -> Path:
    return out_dir / f"nih_vlm_shard_{shard_id:04d}.jsonl"


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
    parser.add_argument("--out_dir", default="data/outputs_vlm_nih")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    device = require_cuda_device(args.gpu_id)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_canonical_rows(Path(args.canonical_json), args.max_samples)
    done = load_completed_paths(out_dir) if args.resume else set()
    pending = [r for r in rows if normalize_path(r["path"]) not in done]
    print({"total": len(rows), "pending": len(pending), "already_scored": len(done)})

    if not pending:
        return

    model_dir = ensure_model_snapshot(args.model_dir)
    model = load_base_qwen_model(model_dir, device, gradient_checkpointing=False)
    processor = load_processor(model_dir, local_files_only=True)
    model.eval()

    shard_id = max(
        [int(p.stem.split("_")[-1]) for p in out_dir.glob("nih_vlm_shard_*.jsonl")],
        default=-1,
    ) + 1
    buffer: list[dict] = []
    errors = 0

    for row in tqdm(pending, desc="VLM NIH"):
        try:
            pseudo = {
                "path": row["path"],
                "image_id": row.get("image_id"),
            }
            probs, failed = generate_row_json_probs(model, processor, pseudo, image_root, device)
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
            outp = shard_path(out_dir, shard_id)
            with outp.open("w", encoding="utf-8") as f:
                for rec in buffer:
                    f.write(json.dumps(rec) + "\n")
            print({"wrote_shard": str(outp), "n": len(buffer)})
            shard_id += 1
            buffer = []

    if buffer:
        outp = shard_path(out_dir, shard_id)
        with outp.open("w", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec) + "\n")
        print({"wrote_shard": str(outp), "n": len(buffer)})

    write_json(
        out_dir / "scoring_report.json",
        {"pending_start": len(pending), "errors": errors, "label_order": VLM_LABELS},
    )
    print({"errors": errors, "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()
