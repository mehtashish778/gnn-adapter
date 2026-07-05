#!/usr/bin/env python3
"""Apply VLM scores from an aligned JSON onto a fixed reference split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common_multilabel import normalize_path, write_json


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep reference train/val/test paths; replace x_probs/x_logits from aligned VLM JSON."
    )
    parser.add_argument(
        "--reference_splits_dir",
        default="data/processed/splits",
        help="Qwen2 split manifests (train/val/test_rows.json).",
    )
    parser.add_argument(
        "--aligned_json",
        default="data/processed/multilabel/aligned_vlm_targets_qwen35_2b_qwen2subset.json",
        help="Aligned VLM scores to inject (e.g. Qwen3.5).",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/splits/qwen35_qwen2_splits",
        help="Output split directory with Qwen2 paths + new VLM scores.",
    )
    parser.add_argument(
        "--on_missing",
        choices=["drop", "fail"],
        default="drop",
        help="If a reference path has no VLM score, drop the row or abort.",
    )
    args = parser.parse_args()

    aligned = json.loads(Path(args.aligned_json).read_text(encoding="utf-8"))
    score_by_path = {normalize_path(r["path"]): r for r in aligned["rows"]}

    ref_dir = Path(args.reference_splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "reference_splits_dir": str(ref_dir),
        "aligned_json": args.aligned_json,
        "on_missing": args.on_missing,
        "splits": {},
        "missing_paths": [],
    }

    for split in ("train", "val", "test"):
        ref_rows = load_rows(ref_dir / f"{split}_rows.json")
        out_rows = []
        missing = []
        for row in ref_rows:
            path = normalize_path(row["path"])
            src = score_by_path.get(path)
            if src is None:
                missing.append(path)
                if args.on_missing == "fail":
                    raise SystemExit(f"Missing VLM score for {path} ({split})")
                continue
            out_rows.append(
                {
                    **row,
                    "x_probs": src["x_probs"],
                    "x_logits": src["x_logits"],
                }
            )
        write_json(out_dir / f"{split}_rows.json", {"rows": out_rows})
        report["splits"][split] = {
            "reference": len(ref_rows),
            "written": len(out_rows),
            "dropped_missing_vlm": len(missing),
        }
        report["missing_paths"].extend(missing)

    report["missing_paths"] = sorted(set(report["missing_paths"]))
    report["total_reference"] = sum(v["reference"] for v in report["splits"].values())
    report["total_written"] = sum(v["written"] for v in report["splits"].values())
    report["total_dropped"] = report["total_reference"] - report["total_written"]
    write_json(out_dir / "split_manifest_v1.json", report)
    print(report)


if __name__ == "__main__":
    main()
