#!/usr/bin/env python3
import argparse
from pathlib import Path

from common_multilabel import VLM_LABELS, normalize_path, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description="Align VLM JSONL rows with canonical multi-label labels.")
    parser.add_argument("--canonical_json", default="data/processed/multilabel/canonical_labels.json")
    parser.add_argument("--vlm_dir", default="data/outputs_vlm_corrected copy")
    parser.add_argument("--out_json", default="data/processed/multilabel/aligned_vlm_targets.json")
    args = parser.parse_args()

    canonical = write_safe_read(Path(args.canonical_json))
    label_map = {normalize_path(r["path"]): r for r in canonical["rows"]}

    aligned = []
    invalid_rows = []
    total = 0
    matched = 0
    unmatched = 0

    for shard in sorted(Path(args.vlm_dir).glob("*.jsonl")):
        for row in read_jsonl(shard):
            total += 1
            path = normalize_path(row.get("path", ""))
            if "error" in row:
                invalid_rows.append(
                    {
                        "path": path,
                        "image_id": row.get("image_id"),
                        "error": row.get("error"),
                        "retry_attempts": row.get("retry_attempts"),
                    }
                )
                continue
            if path not in label_map:
                unmatched += 1
                continue
            scores = row.get("scores", {})
            x_probs = [float(scores.get(lbl, 0.0)) for lbl in VLM_LABELS]
            x_logits = [safe_logit(p) for p in x_probs]
            target = label_map[path]
            aligned.append(
                {
                    "path": path,
                    "image_id": row.get("image_id"),
                    "patient_id": target["patient_id"],
                    "x_probs": x_probs,
                    "x_logits": x_logits,
                    "y_true": [int(target["labels"][lbl]) for lbl in VLM_LABELS],
                    "y_mask": [int(target["mask"][lbl]) for lbl in VLM_LABELS],
                }
            )
            matched += 1

    payload = {
        "meta": {
            "label_order": VLM_LABELS,
            "total_vlm_rows": total,
            "matched_rows": matched,
            "unmatched_rows": unmatched,
            "invalid_rows": len(invalid_rows),
        },
        "rows": aligned,
        "invalid_rows": invalid_rows,
    }
    write_json(Path(args.out_json), payload)
    write_json(Path("data/processed/multilabel/alignment_report.json"), payload["meta"])
    print(payload["meta"])


def write_safe_read(path: Path):
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_logit(p: float, eps: float = 1e-6) -> float:
    import math

    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))


if __name__ == "__main__":
    main()
