#!/usr/bin/env python3
"""Build canonical multi-label targets from NIH ChestX-ray14 Data_Entry CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common_multilabel import NIH_FINDINGS_FOR_VLM, VLM_LABELS, normalize_path, write_json


def build_image_index_map(nih_root: Path) -> dict[str, str]:
    """Map Image Index filename -> relative path under data/raw (nih_chestxray14/...)."""
    index: dict[str, str] = {}
    for shard in sorted(nih_root.glob("images_*")):
        img_dir = shard / "images"
        if not img_dir.is_dir():
            continue
        for p in img_dir.glob("*.png"):
            rel = normalize_path(f"nih_chestxray14/{shard.name}/images/{p.name}")
            index[p.name] = rel
    return index


def parse_nih_labels(finding_labels: str) -> set[str]:
    text = (finding_labels or "").strip()
    if not text:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nih_csv",
        default="data/raw/nih_chestxray14/Data_Entry_2017.csv",
    )
    parser.add_argument("--nih_root", default="data/raw/nih_chestxray14")
    parser.add_argument(
        "--out_json",
        default="data/processed/multilabel/nih/canonical_labels.json",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()

    nih_root = Path(args.nih_root)
    csv_path = Path(args.nih_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing NIH CSV: {csv_path}")

    print({"indexing_images": str(nih_root)})
    image_map = build_image_index_map(nih_root)
    print({"indexed_png_files": len(image_map)})

    out_rows = []
    missing_images = 0
    skipped_no_finding_only = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.max_samples > 0 and len(out_rows) >= args.max_samples:
                break
            image_index = (row.get("Image Index") or "").strip()
            if not image_index:
                continue
            rel = image_map.get(image_index)
            if rel is None:
                missing_images += 1
                continue

            findings = parse_nih_labels(row.get("Finding Labels", ""))
            if findings == {"No Finding"}:
                skipped_no_finding_only += 1

            labels = {}
            mask = {}
            for lbl in VLM_LABELS:
                if lbl == "No Finding":
                    labels[lbl] = 0
                    mask[lbl] = 0
                elif lbl in NIH_FINDINGS_FOR_VLM:
                    labels[lbl] = int(lbl in findings)
                    mask[lbl] = 1
                else:
                    labels[lbl] = 0
                    mask[lbl] = 0

            patient_id = (row.get("Patient ID") or "unknown").strip()
            out_rows.append(
                {
                    "path": rel,
                    "image_id": image_index,
                    "patient_id": patient_id,
                    "labels": labels,
                    "mask": mask,
                }
            )

    payload = {
        "meta": {
            "source": str(csv_path),
            "nih_root": str(nih_root),
            "label_order": VLM_LABELS,
            "num_rows": len(out_rows),
            "missing_images": missing_images,
            "skipped_no_finding_only": skipped_no_finding_only,
            "max_samples": args.max_samples,
        },
        "rows": out_rows,
    }
    out_path = Path(args.out_json)
    write_json(out_path, payload)
    write_json(
        Path("data/processed/multilabel/nih/canonical_labels_report.json"),
        payload["meta"],
    )
    print(json.dumps(payload["meta"], indent=2))


if __name__ == "__main__":
    main()
