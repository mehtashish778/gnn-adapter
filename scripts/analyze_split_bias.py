#!/usr/bin/env python3
"""Check whether Qwen2 vs Qwen3.5 split/pool differences could bias metrics."""
import json
from pathlib import Path

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Edema",
    "Consolidation",
    "No Finding",
]


def load_rows(p: Path) -> list[dict]:
    return json.load(p.open())["rows"]


def path_map(rows: list[dict]) -> dict[str, dict]:
    return {r["path"]: r for r in rows}


def pos_rate(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {lbl: 0.0 for lbl in LABELS}
    out = {}
    for i, lbl in enumerate(LABELS):
        pos = sum(r["y_true"][i] * r["y_mask"][i] for r in rows)
        denom = sum(r["y_mask"][i] for r in rows)
        out[lbl] = pos / denom if denom else 0.0
    out["macro_pos"] = sum(out[lbl] for lbl in LABELS) / len(LABELS)
    return out


def main() -> None:
    q2_aligned = path_map(load_rows(Path("data/processed/multilabel/aligned_vlm_targets.json")))
    q3_aligned = path_map(
        load_rows(Path("data/processed/multilabel/aligned_vlm_targets_qwen35_2b_qwen2subset.json"))
    )
    only2 = set(q2_aligned) - set(q3_aligned)

    q2_test = load_rows(Path("data/processed/splits/test_rows.json"))
    q3_test = load_rows(Path("data/processed/splits/qwen35_2b_qwen2subset/test_rows.json"))
    q2_train = load_rows(Path("data/processed/splits/train_rows.json"))
    q3_train = load_rows(Path("data/processed/splits/qwen35_2b_qwen2subset/train_rows.json"))

    print("=== 11 missing images (Qwen2 only) ===")
    missing = [q2_aligned[p] for p in sorted(only2)]
    mr = pos_rate(missing)
    pool_r = pos_rate(list(q2_aligned.values()))
    print(f"  count: {len(missing)}")
    for lbl in LABELS:
        print(f"  {lbl:16s} missing={mr[lbl]:.3f}  full_pool={pool_r[lbl]:.3f}  delta={mr[lbl]-pool_r[lbl]:+.3f}")

    print("\n=== Test set label prevalence (ground truth) ===")
    r2 = pos_rate(q2_test)
    r3 = pos_rate(q3_test)
    print(f"  Qwen2 test n={len(q2_test)}  Qwen3.5 test n={len(q3_test)}")
    for lbl in LABELS:
        print(f"  {lbl:16s} q2={r2[lbl]:.3f}  q35={r3[lbl]:.3f}  delta={r3[lbl]-r2[lbl]:+.3f}")
    print(f"  macro_pos       q2={r2['macro_pos']:.3f}  q35={r3['macro_pos']:.3f}  delta={r3['macro_pos']-r2['macro_pos']:+.3f}")

    print("\n=== Train set label prevalence ===")
    tr2 = pos_rate(q2_train)
    tr3 = pos_rate(q3_train)
    for lbl in LABELS:
        print(f"  {lbl:16s} q2={tr2[lbl]:.3f}  q35={tr3[lbl]:.3f}  delta={tr3[lbl]-tr2[lbl]:+.3f}")

    # images that moved train<->test between pipelines
    q2t = {r["path"] for r in q2_train}
    q3t = {r["path"] for r in q3_train}
    q2e = {r["path"] for r in q2_test}
    q3e = {r["path"] for r in q3_test}
    train_to_test = q2t & q3e
    test_to_train = q2e & q3t
    print("\n=== Split leakage / reassignment ===")
    print(f"  Qwen2 train → Qwen3.5 test: {len(train_to_test)} images")
    print(f"  Qwen2 test → Qwen3.5 train: {len(test_to_train)} images")
    if train_to_test:
        leaked = [q2_aligned[p] for p in train_to_test if p in q2_aligned]
        lr = pos_rate(leaked)
        print(f"  (train→test) macro_pos={lr['macro_pos']:.3f} vs q2_train={tr2['macro_pos']:.3f}")


if __name__ == "__main__":
    main()
