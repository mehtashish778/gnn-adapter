#!/usr/bin/env python3
import json
from pathlib import Path


def paths(p: Path) -> set[str]:
    return {r["path"] for r in json.load(p.open())["rows"]}


def split_of(p: str, train: set, val: set, test: set) -> str:
    if p in train:
        return "train"
    if p in val:
        return "val"
    if p in test:
        return "test"
    return "?"


def report(name: str, a: set, b: set) -> None:
    inter = a & b
    print(f"=== {name} ===")
    print(f"  Qwen2: {len(a)}  Qwen3.5: {len(b)}")
    print(
        f"  overlap: {len(inter)} "
        f"({100 * len(inter) / len(a):.2f}% of Qwen2, {100 * len(inter) / len(b):.2f}% of Qwen3.5)"
    )
    print(f"  only Qwen2: {len(a - b)}  only Qwen3.5: {len(b - a)}")


def main() -> None:
    q2t = paths(Path("data/processed/splits/train_rows.json"))
    q2v = paths(Path("data/processed/splits/val_rows.json"))
    q2e = paths(Path("data/processed/splits/test_rows.json"))
    q2all = q2t | q2v | q2e

    q3t = paths(Path("data/processed/splits/qwen35_2b_qwen2subset/train_rows.json"))
    q3v = paths(Path("data/processed/splits/qwen35_2b_qwen2subset/val_rows.json"))
    q3e = paths(Path("data/processed/splits/qwen35_2b_qwen2subset/test_rows.json"))
    fair_dir = Path("data/processed/splits/qwen35_qwen2_splits")
    if (fair_dir / "train_rows.json").is_file():
        q3f_t = paths(fair_dir / "train_rows.json")
        q3f_v = paths(fair_dir / "val_rows.json")
        q3f_e = paths(fair_dir / "test_rows.json")
        report("Train (fair qwen2 splits)", q2t, q3f_t)
        report("Val (fair qwen2 splits)", q2v, q3f_v)
        report("Test (fair qwen2 splits)", q2e, q3f_e)
    q3all = q3t | q3v | q3e

    report("Train", q2t, q3t)
    report("Val", q2v, q3v)
    report("Test", q2e, q3e)
    report("All splits (union)", q2all, q3all)

    union = q2all | q3all
    inter = q2all & q3all
    print("=== Summary ===")
    print(f"  Total unique paths (union): {len(union)}")
    print(f"  Shared paths: {len(inter)} ({100 * len(inter) / len(union):.3f}% of union)")
    print(f"  Only in Qwen2 splits: {len(q2all - q3all)}")
    print(f"  Only in Qwen3.5 splits: {len(q3all - q2all)}")

    mismatch = 0
    for p in inter:
        s2 = split_of(p, q2t, q2v, q2e)
        s3 = split_of(p, q3t, q3v, q3e)
        if s2 != s3:
            mismatch += 1
    print(f"  Shared paths assigned to DIFFERENT split: {mismatch} ({100 * mismatch / len(inter):.2f}% of shared)")

    # patient_id consistency
    def path_to_patient(rows_dir: Path, split_files: list[str]) -> dict[str, str]:
        out = {}
        for sf in split_files:
            for r in json.load((rows_dir / sf).open())["rows"]:
                out[r["path"]] = r.get("patient_id", "?")
        return out

    q2_pat = path_to_patient(Path("data/processed/splits"), ["train_rows.json", "val_rows.json", "test_rows.json"])
    q3_pat = path_to_patient(
        Path("data/processed/splits/qwen35_2b_qwen2subset"),
        ["train_rows.json", "val_rows.json", "test_rows.json"],
    )
    pat_diff = sum(1 for p in inter if q2_pat.get(p) != q3_pat.get(p))
    print(f"  Shared paths with different patient_id: {pat_diff}")

    from collections import defaultdict

    def patient_splits(path_map: dict[str, str], split_sets: tuple[set, set, set]) -> dict[str, set[str]]:
        train, val, test = split_sets
        ps: dict[str, set[str]] = defaultdict(set)
        for path, pid in path_map.items():
            if path in train:
                ps[pid].add("train")
            elif path in val:
                ps[pid].add("val")
            elif path in test:
                ps[pid].add("test")
        return ps

    p2s = patient_splits(q2_pat, (q2t, q2v, q2e))
    p3s = patient_splits(q3_pat, (q3t, q3v, q3e))
    common_pat = set(p2s) & set(p3s)
    pat_split_mismatch = sum(1 for pid in common_pat if p2s[pid] != p3s[pid])
    print(f"  Patients with different split bucket: {pat_split_mismatch} / {len(common_pat)}")


def aligned_pool_report() -> None:
    def paths_from_aligned(p: Path) -> set[str]:
        return {r["path"] for r in json.load(p.open())["rows"]}

    q2 = paths_from_aligned(Path("data/processed/multilabel/aligned_vlm_targets.json"))
    q3 = paths_from_aligned(Path("data/processed/multilabel/aligned_vlm_targets_qwen35_2b_qwen2subset.json"))
    only2 = sorted(q2 - q3)
    only3 = sorted(q3 - q2)
    print("\n=== Aligned VLM JSON (image pool) ===")
    print(f"  Qwen2 aligned rows: {len(q2)}")
    print(f"  Qwen3.5 aligned rows: {len(q3)}")
    print(f"  Overlap: {len(q2 & q3)}")
    print(f"  Only in Qwen2 aligned: {len(only2)}")
    print(f"  Only in Qwen3.5 aligned: {len(only3)}")
    if only2:
        print("  Paths missing from Qwen3.5 (vLLM/align failures):")
        for p in only2:
            print(f"    {p}")


if __name__ == "__main__":
    main()
    aligned_pool_report()

    for label, p in [
        ("Qwen2 CBM posthoc", "data/processed/experiments/cbm_posthoc/default/cbm_posthoc_default/test_predictions.json"),
        ("Qwen3.5 CBM posthoc", "data/processed/experiments/cbm_posthoc/qwen35_2b_qwen2subset/cbm_posthoc_qwen35_2b_qwen2subset/test_predictions.json"),
        ("Qwen2 CBM labelfree", "data/processed/experiments/cbm_labelfree/default/cbm_labelfree_default/test_predictions.json"),
        ("Qwen3.5 CBM labelfree", "data/processed/experiments/cbm_labelfree/qwen35_2b_qwen2subset/cbm_labelfree_qwen35_2b_qwen2subset/test_predictions.json"),
    ]:
        payload = json.load(Path(p).open())
        print(f"  {label}: {len(payload['y_true'])} test rows")
