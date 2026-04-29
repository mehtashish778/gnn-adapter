#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import write_json
from model_registry import MODEL_SPECS


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(model_id: str, protocol: str):
    base = Path("data/processed/experiments") / model_id / protocol
    best_ptr = read_json(base / "best.json")
    if best_ptr and best_ptr.get("run_dir"):
        p = Path(best_ptr["run_dir"])
        if p.exists():
            return p
    latest_ptr = read_json(base / "latest.json")
    if latest_ptr and latest_ptr.get("run_dir"):
        p = Path(latest_ptr["run_dir"])
        if p.exists():
            return p
    return None


def extract_default_metrics(model_id: str):
    run_dir = resolve_run_dir(model_id, "default")
    if run_dir is not None:
        metrics = read_json(run_dir / "metrics.json") or {}
        return metrics

    # Legacy fallbacks
    legacy = {
        "vlm_zeroshot": "data/processed/experiments/baseline_frozen_vlm/metrics.json",
        "vlm_mlp": "data/processed/experiments/baseline_mlp/metrics.json",
        "gnn07_label_residual": "data/processed/experiments/gnn_adapter/metrics.json",
        "gnn12_clip_vlm_homo": "data/processed/experiments/clip_vlm_gnn_adapter/metrics.json",
        "gnn13_clip_bipartite": "data/processed/experiments/bipartite_clip_gnn_adapter/metrics.json",
    }
    p = legacy.get(model_id)
    return read_json(Path(p)) if p else None


def extract_calibrated_metrics(model_id: str):
    run_dir = resolve_run_dir(model_id, "calibrated4way")
    if run_dir is not None:
        val = read_json(run_dir / "val_metrics_calibrated.json")
        test = read_json(run_dir / "test_metrics_calibrated.json")
        if val and test:
            return {"val_macro_f1": val.get("macro_f1"), "test_macro_f1": test.get("macro_f1"), "run": run_dir.name}
        metrics = read_json(run_dir / "metrics.json") or {}
        return {
            "val_macro_f1": metrics.get("val_macro_f1@per_class_thr"),
            "test_macro_f1": metrics.get("test_macro_f1@per_class_thr"),
            "run": run_dir.name,
        }

    # Legacy fallback directories used before organized layout.
    legacy_dirs = {
        "vlm_mlp": "data/processed/experiments/mlp_calibrated",
        "gnn07_label_residual": "data/processed/experiments/gnn_calibrated",
        "gnn12_clip_vlm_homo": "data/processed/experiments/clip_vlm_gnn_calibrated4way",
        "gnn13_clip_bipartite": "data/processed/experiments/bipartite_clip_gnn_calibrated4way",
    }
    p = legacy_dirs.get(model_id)
    if not p:
        return None
    run_dir = Path(p)
    val = read_json(run_dir / "val_metrics_calibrated.json")
    test = read_json(run_dir / "test_metrics_calibrated.json")
    if val and test:
        return {"val_macro_f1": val.get("macro_f1"), "test_macro_f1": test.get("macro_f1"), "run": run_dir.name}
    return None


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Package experiment summary into markdown report.")
    parser.add_argument("--out_md", default="reports/gnn_adapter/report.md")
    parser.add_argument("--comparison_out_md", default="reports/comparison/overall.md")
    args = parser.parse_args()

    model_rows = []
    for model_id, spec in MODEL_SPECS.items():
        d = extract_default_metrics(model_id) or {}
        c = extract_calibrated_metrics(model_id) or {}
        if model_id == "vlm_zeroshot":
            default_val05 = (d.get("val") or {}).get("macro_f1")
            default_test05 = (d.get("test") or {}).get("macro_f1")
            default_valthr = None
            default_testthr = None
        else:
            default_val05 = d.get("best_val_macro_f1", d.get("val_macro_f1@0.5"))
            default_test05 = d.get("test_macro_f1@0.5")
            default_valthr = d.get("val_macro_f1@per_class_thr")
            default_testthr = d.get("test_macro_f1@per_class_thr")

        model_rows.append(
            {
                "model_id": model_id,
                "display_name": spec.display_name,
                "description": spec.description,
                "default_val_05": default_val05,
                "default_test_05": default_test05,
                "default_val_thr": default_valthr,
                "default_test_thr": default_testthr,
                "calib_val": c.get("val_macro_f1"),
                "calib_test": c.get("test_macro_f1"),
                "calib_run": c.get("run"),
            }
        )

    for out_file in [Path(args.out_md), Path(args.comparison_out_md)]:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            f.write("# Model Comparison Report\n\n")
            f.write("## Model Registry\n\n")
            for r in model_rows:
                f.write(f"- `{r['model_id']}` ({r['display_name']}): {r['description']}\n")
            f.write("\n## Default Split Comparison (from metrics.json)\n\n")
            f.write("| Model | Val @0.5 | Test @0.5 | Val @per_class_thr | Test @per_class_thr |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in model_rows:
                f.write(
                    f"| {r['display_name']} | {fmt(r['default_val_05'])} | {fmt(r['default_test_05'])} | {fmt(r['default_val_thr'])} | {fmt(r['default_test_thr'])} |\n"
                )

            f.write("\n## Calibrated4way Results\n\n")
            f.write("| Model | Run | Val macro F1 | Test macro F1 |\n")
            f.write("|---|---|---:|---:|\n")
            for r in model_rows:
                f.write(f"| {r['display_name']} | `{r['calib_run'] or 'NA'}` | {fmt(r['calib_val'])} | {fmt(r['calib_test'])} |\n")

            f.write("\n## Calibrated vs Default\n\n")
            f.write(
                "| Model | Default Val @0.5 | Default Test @0.5 | Default Val @per_class_thr | Default Test @per_class_thr | Calib4way Val | Calib4way Test |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in model_rows:
                f.write(
                    f"| {r['display_name']} | {fmt(r['default_val_05'])} | {fmt(r['default_test_05'])} | {fmt(r['default_val_thr'])} | {fmt(r['default_test_thr'])} | {fmt(r['calib_val'])} | {fmt(r['calib_test'])} |\n"
                )

    # save machine-readable summary
    write_json(Path("reports/comparison/overall.json"), {"models": model_rows})
    print({"report": args.out_md, "comparison_report": args.comparison_out_md})


if __name__ == "__main__":
    main()
