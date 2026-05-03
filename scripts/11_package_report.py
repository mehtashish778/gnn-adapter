#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from common_multilabel import subset_accuracy_masked_lists, write_json
from model_registry import MODEL_SPECS

BASELINE_FROZEN_VLM = Path("data/processed/experiments/baseline_frozen_vlm/metrics.json")


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(model_id: str, protocol: str) -> Optional[Path]:
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


def get_default_run_dir(model_id: str) -> Optional[Path]:
    r = resolve_run_dir(model_id, "default")
    if r is not None:
        return r
    legacy_parent = {
        "vlm_mlp": Path("data/processed/experiments/baseline_mlp"),
        "gnn07_label_residual": Path("data/processed/experiments/gnn_adapter"),
        "gnn12_clip_vlm_homo": Path("data/processed/experiments/clip_vlm_gnn_adapter"),
        "gnn13_clip_bipartite": Path("data/processed/experiments/bipartite_clip_gnn_adapter"),
    }
    p = legacy_parent.get(model_id)
    if p and p.is_dir():
        return p
    return None


def extract_default_metrics(model_id: str) -> Optional[dict]:
    run_dir = resolve_run_dir(model_id, "default")
    if run_dir is not None:
        return read_json(run_dir / "metrics.json") or {}

    legacy = {
        "vlm_zeroshot": str(BASELINE_FROZEN_VLM),
        "vlm_mlp": "data/processed/experiments/baseline_mlp/metrics.json",
        "gnn07_label_residual": "data/processed/experiments/gnn_adapter/metrics.json",
        "gnn12_clip_vlm_homo": "data/processed/experiments/clip_vlm_gnn_adapter/metrics.json",
        "gnn13_clip_bipartite": "data/processed/experiments/bipartite_clip_gnn_adapter/metrics.json",
    }
    p = legacy.get(model_id)
    return read_json(Path(p)) if p else None


def thresholds_for_predictions(run_dir: Path, dim: int) -> Optional[list]:
    js = read_json(run_dir / "per_class_thresholds.json")
    th = js.get("thresholds") if js else None
    if isinstance(th, list) and len(th) == dim:
        return [float(x) for x in th]
    return None


def subset_accuracy_from_predictions_path(pred_path: Path, thresholds: list) -> Optional[float]:
    """Exact multi-label match rate for one split; thresholds length must equal num labels."""
    data = read_json(pred_path)
    if not data or not isinstance(data.get("probs"), list) or not data["probs"]:
        return None
    probs = data["probs"]
    if len(thresholds) != len(probs[0]):
        return None
    acc, _ = subset_accuracy_masked_lists(probs, data["y_true"], data["y_mask"], thresholds)
    return float(acc)


def infer_label_dim(run_dir: Path) -> Optional[int]:
    for name in ("val_predictions.json", "test_predictions.json", "calib_predictions.json"):
        p = run_dir / name
        if not p.exists():
            continue
        data = read_json(p)
        if data and isinstance(data.get("probs"), list) and data["probs"]:
            return len(data["probs"][0])
    return None


def fill_subset_defaults_from_predictions(
    model_id: str,
    run_dir: Optional[Path],
    val_05: Optional[float],
    test_05: Optional[float],
    val_thr: Optional[float],
    test_thr: Optional[float],
):
    """Backfill subset columns from val/test predictions JSON (0.5 and/or per-class thresholds)."""
    if model_id == "vlm_zeroshot" or run_dir is None or not run_dir.is_dir():
        return val_05, test_05, val_thr, test_thr

    dim = infer_label_dim(run_dir)
    if dim is None:
        return val_05, test_05, val_thr, test_thr

    thr = thresholds_for_predictions(run_dir, dim)
    fixed05 = [0.5] * dim
    vpn = run_dir / "val_predictions.json"
    tpn = run_dir / "test_predictions.json"

    if val_05 is None and vpn.exists():
        val_05 = subset_accuracy_from_predictions_path(vpn, fixed05)
    if test_05 is None and tpn.exists():
        test_05 = subset_accuracy_from_predictions_path(tpn, fixed05)
    if thr is not None:
        if val_thr is None and vpn.exists():
            val_thr = subset_accuracy_from_predictions_path(vpn, thr)
        if test_thr is None and tpn.exists():
            test_thr = subset_accuracy_from_predictions_path(tpn, thr)

    return val_05, test_05, val_thr, test_thr


def enrich_calibrated_subsets(run_dir: Path, out: Dict[str, Any]) -> None:
    """Populate val_subset / test_subset using *_predictions.json when still missing."""
    dim = infer_label_dim(run_dir)
    if dim is None:
        return
    thr = thresholds_for_predictions(run_dir, dim) or ([0.5] * dim)
    for key, fname in (("val_subset", "val_predictions.json"), ("test_subset", "test_predictions.json")):
        if out.get(key) is not None:
            continue
        pp = run_dir / fname
        if not pp.exists():
            continue
        acc = subset_accuracy_from_predictions_path(pp, thr)
        if acc is not None:
            out[key] = acc


def augment_zeroshot_metrics(d: Optional[dict]) -> dict:
    """Registry zeroshot metrics may omit subset_accuracy; overlay from latest 05_run_baseline_frozen_vlm output."""
    out = copy.deepcopy(d) if d else {}
    b = read_json(BASELINE_FROZEN_VLM)
    if not b:
        return out
    for split in ("val", "test"):
        if split not in out or not isinstance(out.get(split), dict):
            out[split] = {}
        if out[split].get("subset_accuracy") is None and (b.get(split) or {}).get("subset_accuracy") is not None:
            out[split]["subset_accuracy"] = b[split]["subset_accuracy"]
    return out


def extract_default_subsets(
    model_id: str, d: dict, run_dir: Optional[Path]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns val@0.5, test@0.5, val@per_class_thr, test@per_class_thr subset accuracies."""
    if model_id == "vlm_zeroshot":
        z = augment_zeroshot_metrics(d)
        val_05 = (z.get("val") or {}).get("subset_accuracy")
        test_05 = (z.get("test") or {}).get("subset_accuracy")
        return val_05, test_05, None, None

    val_05 = d.get("val_subset_accuracy@0.5")
    test_05 = d.get("test_subset_accuracy@0.5")
    val_thr = d.get("val_subset_accuracy@per_class_thr")
    test_thr = d.get("test_subset_accuracy@per_class_thr")

    if run_dir is not None and run_dir.is_dir():
        if val_thr is None:
            vm = read_json(run_dir / "val_metrics_calibrated.json")
            if vm is not None:
                val_thr = vm.get("subset_accuracy")
        if test_thr is None:
            tm = read_json(run_dir / "test_metrics_calibrated.json")
            if tm is not None:
                test_thr = tm.get("subset_accuracy")

    return fill_subset_defaults_from_predictions(model_id, run_dir, val_05, test_05, val_thr, test_thr)


def extract_calibrated_metrics(model_id: str) -> Optional[Dict[str, Any]]:
    def pack(run_dir: Path) -> Dict[str, Any]:
        run_name = run_dir.name
        val = read_json(run_dir / "val_metrics_calibrated.json")
        test = read_json(run_dir / "test_metrics_calibrated.json")
        metrics = read_json(run_dir / "metrics.json") or {}
        out: Dict[str, Any] = {"run": run_name}
        if val:
            out["val_macro_f1"] = val.get("macro_f1")
            out["val_subset"] = val.get("subset_accuracy")
        if test:
            out["test_macro_f1"] = test.get("macro_f1")
            out["test_subset"] = test.get("subset_accuracy")
        if out.get("val_macro_f1") is None:
            out["val_macro_f1"] = metrics.get("val_macro_f1@per_class_thr")
        if out.get("test_macro_f1") is None:
            out["test_macro_f1"] = metrics.get("test_macro_f1@per_class_thr")
        if out.get("val_subset") is None:
            out["val_subset"] = metrics.get("val_subset_accuracy@per_class_thr")
        if out.get("test_subset") is None:
            out["test_subset"] = metrics.get("test_subset_accuracy@per_class_thr")
        if isinstance(metrics.get("val"), dict):
            if out.get("val_macro_f1") is None:
                out["val_macro_f1"] = metrics["val"].get("macro_f1")
            if out.get("val_subset") is None:
                out["val_subset"] = metrics["val"].get("subset_accuracy")
        if isinstance(metrics.get("test"), dict):
            if out.get("test_macro_f1") is None:
                out["test_macro_f1"] = metrics["test"].get("macro_f1")
            if out.get("test_subset") is None:
                out["test_subset"] = metrics["test"].get("subset_accuracy")
        enrich_calibrated_subsets(run_dir, out)
        return out

    run_dir = resolve_run_dir(model_id, "calibrated4way")
    if run_dir is not None:
        return pack(run_dir)

    legacy_dirs = {
        "vlm_mlp": Path("data/processed/experiments/mlp_calibrated"),
        "gnn07_label_residual": Path("data/processed/experiments/gnn_calibrated"),
        "gnn12_clip_vlm_homo": Path("data/processed/experiments/clip_vlm_gnn_calibrated4way"),
        "gnn13_clip_bipartite": Path("data/processed/experiments/bipartite_clip_gnn_calibrated4way"),
    }
    p = legacy_dirs.get(model_id)
    if p and p.is_dir():
        return pack(p)
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
        if model_id == "vlm_zeroshot":
            d = augment_zeroshot_metrics(d)
        run_dir = get_default_run_dir(model_id)

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

        ds_val_05, ds_test_05, ds_val_thr, ds_test_thr = extract_default_subsets(model_id, d, run_dir)

        model_rows.append(
            {
                "model_id": model_id,
                "display_name": spec.display_name,
                "description": spec.description,
                "default_val_05": default_val05,
                "default_test_05": default_test05,
                "default_val_thr": default_valthr,
                "default_test_thr": default_testthr,
                "default_val_subset_05": ds_val_05,
                "default_test_subset_05": ds_test_05,
                "default_val_subset_thr": ds_val_thr,
                "default_test_subset_thr": ds_test_thr,
                "calib_val": c.get("val_macro_f1"),
                "calib_test": c.get("test_macro_f1"),
                "calib_val_subset": c.get("val_subset"),
                "calib_test_subset": c.get("test_subset"),
                "calib_run": c.get("run"),
            }
        )

    notes = """
## Notes on NA & how subset is resolved

- **Subset @0.5** for adapters uses `val_predictions.json` / `test_predictions.json` with thresholds `0.5` on each class when absent from `metrics.json`.
- **Subset @per_class_thr** uses the same predictions with `per_class_thresholds.json` in that run folder (otherwise NA). Frozen VLM has no tuned per-class thresholds in this workflow.
- If `*_metrics_calibrated.json` exists, its subset/precision aggregates are preferred before recomputing from JSON predictions.
- **VLMZeroShot** subset @0.5 is synced from `baseline_frozen_vlm/metrics.json` when the registry `metrics.json` omits it. Calibrated-row subset stays **NA** until that run exports predictions JSON (same shapes as adapters).
- Compare models using the same underlying split/check that `subset_n_examples` agrees when benchmarking.
"""

    for out_file in [Path(args.out_md), Path(args.comparison_out_md)]:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            f.write("# Model Comparison Report\n\n")
            f.write("## Model Registry\n\n")
            for r in model_rows:
                f.write(f"- `{r['model_id']}` ({r['display_name']}): {r['description']}\n")

            f.write("\n## Default Split Comparison (macro F1, from metrics.json)\n\n")
            f.write("| Model | Val @0.5 | Test @0.5 | Val @per_class_thr | Test @per_class_thr |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in model_rows:
                f.write(
                    f"| {r['display_name']} | {fmt(r['default_val_05'])} | {fmt(r['default_test_05'])} | {fmt(r['default_val_thr'])} | {fmt(r['default_test_thr'])} |\n"
                )

            f.write("\n## Default Split — subset accuracy (exact multi-label match)\n\n")
            f.write("| Model | Val @0.5 | Test @0.5 | Val @per_class_thr | Test @per_class_thr |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in model_rows:
                f.write(
                    f"| {r['display_name']} | {fmt(r['default_val_subset_05'])} | {fmt(r['default_test_subset_05'])} | {fmt(r['default_val_subset_thr'])} | {fmt(r['default_test_subset_thr'])} |\n"
                )

            f.write("\n## Calibrated4way Results\n\n")
            f.write("| Model | Run | Val macro F1 | Test macro F1 | Val subset acc. | Test subset acc. |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for r in model_rows:
                f.write(
                    f"| {r['display_name']} | `{r['calib_run'] or 'NA'}` | {fmt(r['calib_val'])} | {fmt(r['calib_test'])} | {fmt(r['calib_val_subset'])} | {fmt(r['calib_test_subset'])} |\n"
                )

            f.write("\n## Calibrated vs Default (macro F1 + subset accuracy)\n\n")
            f.write(
                "| Model | Def val F1@0.5 | Def test F1@0.5 | Def val subset@0.5 | Def test subset@0.5 | Cal val F1 | Cal test F1 | Cal val subset | Cal test subset |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in model_rows:
                f.write(
                    f"| {r['display_name']} | {fmt(r['default_val_05'])} | {fmt(r['default_test_05'])} | {fmt(r['default_val_subset_05'])} | {fmt(r['default_test_subset_05'])} | {fmt(r['calib_val'])} | {fmt(r['calib_test'])} | {fmt(r['calib_val_subset'])} | {fmt(r['calib_test_subset'])} |\n"
                )

            f.write(notes)

    write_json(Path("reports/comparison/overall.json"), {"models": model_rows})
    print({"report": args.out_md, "comparison_report": args.comparison_out_md})


if __name__ == "__main__":
    main()
