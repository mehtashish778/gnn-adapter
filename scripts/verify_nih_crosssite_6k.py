#!/usr/bin/env python3
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def load_paths(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    return [r["path"] for r in d["rows"]], d.get("meta", {}).get("num_rows", len(d["rows"]))

p2 = REPO / "data/processed/splits/nih/test_rows_n6000.json"
p35 = REPO / "data/processed/splits/nih/test_rows_qwen35_2b_n6000.json"
paths2, n2 = load_paths(p2)
paths35, n35 = load_paths(p35)
print("=== NIH test row files ===")
print(f"test_rows_n6000.json: {n2} rows")
print(f"test_rows_qwen35_2b_n6000.json: {n35} rows")
print(f"Same count: {len(paths2) == len(paths35)}")
print(f"Same paths+order: {paths2 == paths35}")

runs = [
    ("Qwen2 frozen", "vlm_zeroshot/nih/crosssite_eval"),
    ("Qwen3.5 frozen", "vlm_zeroshot/nih/crosssite_eval_qwen35_2b"),
    ("Qwen2 CBM post-hoc", "cbm_posthoc/nih/crosssite_eval"),
    ("Qwen3.5 CBM post-hoc", "cbm_posthoc/nih/crosssite_eval_qwen35_2b"),
    ("Qwen2 CBM LF", "cbm_labelfree/nih/crosssite_eval"),
    ("Qwen3.5 CBM LF", "cbm_labelfree/nih/crosssite_eval_qwen35_2b"),
    ("Qwen2 CCA", "cca/nih/crosssite_eval"),
    ("Qwen3.5 CCA", "cca/nih/crosssite_eval_qwen35_2b"),
    ("Qwen2 LoRA", "qwen2vl_lora_r16/nih/crosssite_eval"),
    ("Qwen3.5 LoRA", "qwen35_2b_lora_r16/nih/crosssite_eval_qwen35_2b"),
]
print("\n=== Cross-site eval outputs ===")
for name, rel in runs:
    rd = REPO / "data/processed/experiments" / rel
    mpath = rd / "metrics.json"
    tpath = rd / "test_predictions.json"
    n = None
    cross = protocol = None
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        cross = m.get("cross_site")
        protocol = m.get("protocol")
        if "test" in m and isinstance(m["test"], dict):
            n = m["test"].get("subset_n_examples")
    if tpath.exists():
        t = json.loads(tpath.read_text(encoding="utf-8"))
        n = len(t.get("probs", t.get("y_true", [])))
    print(f"{name}: exists={mpath.exists()} n={n} cross_site={cross} protocol={protocol}")
