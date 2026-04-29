import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


VLM_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Edema",
    "Consolidation",
    "No Finding",
]

CSV_TO_VLM = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Pleural Effusion": "Effusion",
    "Pneumonia": "Pneumonia",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "No Finding": "No Finding",
}


def normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def clip_image_embeds_tensor(clip_model, pixel_values) -> Any:
    """
    HF CLIPModel.get_image_features may return a Tensor (older) or BaseModelOutputWithPooling (newer).
    Projected image embeddings are the returned tensor or .pooler_output.
    """
    import torch

    out = clip_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(out):
        return out
    po = getattr(out, "pooler_output", None)
    if po is not None:
        return po
    raise RuntimeError("CLIP get_image_features returned unexpected type %s" % type(out))


def resolve_dataset_image_path(image_root: Path, rel_path: str) -> Path:
    """
    Split rows often use CheXpert-v1.0-small/train/... while files on disk are image_root/train/...
    Try the path as-is, then with the CheXpert-v1.0-small/ prefix stripped.
    """
    rel = normalize_path(rel_path)
    candidates = [image_root / rel]
    prefix = "CheXpert-v1.0-small/"
    if rel.startswith(prefix):
        candidates.append(image_root / rel[len(prefix) :])
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing image (tried {candidates}) for rel_path={rel_path!r}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_json(path: Path, payload) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_uncertain(value: str, policy: str) -> Tuple[int, int]:
    if value is None or value == "":
        return 0, 0
    x = float(value)
    if x == 1.0:
        return 1, 1
    if x == 0.0:
        return 0, 1
    if x == -1.0:
        if policy == "u_ones":
            return 1, 1
        if policy == "u_zeros":
            return 0, 1
        if policy == "ignore":
            return 0, 0
    return 0, 0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def train_val_test_split(
    rows: List[dict],
    key: str,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    ids = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    train, val, test = [], [], []
    for k, group_rows in groups.items():
        if k in train_ids:
            train.extend(group_rows)
        elif k in val_ids:
            val.extend(group_rows)
        else:
            test.extend(group_rows)
    return train, val, test


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
