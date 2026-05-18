import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


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


def subset_accuracy_masked_lists(
    probs: Sequence[Sequence[float]],
    y_true: Sequence[Sequence[float]],
    y_mask: Sequence[Sequence[float]],
    thresholds: Sequence[float],
) -> Tuple[float, int]:
    """
    Exact multi-label match rate: fraction of samples where binary predictions match y_true on
    every label with y_mask > 0. Rows with no supervised labels are skipped.

    Returns (accuracy, n_samples_used).
    """
    c = len(thresholds)
    n_exact = n_used = 0
    for y, m, p in zip(y_true, y_mask, probs):
        if sum(int(mi != 0) for mi in m) == 0:
            continue
        n_used += 1
        ok = True
        for i in range(c):
            if m[i] == 0:
                continue
            pred = 1 if float(p[i]) >= float(thresholds[i]) else 0
            if pred != int(y[i]):
                ok = False
                break
        if ok:
            n_exact += 1
    return (n_exact / n_used, n_used) if n_used else (0.0, 0)


def masked_subset_accuracy(
    probs: Any,
    y_true: Any,
    y_mask: Any,
    threshold: Union[float, Sequence[float]] = 0.5,
) -> float:
    """
    Torch version of subset_accuracy_masked_lists. probs/y_true/y_mask shape (N, C).
    """
    import torch

    if isinstance(threshold, (list, tuple)):
        thr = torch.tensor(threshold, dtype=probs.dtype, device=probs.device)
        pred = (probs >= thr.unsqueeze(0)).float()
    else:
        pred = (probs >= float(threshold)).float()
    supervised = y_mask > 0
    used_rows = supervised.any(dim=1)
    if used_rows.sum() == 0:
        return 0.0
    mismatch = ((pred != y_true) & supervised).any(dim=1)
    ok = (~mismatch) & used_rows
    return (ok.sum().float() / used_rows.sum().float()).item()


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


def load_rows(path: Path) -> List[dict]:
    """Load aligned multilabel rows from a splits JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["rows"]


def row_ids(rows: Sequence[dict]) -> List[str]:
    """Stable row identifiers for feature-cache integrity checks."""
    return [normalize_path(str(r.get("path", i))) for i, r in enumerate(rows)]


def to_vlm_feature_tensors(rows: Sequence[dict]):
    """Stack VLM logits/probs and labels into tensors; MLP input is flattened [z;p]."""
    import torch

    x_probs = torch.tensor([r["x_probs"] for r in rows], dtype=torch.float32)
    x_logits = torch.tensor([r["x_logits"] for r in rows], dtype=torch.float32)
    y_true = torch.tensor([r["y_true"] for r in rows], dtype=torch.float32)
    y_mask = torch.tensor([r["y_mask"] for r in rows], dtype=torch.float32)
    x = torch.stack([x_logits, x_probs], dim=-1).reshape(len(rows), -1)
    return x, x_logits, x_probs, y_true, y_mask


def to_vlm_training_batch(rows: Sequence[dict]):
    """Return ``(x, y_true, y_mask)`` for MLP / training_engine (not logits/probs)."""
    x, _x_logits, _x_probs, y_true, y_mask = to_vlm_feature_tensors(rows)
    return x, y_true, y_mask


def to_label_tensors(rows: Sequence[dict]):
    """VLM logits/probs and labels only (no flattened MLP input)."""
    import torch

    x_logits = torch.tensor([r["x_logits"] for r in rows], dtype=torch.float32)
    x_probs = torch.tensor([r["x_probs"] for r in rows], dtype=torch.float32)
    y_true = torch.tensor([r["y_true"] for r in rows], dtype=torch.float32)
    y_mask = torch.tensor([r["y_mask"] for r in rows], dtype=torch.float32)
    return x_logits, x_probs, y_true, y_mask


def build_adj(num_nodes: int, edge_index: Sequence[Sequence[int]], edge_weight: Sequence[float]):
    """Row-normalized adjacency with self-loops."""
    import torch

    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for s, t, w in zip(edge_index[0], edge_index[1], edge_weight):
        a[int(s), int(t)] = float(w)
    a = a + torch.eye(num_nodes, dtype=torch.float32)
    deg = a.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return a / deg


def masked_macro_f1(probs, y_true, y_mask, threshold: Union[float, Sequence[float]] = 0.5) -> float:
    """Per-class masked macro-F1; threshold may be scalar or per-class list."""
    import torch

    c = probs.shape[1]
    if isinstance(threshold, (list, tuple)):
        thr = torch.tensor(threshold, dtype=probs.dtype, device=probs.device)
        pred = (probs >= thr.unsqueeze(0)).float()
    else:
        pred = (probs >= float(threshold)).float()
    f1s = []
    for i in range(c):
        mask = y_mask[:, i] > 0
        if mask.sum() == 0:
            f1s.append(torch.tensor(0.0, device=probs.device))
            continue
        p = pred[mask, i]
        y = y_true[mask, i]
        tp = ((p == 1) & (y == 1)).sum().float()
        fp = ((p == 1) & (y == 0)).sum().float()
        fn = ((p == 0) & (y == 1)).sum().float()
        denom = (2 * tp + fp + fn).clamp(min=1e-8)
        f1s.append((2 * tp) / denom)
    return torch.stack(f1s).mean().item()


def masked_bce_with_logits(out, y_true, y_mask, pos_weight) -> Any:
    """Masked binary cross-entropy with logits."""
    import torch.nn.functional as F

    raw = F.binary_cross_entropy_with_logits(out, y_true, pos_weight=pos_weight, reduction="none")
    return (raw * y_mask).sum() / y_mask.sum().clamp(min=1.0)


def probabilistic_metrics(probs, y_true, y_mask, n_ece_bins: int = 15) -> dict:
    """Per-class AUROC, AUPRC, ECE, Brier; mean (macro) over masked classes.

    Inputs are torch tensors of shape (N, C) or numpy arrays. Masked entries
    (y_mask == 0) are excluded per-class. Classes with zero positives or
    zero negatives after masking get NaN AUROC/AUPRC and are excluded from
    the macro mean (Brier/ECE still computed where possible).
    """
    import numpy as np

    try:
        import torch

        if hasattr(probs, "detach"):
            probs = probs.detach().cpu().numpy()
        if hasattr(y_true, "detach"):
            y_true = y_true.detach().cpu().numpy()
        if hasattr(y_mask, "detach"):
            y_mask = y_mask.detach().cpu().numpy()
    except ImportError:
        pass

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        roc_auc_score = average_precision_score = None

    c = probs.shape[1]
    aurocs, auprcs, briers, eces = [], [], [], []
    per_class: list[dict] = []
    for i in range(c):
        m = y_mask[:, i] > 0.5
        if m.sum() < 2:
            per_class.append({"auroc": float("nan"), "auprc": float("nan"), "ece": float("nan"), "brier": float("nan")})
            continue
        p = probs[m, i].astype(np.float64)
        y = y_true[m, i].astype(np.float64)
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        if roc_auc_score is not None and n_pos > 0 and n_neg > 0:
            auroc = float(roc_auc_score(y, p))
            auprc = float(average_precision_score(y, p))
        else:
            auroc = float("nan")
            auprc = float("nan")
        brier = float(((p - y) ** 2).mean())
        bins = np.linspace(0.0, 1.0, n_ece_bins + 1)
        bin_idx = np.digitize(p, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_ece_bins - 1)
        ece = 0.0
        for b in range(n_ece_bins):
            sel = bin_idx == b
            if not sel.any():
                continue
            avg_conf = float(p[sel].mean())
            avg_acc = float(y[sel].mean())
            ece += (sel.sum() / len(p)) * abs(avg_conf - avg_acc)
        per_class.append({"auroc": auroc, "auprc": auprc, "ece": float(ece), "brier": brier})
        if not np.isnan(auroc):
            aurocs.append(auroc)
        if not np.isnan(auprc):
            auprcs.append(auprc)
        briers.append(brier)
        eces.append(float(ece))

    def _macro(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "macro_auroc": _macro(aurocs),
        "macro_auprc": _macro(auprcs),
        "macro_ece": _macro(eces),
        "macro_brier": _macro(briers),
        "per_class": per_class,
    }


def compute_pos_weight(y_true, y_mask, max_weight: float = 100.0):
    """Per-class positive weights for BCE (neg/pos ratio, clamped)."""
    import torch

    pos = (y_true * y_mask).sum(dim=0)
    neg = ((1 - y_true) * y_mask).sum(dim=0).clamp(min=1)
    return (neg / pos.clamp(min=1)).clamp(max=max_weight)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_per_class_thresholds(path: Path) -> Optional[List[float]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    th = payload.get("thresholds")
    if not th:
        return None
    return [float(x) for x in th]


def build_standard_argparser(description: str) -> argparse.ArgumentParser:
    """Shared CLI flags for training scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--calib_rows_json", default=None, help="Optional calibration rows JSON.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--resume_from", default="", help="Optional checkpoint path.")
    parser.add_argument("--gpu_id", type=int, default=0, help="Single GPU index.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def require_cuda_device(gpu_id: int):
    """Return torch device after validating CUDA availability."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only mode: CUDA is not available.")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid --gpu_id {gpu_id}; available GPUs: 0..{torch.cuda.device_count() - 1}"
        )
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")
