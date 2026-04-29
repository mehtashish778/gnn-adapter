#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common_multilabel import write_json
from model_registry import resolve_experiment_dir, update_run_registry


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["rows"]


def to_tensors(rows):
    import torch

    x_probs = torch.tensor([r["x_probs"] for r in rows], dtype=torch.float32)
    x_logits = torch.tensor([r["x_logits"] for r in rows], dtype=torch.float32)
    y_true = torch.tensor([r["y_true"] for r in rows], dtype=torch.float32)
    y_mask = torch.tensor([r["y_mask"] for r in rows], dtype=torch.float32)
    x = torch.stack([x_logits, x_probs], dim=-1).reshape(len(rows), -1)
    return x, y_true, y_mask


def macro_f1(prob, y, m, thr=0.5):
    import torch

    c = y.shape[1]
    pred = (prob >= thr).float()
    vals = []
    for i in range(c):
        mask = m[:, i] > 0
        if mask.sum() == 0:
            vals.append(torch.tensor(0.0))
            continue
        p = pred[mask, i]
        t = y[mask, i]
        tp = ((p == 1) & (t == 1)).sum().float()
        fp = ((p == 1) & (t == 0)).sum().float()
        fn = ((p == 0) & (t == 1)).sum().float()
        vals.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
    return torch.stack(vals).mean().item()


def main():
    parser = argparse.ArgumentParser(description="Run simple MLP residual baseline.")
    parser.add_argument("--train_rows_json", default="data/processed/splits/train_rows.json")
    parser.add_argument("--val_rows_json", default="data/processed/splits/val_rows.json")
    parser.add_argument("--test_rows_json", default="data/processed/splits/test_rows.json")
    parser.add_argument("--calib_rows_json", default=None, help="Optional calibration rows JSON (threshold tuning).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--model_id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--resume_from", default="", help="Optional checkpoint path to initialize model weights.")
    parser.add_argument("--gpu_id", type=int, default=0, help="Single GPU index to use.")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("This script requires PyTorch.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only mode: CUDA is not available.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid --gpu_id {args.gpu_id}; available GPUs: 0..{torch.cuda.device_count()-1}")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    tr = load_rows(Path(args.train_rows_json))
    va = load_rows(Path(args.val_rows_json))
    te = load_rows(Path(args.test_rows_json))
    ca = load_rows(Path(args.calib_rows_json)) if args.calib_rows_json else None

    xtr, ytr, mtr = to_tensors(tr)
    xva, yva, mva = to_tensors(va)
    xte, yte, mte = to_tensors(te)
    if ca is not None:
        xca, yca, mca = to_tensors(ca)
    c = ytr.shape[1]
    d = xtr.shape[1]

    model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, c))
    model = model.to(device)
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    mtr = mtr.to(device)
    xva = xva.to(device)
    yva = yva.to(device)
    mva = mva.to(device)
    xte = xte.to(device)
    yte = yte.to(device)
    mte = mte.to(device)
    if ca is not None:
        xca = xca.to(device)
        yca = yca.to(device)
        mca = mca.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    pos = (ytr * mtr).sum(dim=0)
    neg = ((1 - ytr) * mtr).sum(dim=0).clamp(min=1)
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=100.0)
    best = {"val_macro_f1": -1, "state": None}

    for _ in range(args.epochs):
        model.train()
        out = model(xtr)
        raw_loss = F.binary_cross_entropy_with_logits(out, ytr, pos_weight=pos_weight, reduction="none")
        loss = (raw_loss * mtr).sum() / mtr.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_f1 = macro_f1(torch.sigmoid(model(xva)), yva, mva)
        if val_f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": val_f1, "state": {k: v.cpu() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(xva))
        test_prob = torch.sigmoid(model(xte))
    out_dir = resolve_experiment_dir(
        out_dir=args.out_dir or None,
        model_id=args.model_id or None,
        protocol=args.protocol or None,
        run_id=args.run_id or None,
        default_legacy_out_dir="data/processed/experiments/baseline_mlp",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "val_predictions.json",
        {"probs": val_prob.tolist(), "y_true": yva.tolist(), "y_mask": mva.tolist()},
    )
    write_json(out_dir / "metrics.json", {"best_val_macro_f1": best["val_macro_f1"], "test_macro_f1@0.5": macro_f1(test_prob, yte, mte)})
    torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")
    write_json(
        out_dir / "test_predictions.json",
        {"probs": test_prob.tolist(), "y_true": yte.tolist(), "y_mask": mte.tolist()},
    )
    if ca is not None:
        with torch.no_grad():
            calib_prob = torch.sigmoid(model(xca))
        write_json(
            out_dir / "calib_predictions.json",
            {"probs": calib_prob.tolist(), "y_true": yca.tolist(), "y_mask": mca.tolist()},
        )
    if args.model_id and args.protocol:
        update_run_registry(
            model_id=args.model_id,
            protocol=args.protocol,
            run_dir=out_dir,
            metrics={
                "val_macro_f1@0.5": best["val_macro_f1"],
                "test_macro_f1@0.5": macro_f1(test_prob, yte, mte),
            },
            hparams={"epochs": args.epochs, "lr": args.lr},
        )
    print({"best_val_macro_f1": best["val_macro_f1"]})


if __name__ == "__main__":
    main()
