#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[3]
    target = repo / "scripts" / "07_train_gnn_adapter.py"
    cmd = [sys.executable, str(target), "--model_id", "gnn07_label_residual", "--protocol", "default", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=repo))


if __name__ == "__main__":
    main()

