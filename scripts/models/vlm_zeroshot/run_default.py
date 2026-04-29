#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[3]
    target = repo / "scripts" / "05_run_baseline_frozen_vlm.py"
    cmd = [sys.executable, str(target), "--model_id", "vlm_zeroshot", "--protocol", "default", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=repo))


if __name__ == "__main__":
    main()

