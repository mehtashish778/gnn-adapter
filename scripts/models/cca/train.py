#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[3]
    target = repo / "scripts" / "14_train_cca.py"
    cmd = [sys.executable, str(target), "--model_id", "cca", "--protocol", "default", *sys.argv[1:]]
    env = os.environ.copy()
    scripts = str(repo / "scripts")
    env["PYTHONPATH"] = scripts + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    raise SystemExit(subprocess.call(cmd, cwd=repo, env=env))


if __name__ == "__main__":
    main()
