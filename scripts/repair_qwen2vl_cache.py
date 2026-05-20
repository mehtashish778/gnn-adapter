#!/usr/bin/env python3
"""Repair incomplete Qwen2-VL-2B-Instruct HF cache (re-download missing shards)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from qwen2vl_lora_common import (
    DEFAULT_HF_CACHE,
    DEFAULT_MODEL_ROOT,
    ensure_model_snapshot,
    validate_model_weights,
)


def main():
    parser = argparse.ArgumentParser(description="Repair Qwen2-VL HF cache weights.")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--cache_dir", default=str(DEFAULT_HF_CACHE))
    args = parser.parse_args()

    resolved = ensure_model_snapshot(
        Path(args.model_path),
        cache_dir=Path(args.cache_dir),
        allow_download=True,
    )
    ok = validate_model_weights(resolved)
    print({"snapshot": str(resolved), "weights_ok": ok})
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
