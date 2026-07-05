#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch

from common_multilabel import load_rows, masked_macro_f1, to_label_tensors


def main() -> None:
    path = Path(sys.argv[1])
    _, probs, y, m = to_label_tensors(load_rows(path))
    f1 = masked_macro_f1(probs, y, m, threshold=0.5)
    print({"path": str(path), "n": len(probs), "test_macro_f1@0.5": float(f1)})


if __name__ == "__main__":
    main()
