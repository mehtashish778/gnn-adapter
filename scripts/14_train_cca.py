#!/usr/bin/env python3
"""Train Compositional Concept Adapter (CCA). Implementation in cca_train_core."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cca_train_core import main

if __name__ == "__main__":
    main()
