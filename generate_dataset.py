#!/usr/bin/env python
"""Dataset generation entry point.

The Applied Materials help document's recommended submission layout puts
`generate_dataset.py` and `localize.py` at the repository root, so both exist
here as the obvious things to run. This is a thin delegator: the real
implementation is `scripts/generate_dataset.py`, driving
`generator/dataset_generator.py`. Every argument is forwarded unchanged, so
the two invocations are interchangeable.

    python generate_dataset.py --help
    python generate_dataset.py                       # all 4 splits, seed 777001
    python generate_dataset.py --splits development  # one split

Regenerates the benchmark deterministically from the seed - see README
"Reproducibility" for the caveat about OpenCV version pinning.
"""
from __future__ import annotations

import os
import runpy
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(PROJECT_ROOT, "scripts", "generate_dataset.py")

if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    sys.argv[0] = TARGET
    runpy.run_path(TARGET, run_name="__main__")
