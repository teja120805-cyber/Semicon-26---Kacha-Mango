#!/usr/bin/env python
"""Localization / inference entry point.

The Applied Materials help document's recommended submission layout puts
`generate_dataset.py` and `localize.py` at the repository root, so both exist
here as the obvious things to run. This is a thin delegator: the real CLI is
`scripts/localize_pair.py`, driving the one production implementation in
`pipeline/localize.py`. Every argument is forwarded unchanged.

Single pair:

    python localize.py --reference ref.png --search search.png

Batch, no source-code changes required (CSV with reference_path,search_path
columns, each resolved relative to the CSV's own directory unless absolute):

    python localize.py --batch-csv pairs.csv --out predictions.csv

Output: pred_x, pred_y in Search-image pixels, origin top-left, x increasing
right and y increasing downward - the convention the problem statement
specifies. Also confidence, ambiguous, ambiguity_ratio, runtime_s. Ground
truth is never read, so this works on genuinely unknown pairs.

NOTE for evaluators: there is exactly one localization implementation in this
repository. This wrapper, `scripts/localize_pair.py`, `evaluation/evaluate.py`
and the Streamlit app all call the same `pipeline.localize.localize`.
"""
from __future__ import annotations

import os
import runpy
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(PROJECT_ROOT, "scripts", "localize_pair.py")

if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    sys.argv[0] = TARGET
    runpy.run_path(TARGET, run_name="__main__")
