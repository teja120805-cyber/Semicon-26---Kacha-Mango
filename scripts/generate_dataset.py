#!/usr/bin/env python
"""Generate the full DriftSense V2 dataset (all 4 splits), the bonus
acquisition-variant demo set, and import the external cross_generator
evaluation surface - the three commands the README documents separately,
run together for convenience.

    python scripts/generate_dataset.py [--seed 777001] [--splits development]
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from generator.dataset_generator import generate_dataset, generate_acquisition_variant_demo  # noqa: E402
from scripts.import_cross_generator_eval_data import import_cross_generator_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--seed", type=int, default=777001)
    parser.add_argument("--splits", default=None, help="Comma-separated subset (default: all 4).")
    parser.add_argument("--skip-acquisition-variants", action="store_true")
    parser.add_argument("--skip-cross-generator", action="store_true")
    args = parser.parse_args()

    only_splits = args.splits.split(",") if args.splits else None
    generate_dataset(args.out, seed=args.seed, only_splits=only_splits)

    if not args.skip_acquisition_variants:
        generate_acquisition_variant_demo(args.out, seed=args.seed)

    if not args.skip_cross_generator:
        try:
            import_cross_generator_data(dest_dir=os.path.join(args.out, "cross_generator"))
        except FileNotFoundError as e:
            print(f"Skipping cross_generator import: {e}")


if __name__ == "__main__":
    main()
