"""Generates the targeted validation set (dataset_config.py) into
experiments/finer_grid_validation/data/ - entirely isolated from data/ and
every other experiment's data. Does NOT touch the production dataset.
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator.dataset_generator import generate_dataset  # noqa: E402
from experiments.finer_grid_validation.dataset_config import TARGETED_FAMILIES, VALIDATION_SEED  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")


def main() -> None:
    manifest_path = os.path.join(DATA_DIR, "validation", "ground_truth.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
        if len(existing) == 132:
            print(f"Targeted validation set already present at {DATA_DIR} (132 pairs) - skipping regeneration.")
            return

    print(f"Generating targeted validation set (seed={VALIDATION_SEED}, 11 families x 12 pairs) -> {DATA_DIR}")
    generate_dataset(DATA_DIR, seed=VALIDATION_SEED, families=TARGETED_FAMILIES,
                      only_splits=["validation"], verbose=False)

    with open(manifest_path) as f:
        records = json.load(f)
    families_present = sorted(set(r["structural_family"] for r in records))
    print(f"Generated {len(records)} pairs across {len(families_present)} families: {families_present}")


if __name__ == "__main__":
    main()
