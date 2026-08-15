"""Generates the SECOND independent-seed targeted validation set (Phase 5
robustness check) - same 11 families/categories as dataset A, different
seed, into experiments/finer_grid_validation/data_b/. Isolated from
data/, data/ (dataset A), and every other experiment's data.
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator.dataset_generator import generate_dataset  # noqa: E402
from experiments.finer_grid_validation.dataset_config import TARGETED_FAMILIES, VALIDATION_SEED_B  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data_b")


def main() -> None:
    manifest_path = os.path.join(DATA_DIR, "validation", "ground_truth.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
        if len(existing) == 132:
            print(f"Dataset B already present at {DATA_DIR} (132 pairs) - skipping regeneration.")
            return

    print(f"Generating dataset B (seed={VALIDATION_SEED_B}, 11 families x 12 pairs) -> {DATA_DIR}")
    generate_dataset(DATA_DIR, seed=VALIDATION_SEED_B, families=TARGETED_FAMILIES,
                      only_splits=["validation"], verbose=False)

    with open(manifest_path) as f:
        records = json.load(f)
    print(f"Generated {len(records)} pairs across {len(set(r['structural_family'] for r in records))} families")


if __name__ == "__main__":
    main()
