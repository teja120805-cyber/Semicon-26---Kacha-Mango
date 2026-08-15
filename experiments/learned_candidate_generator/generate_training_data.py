"""Generates the expanded dev-only training data (dataset_config.py) into
experiments/learned_candidate_generator/dev_data/ - entirely isolated from
data/development/ (production) and every frozen-benchmark split. Also
copies the 3 existing production development families into the same
manifest (reading their already-generated files under data/development/,
not regenerating them) so training can draw on all ~184 pairs uniformly.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator.dataset_generator import generate_dataset  # noqa: E402
from experiments.learned_candidate_generator.dataset_config import DEV_EXPANSION_FAMILIES, DEV_EXPANSION_SEED  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_DATA_DIR = os.path.join(EXP_DIR, "dev_data")


def _copy_production_dev_families(dest_dir: str) -> list[dict]:
    """Read (not regenerate) the 3 existing production development families
    from data/development/, so the combined manifest covers all ~184 pairs
    without duplicating already-correct, already-validated files."""
    src_manifest_path = os.path.join(PROJECT_ROOT, "data", "development", "ground_truth.json")
    with open(src_manifest_path) as f:
        records = json.load(f)
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for r in records:
        for key in ("reference_path", "search_path"):
            src = os.path.join(PROJECT_ROOT, "data", r[key])
            dst = os.path.join(dest_dir, os.path.basename(r[key]))
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
        r2 = dict(r)
        r2["reference_path"] = f"development/{os.path.basename(r['reference_path'])}"
        r2["search_path"] = f"development/{os.path.basename(r['search_path'])}"
        copied.append(r2)
    return copied


def main() -> None:
    if os.path.isdir(os.path.join(DEV_DATA_DIR, "development")) and os.path.isfile(
        os.path.join(DEV_DATA_DIR, "development", "ground_truth.json")
    ):
        print(f"Expanded dev data already present at {DEV_DATA_DIR} - skipping regeneration.")
        return

    print(f"Generating {len(DEV_EXPANSION_FAMILIES)} new dev families (seed={DEV_EXPANSION_SEED}) -> {DEV_DATA_DIR}")
    generate_dataset(DEV_DATA_DIR, seed=DEV_EXPANSION_SEED, families=DEV_EXPANSION_FAMILIES,
                      only_splits=["development"], verbose=False)

    with open(os.path.join(DEV_DATA_DIR, "development", "ground_truth.json")) as f:
        new_records = json.load(f)

    prod_records = _copy_production_dev_families(os.path.join(DEV_DATA_DIR, "development"))
    combined = new_records + prod_records
    with open(os.path.join(DEV_DATA_DIR, "development", "ground_truth.json"), "w") as f:
        json.dump(combined, f, indent=2)

    families_present = sorted(set(r["structural_family"] for r in combined))
    print(f"Combined manifest: {len(combined)} pairs across {len(families_present)} families: {families_present}")


if __name__ == "__main__":
    main()
