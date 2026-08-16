#!/usr/bin/env python
"""Generates the expanded development-only training set recommended (but
never executed) in reports/DATASET_AUDIT.md section 5: ~150-180 pairs
across ~12-15 structural family variants covering every major axis already
validated elsewhere in the benchmark (boundary-crossing, same-preset-
boundary, multi-mat, rotation drift, scale drift, heavy noise, and more),
roughly 10-12 pairs per family - vs. the production `development` split's
24 pairs / 3 families, the direct diagnosed cause of embedding_reranker_v1's
72-triplet overfitting.

Written to THIS experiment's own data/ directory, split name "development"
(so model/dataset.py::TripletPatchDataset, which always reads
"<data_root>/development/ground_truth.json", can be pointed at it via
data_root=experiments/learned_reranker_v2/data - never overwrites or reads
the production data/development/ split). Uses seed 910001 - distinct from
production (777001), reports/ACCURACY_FORENSICS.md's dedicated sweep seed
(830001), and every other experiment's seed - so there is zero possibility
of RNG-level overlap with any existing split, by construction (every pair's
RNG is default_rng([seed, family_salt(split, family_name), pair_index]) -
see generator/dataset_generator.py's module docstring).

Calls generator.dataset_generator.generate_dataset unmodified - never
forks or edits generator/. Never touches production data/.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)

from generator.dataset_generator import generate_dataset  # noqa: E402

SEED = 910001
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
PAIRS_PER_FAMILY = 12

# Mirrors every major structural/degradation axis already validated
# elsewhere in the benchmark (validation/held_out/challenge's own family
# definitions in generator/dataset_generator.py::FAMILIES), renamed with an
# "ldev_" prefix and all placed under split="development" so this is a
# private, non-overlapping expansion - never the same family name as any
# production split, and a different seed regardless.
FAMILIES = [
    dict(name="ldev_strip_anchor", split="development", crop_mode="strip_center", n=PAIRS_PER_FAMILY,
         overrides={}, description="Non-periodic landmark baseline."),
    dict(name="ldev_single_mat", split="development", crop_mode="single_mat", n=PAIRS_PER_FAMILY,
         overrides={}, description="Deep in-mat, baseline periodic case."),
    dict(name="ldev_dense_periodic", split="development", crop_mode="single_mat", n=PAIRS_PER_FAMILY,
         overrides={"force_preset": "mat_dense"}, description="Densest pitch, maximum ambiguity."),
    dict(name="ldev_mat_boundary", split="development", crop_mode="mat_boundary", n=PAIRS_PER_FAMILY,
         overrides={}, description="Boundary-crossing, any presets."),
    dict(name="ldev_same_preset_boundary", split="development", crop_mode="same_preset_boundary",
         n=PAIRS_PER_FAMILY, overrides={}, description="Hardest boundary case - same preset both sides."),
    dict(name="ldev_multi_mat", split="development", crop_mode="multi_mat", n=PAIRS_PER_FAMILY,
         overrides={}, description="Touches 3+ mats."),
    dict(name="ldev_rotation_drift", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"_rotation_range": (-4.0, 4.0)}, description="Residual rotation drift."),
    dict(name="ldev_scale_drift", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"_scale_range": (0.90, 1.10)}, description="Residual scale drift, literal 9:1-11:1."),
    dict(name="ldev_heavy_noise", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"dose_search": 40.0, "read_noise_sigma_search": 10.0}, description="Acquisition-noise stress."),
    dict(name="ldev_linewidth_bias", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"linewidth_bias_nm": 4.0}, description="Deterministic global CD/etch bias."),
    dict(name="ldev_vignette_gamma", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"vignette_strength": 0.35, "gamma": 1.3}, description="Radiometric falloff + gain nonlinearity."),
    dict(name="ldev_combined_acquisition", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"dose_search": 60.0, "_rotation_range": (-4.0, 4.0), "_scale_range": (0.90, 1.10)},
         description="Rotation + scale + noise combined."),
    dict(name="ldev_barrel_charging", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"barrel_k": 0.003, "charging_prob": 0.015, "charging_intensity": 60.0},
         description="Scan-linearity distortion + charging streaks."),
    dict(name="ldev_speckle_saltpepper", split="development", crop_mode="random", n=PAIRS_PER_FAMILY,
         overrides={"speckle_sigma": 0.12, "salt_pepper_amount": 0.01}, description="Speckle + impulse noise."),
    dict(name="ldev_worst_case", split="development", crop_mode="same_preset_boundary", n=PAIRS_PER_FAMILY,
         overrides={"force_preset": "mat_dense", "mat_size_nm": 3200, "dose_search": 45.0,
                    "_rotation_range": (-4.0, 4.0), "_scale_range": (0.90, 1.10),
                    "barrel_k": 0.002, "speckle_sigma": 0.08},
         description="Every hard axis combined on the hardest boundary mode."),
]


def main() -> None:
    total = sum(f["n"] for f in FAMILIES)
    print(f"Generating {total} pairs across {len(FAMILIES)} families (seed={SEED}) -> {OUT_DIR}")
    generate_dataset(OUT_DIR, seed=SEED, families=FAMILIES, verbose=True)
    print("Done.")


if __name__ == "__main__":
    main()
