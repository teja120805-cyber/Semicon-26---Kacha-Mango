"""Generate the tuning and validation surfaces this experiment uses.

Why this exists rather than just using the frozen `development` split:

`reports/PROJECT_STATUS.md` records a known, unresolved methodology defect -
**`development` contains no degraded-acquisition family**, so every dev-only
hyperparameter sweep in this project has been structurally blind to
over-smoothing and noise-amplification damage. P1 (`experiments/parallel_pipeline/`)
failed for exactly that class of reason: it looked plausible on clean pairs
and broke noisy ones. Tuning P3's `alpha` on clean pairs alone would walk
into the same hole.

So the tuning surface here is `development` (frozen, unchanged, 24 pairs)
PLUS a freshly generated 40-pair set at a new seed that deliberately spans
the degraded families `development` lacks. That fresh set is NEW data, not a
frozen scoring surface, so tuning on it is not benchmark mining - but it is
a deviation from "tune on development only" and is called out as such in
REPORT.md rather than buried.

`validation_fresh` uses a third, different seed and is never read during
tuning. Seeds are fixed here so any run is reproducible.

    python -m experiments.discriminability_weighted.make_datasets
"""
from __future__ import annotations

import os

from generator.dataset_generator import generate_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))

TUNE_SEED = 314159
VALIDATE_SEED = 271828

# Deliberately spans the axes `development` does not: impulse/speckle noise,
# low dose, radiometric falloff, and combined rotation+scale drift. Family
# names mirror the frozen benchmark's so per-family comparisons line up, but
# the `split` key is "development" throughout because evaluation/evaluate.py
# only knows four split names and this is a tuning surface, not a new split.
DEGRADED_TUNING_FAMILIES = [
    dict(name="tune_dense_periodic", split="development", crop_mode="single_mat", n=6,
         overrides={"force_preset": "mat_dense"},
         description="Densest pitch, deep in-mat - P3's primary target."),
    dict(name="tune_same_preset_boundary", split="development", crop_mode="same_preset_boundary", n=6,
         overrides={}, description="Hardest boundary mode - the aperiodic content P3 up-weights."),
    dict(name="tune_heavy_noise", split="development", crop_mode="random", n=6,
         overrides={"dose_search": 40.0, "read_noise_sigma_search": 10.0},
         description="Pure acquisition noise - the axis development is blind to."),
    dict(name="tune_speckle_saltpepper", split="development", crop_mode="random", n=6,
         overrides={"speckle_sigma": 0.12, "salt_pepper_amount": 0.01},
         description="Impulse + multiplicative noise - non-Gaussian, where P1 broke."),
    dict(name="tune_vignette_gamma", split="development", crop_mode="random", n=6,
         overrides={"vignette_strength": 0.35, "gamma": 1.3},
         description="Non-stationary radiometric corruption."),
    dict(name="tune_combined_drift", split="development", crop_mode="random", n=5,
         overrides={"dose_search": 60.0, "_rotation_range": (-4.0, 4.0), "_scale_range": (0.90, 1.10)},
         description="Rotation + scale drift + noise combined."),
    dict(name="tune_worst_case", split="development", crop_mode="same_preset_boundary", n=5,
         overrides={"force_preset": "mat_dense", "mat_size_nm": 3200, "dose_search": 45.0,
                    "_rotation_range": (-4.0, 4.0), "_scale_range": (0.90, 1.10),
                    "barrel_k": 0.002, "speckle_sigma": 0.08},
         description="Every hard axis at once."),
]


def _rename(families: list[dict], prefix: str) -> list[dict]:
    out = []
    for f in families:
        g = dict(f)
        g["name"] = f["name"].replace("tune_", prefix)
        out.append(g)
    return out


def main() -> None:
    tune_root = os.path.join(ROOT, "data", "tune_degraded")
    val_root = os.path.join(ROOT, "data", "validate_fresh")

    print(f"=== tuning surface (seed {TUNE_SEED}) -> {tune_root}")
    generate_dataset(tune_root, seed=TUNE_SEED, families=DEGRADED_TUNING_FAMILIES, verbose=False)

    print(f"=== validation surface (seed {VALIDATE_SEED}) -> {val_root}")
    generate_dataset(val_root, seed=VALIDATE_SEED,
                     families=_rename(DEGRADED_TUNING_FAMILIES, "val_"), verbose=False)

    for label, root in (("tune_degraded", tune_root), ("validate_fresh", val_root)):
        n = len([f for f in os.listdir(os.path.join(root, "development")) if f.endswith("_search.png")])
        print(f"{label}: {n} pairs")


if __name__ == "__main__":
    main()
