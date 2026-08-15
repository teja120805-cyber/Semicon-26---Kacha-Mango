"""Targeted validation set for the finer_hypothesis_grid Phase 6
integration decision - deliberately over-represents the exact conditions
the grid change targets (rotation/scale-affected pairs), plus every other
condition requested (A-K), rather than relying on the frozen benchmark's
~20% rotation/scale-affected mix.

Independent seed (651900), distinct from every seed used anywhere else in
this project (production 777001, finer_hypothesis_grid fresh_data 424242,
accuracy_forensics 830001, learned_candidate_generator dev expansion
913700). Designed *before* looking at any prediction from either pipeline
variant - not tuned after the fact.

11 families x 12 pairs = 132 pairs, one family per lettered category so no
single easy/hard family can dominate the pooled result.
"""
from __future__ import annotations

VALIDATION_SEED = 651900
VALIDATION_SEED_B = 782411  # second, independent seed for the Phase 5 robustness check
N_PER_FAMILY = 12

TARGETED_FAMILIES: list[dict] = [
    # A. No structural boundary
    dict(name="A_no_boundary", split="validation", crop_mode="single_mat", n=N_PER_FAMILY, overrides={}),
    # B. High periodicity (densest pitch preset), no boundary, no drift
    dict(name="B_high_periodicity", split="validation", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_dense"}),
    # C. Low periodicity (loosest pitch preset), no boundary, no drift
    dict(name="C_low_periodicity", split="validation", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_legacy"}),
    # D. Rotation only
    dict(name="D_rotation", split="validation", crop_mode="random", n=N_PER_FAMILY,
         overrides={"_rotation_range": (-4.0, 4.0)}),
    # E. Scale variation only
    dict(name="E_scale", split="validation", crop_mode="random", n=N_PER_FAMILY,
         overrides={"_scale_range": (0.93, 1.07)}),
    # F. Rotation + scale interaction
    dict(name="F_rotation_scale", split="validation", crop_mode="random", n=N_PER_FAMILY,
         overrides={"_rotation_range": (-4.0, 4.0), "_scale_range": (0.93, 1.07)}),
    # G. Boundary-crossing references
    dict(name="G_boundary_crossing", split="validation", crop_mode="mat_boundary", n=N_PER_FAMILY, overrides={}),
    # H. Noise / degradation (heavy dose reduction + speckle + salt-pepper combined)
    dict(name="H_noise_degradation", split="validation", crop_mode="random", n=N_PER_FAMILY,
         overrides={"dose_search": 40.0, "speckle_sigma": 0.1, "salt_pepper_amount": 0.01}),
    # I. Boundary + rotation/scale
    dict(name="I_boundary_rotation_scale", split="validation", crop_mode="mat_boundary", n=N_PER_FAMILY,
         overrides={"_rotation_range": (-4.0, 4.0), "_scale_range": (0.93, 1.07)}),
    # J. Periodicity + rotation/scale
    dict(name="J_periodicity_rotation_scale", split="validation", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_dense", "_rotation_range": (-4.0, 4.0), "_scale_range": (0.93, 1.07)}),
    # K. Difficult interaction cases (hardest boundary type + max periodicity + drift + noise combined)
    dict(name="K_difficult_interaction", split="validation", crop_mode="same_preset_boundary", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_dense", "_rotation_range": (-4.0, 4.0), "_scale_range": (0.93, 1.07),
                    "dose_search": 60.0}),
]

# Sanity: 11 families x 12 = 132 pairs total.
assert len(TARGETED_FAMILIES) == 11
assert sum(f["n"] for f in TARGETED_FAMILIES) == 132
