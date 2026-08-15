"""Expanded, dev-only training data for the learned candidate generator.

Feasibility audit (see REPORT.md) confirmed the production `development`
split (24 pairs, 3 families) is the direct, previously-diagnosed cause of
`embedding_reranker_v1`'s overfitting (72 triplets, 2 families actually used
for training). Per reports/DATASET_AUDIT.md section 5's bounded
recommendation, this generates a targeted expansion - NOT touching
data/development/ or any frozen-benchmark split - focused on the failure
modes identified in reports/ACCURACY_FORENSICS.md: periodicity (all 6
presets), boundary proximity, rotation/scale interaction, and noise/
distortion combinations.

Uses a dedicated seed (913700), distinct from both the production seed
(777001) and the finer_hypothesis_grid fresh-data seed (424242), and lives
entirely under experiments/learned_candidate_generator/dev_data/ - deleting
this experiment directory removes all of it, nothing leaks into data/.
"""
from __future__ import annotations

DEV_EXPANSION_SEED = 913700
N_PER_FAMILY = 10

DEV_EXPANSION_FAMILIES: list[dict] = [
    # Periodicity coverage - one family per preset, deep single-mat (no boundary)
    dict(name="dev2_single_mat_relaxed", split="development", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_relaxed"}),
    dict(name="dev2_single_mat_narrow", split="development", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_narrow"}),
    dict(name="dev2_single_mat_compact", split="development", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_compact"}),
    dict(name="dev2_single_mat_legacy", split="development", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_legacy"}),
    dict(name="dev2_single_mat_nominal", split="development", crop_mode="single_mat", n=N_PER_FAMILY,
         overrides={"force_preset": "mat_nominal"}),
    # Boundary proximity coverage
    dict(name="dev2_mat_boundary", split="development", crop_mode="mat_boundary", n=N_PER_FAMILY, overrides={}),
    dict(name="dev2_same_preset_boundary", split="development", crop_mode="same_preset_boundary",
         n=N_PER_FAMILY, overrides={}),
    dict(name="dev2_multi_mat", split="development", crop_mode="multi_mat", n=N_PER_FAMILY, overrides={}),
    dict(name="dev2_strip_center", split="development", crop_mode="strip_center", n=N_PER_FAMILY, overrides={}),
    # Rotation/scale interaction coverage
    dict(name="dev2_rotation_drift", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"_rotation_range": (-4.0, 4.0)}),
    dict(name="dev2_scale_drift", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"_scale_range": (0.93, 1.07)}),
    dict(name="dev2_combined_drift", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"_rotation_range": (-4.0, 4.0), "_scale_range": (0.93, 1.07)}),
    # Noise / distortion combinations
    dict(name="dev2_heavy_noise", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"dose_search": 40.0}),
    dict(name="dev2_vignette_gamma", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"vignette_strength": 0.35, "gamma": 1.3}),
    dict(name="dev2_barrel_charging", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"barrel_k": 0.003, "charging_prob": 0.015, "charging_intensity": 60.0}),
    dict(name="dev2_speckle_saltpepper", split="development", crop_mode="random", n=N_PER_FAMILY,
         overrides={"speckle_sigma": 0.12, "salt_pepper_amount": 0.01}),
]

# 16 families x 10 = 160 new pairs, + the 24 existing production development
# pairs (dev_strip_anchor/dev_single_mat/dev_dense_periodic) if pooled = 184
# total - within the ~150-180 bounded recommendation, not "thousands of
# random pairs".

# Held-out-from-training internal families for early stopping, analogous to
# model/train.py's EARLY_STOP_FAMILIES pattern but drawn from the new,
# larger pool instead of just one family - one family per major axis so
# early stopping tracks generalization across all axes, not just one.
INTERNAL_EARLY_STOP_FAMILIES = (
    "dev2_single_mat_compact",  # periodicity
    "dev2_mat_boundary",        # boundary
    "dev2_combined_drift",      # rotation+scale
    "dev2_speckle_saltpepper",  # noise
)
ALL_FAMILY_NAMES = [f["name"] for f in DEV_EXPANSION_FAMILIES] + [
    "dev_strip_anchor", "dev_single_mat", "dev_dense_periodic",
]
TRAIN_FAMILIES = [n for n in ALL_FAMILY_NAMES if n not in INTERNAL_EARLY_STOP_FAMILIES]
