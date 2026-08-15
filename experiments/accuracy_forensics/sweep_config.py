"""Controlled single-factor and interaction sweep definitions for the Phase 3
accuracy forensics experiment.

Every level within one factor uses the *same* pair_index range, and
`generator.dataset_generator.generate_pair`'s RNG is `default_rng(seed *
1_000_003 + pair_index)` with no extra draws consumed for a directly-set
parameter override (only the `_rotation_range`/`_scale_range` *sampling*
helpers consume extra draws - we set exact values instead). That means, for
any factor whose levels all use crop_mode="random" and don't touch
`force_preset`, every level of that factor is evaluated against the *same*
underlying macro canvas and crop location at a given pair_index - a paired,
repeated-measures design that isolates the swept parameter from scene-to-
scene confounds. Levels that vary crop_mode or force_preset (periodicity,
boundary) necessarily also vary structure, since that IS the factor under
test.

This dataset is independent from, and does not replace, the frozen
production benchmark under data/ (see reports/DATASET_AUDIT.md section 4).
It uses its own dedicated seed so it never collides with the production
seed (777001).
"""
from __future__ import annotations

FORENSICS_SEED = 830001


def lvl(label: str, overrides: dict | None = None, crop_mode: str = "random") -> dict:
    return {"label": label, "crop_mode": crop_mode, "overrides": overrides or {}}


# Ordered loosest (largest pitch, least periodic) -> densest (smallest pitch,
# most periodic), per generator/mat_generator.py::DRAM_MAT_PRESETS.
_PRESETS_BY_PITCH = ["mat_legacy", "mat_relaxed", "mat_nominal", "mat_narrow", "mat_compact", "mat_dense"]

# name -> {"n": pairs per level, "levels": [lvl(...), ...]}
SINGLE_FACTOR_SWEEPS: dict[str, dict] = {
    "rotation_deg": {
        "n": 40,
        "levels": [lvl(f"{v:g}deg", {"rotation_deg": v}) for v in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)],
    },
    "extra_scale": {
        "n": 40,
        "levels": [lvl(f"{v:.2f}x", {"extra_scale": v}) for v in (1.00, 1.02, 1.04, 1.06, 1.08)],
    },
    "periodicity_preset": {
        "n": 40,
        "levels": [lvl(p, {"force_preset": p}, crop_mode="single_mat") for p in _PRESETS_BY_PITCH],
    },
    "boundary_condition": {
        "n": 40,
        "levels": [
            lvl("no_boundary_single_mat", {}, crop_mode="single_mat"),
            lvl("strip_crossing", {}, crop_mode="strip_center"),
            lvl("mat_boundary", {}, crop_mode="mat_boundary"),
            lvl("same_preset_boundary", {}, crop_mode="same_preset_boundary"),
            lvl("multi_mat_3plus", {}, crop_mode="multi_mat"),
        ],
    },
    "search_dose": {
        "n": 40,
        "levels": [lvl(f"dose{int(v)}", {"dose_search": v}) for v in (220.0, 150.0, 100.0, 60.0, 40.0, 25.0)],
    },
    "raster_drift_shear": {
        "n": 30,
        "levels": [lvl(f"shear{v:g}px", {"shear_amplitude_px": v}) for v in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)],
    },
    "row_jitter": {
        "n": 30,
        "levels": [lvl(f"jitter{v:g}px", {"jitter_std_px": v}) for v in (0.0, 0.2, 0.4, 0.8, 1.5, 2.5)],
    },
    "beam_spot_size": {
        "n": 20,
        "levels": [lvl(f"blur{v:g}px", {"blur_search_effective_px": v}) for v in (0.5, 1.0, 1.5, 2.5)],
    },
    "pattern_collapse_threshold": {
        "n": 20,
        "levels": [
            lvl("off", {"collapse_enabled": False}),
            lvl("thr5nm", {"collapse_enabled": True, "collapse_threshold_nm": 5.0}),
            lvl("thr10nm", {"collapse_enabled": True, "collapse_threshold_nm": 10.0}),
            lvl("thr20nm", {"collapse_enabled": True, "collapse_threshold_nm": 20.0}),
        ],
    },
    "reference_dose": {
        "n": 20,
        "levels": [lvl(f"dose{int(v)}", {"dose_reference": v}) for v in (3000.0, 1800.0, 900.0, 400.0)],
    },
    "linewidth_cd_bias": {
        "n": 20,
        "levels": [lvl(f"{v:g}nm", {"linewidth_bias_nm": v}) for v in (0.0, 2.0, 4.0, 8.0)],
    },
    "corner_rounding": {
        "n": 20,
        "levels": [lvl(f"{v:g}px", {"corner_rounding_px": v}) for v in (0.0, 1.0, 2.0, 4.0)],
    },
    "barrel_pincushion": {
        "n": 20,
        "levels": [lvl(f"k{v:g}", {"barrel_k": v}) for v in (-0.006, -0.003, 0.0, 0.003, 0.006)],
    },
    "vignette_strength": {
        "n": 20,
        "levels": [lvl(f"{v:g}", {"vignette_strength": v}) for v in (0.0, 0.15, 0.35, 0.55)],
    },
    "gamma": {
        "n": 20,
        "levels": [lvl(f"{v:g}", {"gamma": v}) for v in (0.7, 0.85, 1.0, 1.3, 1.6)],
    },
    "charging_streaks": {
        "n": 20,
        "levels": [
            lvl(f"prob{v:g}", {"charging_prob": v, "charging_intensity": 60.0 if v > 0 else 0.0})
            for v in (0.0, 0.005, 0.015, 0.03)
        ],
    },
    "speckle_sigma": {
        "n": 20,
        "levels": [lvl(f"{v:g}", {"speckle_sigma": v}) for v in (0.0, 0.04, 0.08, 0.12)],
    },
    "salt_pepper": {
        "n": 20,
        "levels": [lvl(f"{v:g}", {"salt_pepper_amount": v}) for v in (0.0, 0.0025, 0.01, 0.02)],
    },
    "beam_astigmatism": {
        "n": 20,
        "levels": [lvl(f"{v:g}", {"astigmatism_ratio": v}) for v in (1.0, 1.3, 1.6, 2.0)],
    },
}

# Factorial interaction sweeps. Each cell is one (factor combo) point; "n"
# pairs are generated per cell using consecutive pair_index within that cell
# (cells do not share pair_index with each other, so cells are independent
# scene samples - only sweeps *within* one factor's levels are paired).
INTERACTION_SWEEPS: dict[str, dict] = {
    "rotation_x_scale": {
        "n": 25, "crop_modes": ["random"],
        "grid": {"rotation_deg": [0.0, 4.0], "extra_scale": [1.0, 1.07]},
    },
    "rotation_x_boundary": {
        "n": 25, "crop_modes": ["single_mat", "mat_boundary"],
        "grid": {"rotation_deg": [0.0, 4.0]},
    },
    "scale_x_boundary": {
        "n": 25, "crop_modes": ["single_mat", "mat_boundary"],
        "grid": {"extra_scale": [1.0, 1.07]},
    },
    "noise_x_rotation": {
        "n": 25, "crop_modes": ["random"],
        "grid": {"dose_search": [220.0, 60.0], "rotation_deg": [0.0, 4.0]},
    },
    "noise_x_scale": {
        "n": 25, "crop_modes": ["random"],
        "grid": {"dose_search": [220.0, 60.0], "extra_scale": [1.0, 1.07]},
    },
    "rotation_x_scale_x_boundary": {
        "n": 15, "crop_modes": ["single_mat", "mat_boundary"],
        "grid": {"rotation_deg": [0.0, 4.0], "extra_scale": [1.0, 1.07]},
    },
}
