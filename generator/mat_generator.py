"""Named DRAM sub-array presets and the per-mat rendering entry point.

Six presets spanning realistic folded-bitline pitch/critical-dimension
combinations, each with an internally consistent ~2:3 word:bit pitch ratio
(typical of a folded-bitline 6F^2 cell). Presets vary geometry only - never
rendered brightness - matching the same design choice independently used
elsewhere in this project (see reports/V2_ARCHITECTURE_PLAN.md section 2): a mat's
identity must come from real structural difference, not an injected
brightness fingerprint.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import pattern_renderer as pr

DRAM_MAT_PRESETS: dict[str, dict[str, float]] = {
    "mat_dense":   {"feature_nm": 24, "word_pitch_nm": 48,  "bit_pitch_nm": 72},
    "mat_narrow":  {"feature_nm": 32, "word_pitch_nm": 64,  "bit_pitch_nm": 96},
    "mat_nominal": {"feature_nm": 40, "word_pitch_nm": 80,  "bit_pitch_nm": 120},
    "mat_relaxed": {"feature_nm": 50, "word_pitch_nm": 100, "bit_pitch_nm": 150},
    "mat_compact": {"feature_nm": 28, "word_pitch_nm": 56,  "bit_pitch_nm": 84},
    "mat_legacy":  {"feature_nm": 68, "word_pitch_nm": 136, "bit_pitch_nm": 204},
}

PRESET_NAMES = list(DRAM_MAT_PRESETS.keys())


def pick_mat_preset(rng: np.random.Generator, force: Optional[str] = None) -> str:
    if force is not None:
        if force not in DRAM_MAT_PRESETS:
            raise ValueError(f"Unknown preset '{force}'. Options: {PRESET_NAMES}")
        return force
    return PRESET_NAMES[int(rng.integers(0, len(PRESET_NAMES)))]


def generate_mat(size_px: int, preset: str, rng: np.random.Generator, *,
                  feature_size_scale: float = 1.0,
                  linewidth_bias_nm: float = 0.0,
                  collapse_enabled: bool = False,
                  collapse_threshold_nm: float = 10.0,
                  collapse_prob: float = 0.65,
                  corner_rounding_px: float = 0.0) -> np.ndarray:
    """Render one mat using `preset`'s geometry with its own independent RNG.

    The caller is expected to pass a child RNG spawned specifically for this
    mat (see macro_layout.generate_macro_canvas) so that two mats - even
    sharing the same preset - are statistically independent realizations,
    never correlated copies.

    `feature_size_scale` scales word/bit pitch and feature width
    proportionally (1.0 = the preset as tabulated above) - a continuous
    "explore other process nodes" control distinct from picking a different
    discrete preset, matching the reference/demo generator's "Feature size
    scale" slider (V2 previously had no analog for this - see
    reports/DATASET_AUDIT.md section 3).
    """
    p = DRAM_MAT_PRESETS[preset]
    return pr.render_dram_cell_array(
        size_px,
        p["word_pitch_nm"] * feature_size_scale,
        p["bit_pitch_nm"] * feature_size_scale,
        p["feature_nm"] * feature_size_scale,
        rng,
        linewidth_bias_nm=linewidth_bias_nm,
        collapse_enabled=collapse_enabled,
        collapse_threshold_nm=collapse_threshold_nm,
        collapse_prob=collapse_prob,
        corner_rounding_px=corner_rounding_px,
    )
