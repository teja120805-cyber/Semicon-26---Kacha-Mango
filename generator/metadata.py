"""Per-pair metadata assembly and structural difficulty scores.

Both derived scores are pure functions of macro-layout geometry (pitch,
boundary membership) computed from `macro_layout.crop_membership` - never
of any localization algorithm's output - so they describe how hard a crop
*should* be structurally, without leaking anything about how hard it turned
out to be for a specific matcher.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .mat_generator import DRAM_MAT_PRESETS

_WORD_PITCHES = [p["word_pitch_nm"] for p in DRAM_MAT_PRESETS.values()]
PITCH_MIN_NM = float(min(_WORD_PITCHES))
PITCH_MAX_NM = float(max(_WORD_PITCHES))


def periodicity_score(mean_word_pitch_nm: Optional[float]) -> float:
    """Higher = tighter/denser pitch = more locally self-similar structure."""
    if mean_word_pitch_nm is None:
        return 0.0
    span = PITCH_MAX_NM - PITCH_MIN_NM
    return float(np.clip((PITCH_MAX_NM - mean_word_pitch_nm) / span, 0.0, 1.0))


def uniqueness_score(crosses_strip: bool, crosses_mat: bool, num_mats: int) -> float:
    """Higher = more structurally distinctive crop (strip content, a mat
    boundary, or several mats touched all make a crop easier to place
    uniquely than deep, single-mat, single-preset territory)."""
    score = 0.0
    if crosses_strip:
        score += 0.6
    if crosses_mat:
        score += 0.3
    score += min(max(num_mats - 1, 0), 2) * 0.05
    return float(min(score, 1.0))


def build_metadata(*, pair_id: str, seed: int, generator_version: str,
                    reference_path: str, search_path: str,
                    gt_x: float, gt_y: float, membership: dict,
                    family_name: str, split: str, crop_mode: str,
                    fallback_used: bool, crop_tries: int,
                    degradation_params: dict, mat_size_nm: int, strip_width_nm: int,
                    preset_word_pitches: list[float]) -> dict:
    mean_pitch = float(np.mean(preset_word_pitches)) if preset_word_pitches else None
    return {
        "pair_id": pair_id,
        "seed": seed,
        "generator_version": generator_version,
        "reference_path": reference_path,
        "search_path": search_path,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "structural_family": family_name,
        "split": split,
        "crop_mode": crop_mode,
        "crop_mode_fallback_used": fallback_used,
        "crop_mode_tries": crop_tries,
        "mat_ids": membership["mat_ids"],
        "num_mats": membership["num_mats"],
        "presets": membership["presets"],
        "crosses_mat_boundary": membership["crosses_mat_boundary"],
        "crosses_strip_boundary": membership["crosses_strip_boundary"],
        "same_preset_boundary": membership["same_preset_boundary"],
        "periodicity_score": periodicity_score(mean_pitch),
        "uniqueness_score": uniqueness_score(
            membership["crosses_strip_boundary"], membership["crosses_mat_boundary"], membership["num_mats"]
        ),
        "mat_size_nm": mat_size_nm,
        "strip_width_nm": strip_width_nm,
        "degradation_params": degradation_params,
        "timestamp": time.time(),
    }
