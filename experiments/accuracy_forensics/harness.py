"""Instrumented evaluation for the Phase 3 accuracy forensics experiment.

`pipeline/localize.py` is deliberately ground-truth-blind (see its module
docstring) - correct for production, but it means the production code path
cannot answer "where in the pipeline did this pair actually fail?" This
module runs the *same* candidate_generation/ranking/refinement calls
`localize()` makes, in the same order, but additionally compares the
resulting candidate pool against ground truth for diagnostic purposes only.
Ground truth is read here, exactly like evaluation/evaluate.py already does
for scoring - never fed back into candidate_generation/ranking/refinement's
own decisions.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from generator import dataset_generator
from pipeline import candidate_generation, ranking, refinement

GT_TOLERANCE_PX = 5.0
AMBIGUITY_MARGIN = 0.02  # score-margin below which a ranking "miss" is called genuine ambiguity, not a ranking bug


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def instrumented_localize(reference_img: np.ndarray, search_img: np.ndarray,
                           gt_x: float, gt_y: float) -> dict:
    """Runs one localization and returns final prediction/error plus full
    candidate-pool diagnostics, in a single pass (no duplicate correlation
    work relative to pipeline.localize.localize)."""
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)

    t0 = time.perf_counter()
    raw_candidates = candidate_generation.build_candidate_pool(reference, search)
    candidates = candidate_generation.deduplicate_by_location(raw_candidates)
    ranked = ranking.rank_classical(candidates)
    winner = ranked[0]
    coarse_error_px = _dist(winner.x, winner.y, gt_x, gt_y)
    refined_x, refined_y = refinement.refine(reference, search, winner)
    runtime_s = time.perf_counter() - t0
    final_error_px = _dist(refined_x, refined_y, gt_x, gt_y)

    dists = sorted(((_dist(c.x, c.y, gt_x, gt_y), i) for i, c in enumerate(candidates)), key=lambda t: t[0])
    nearest_dist, nearest_idx = dists[0]
    gt_in_pool = nearest_dist <= GT_TOLERANCE_PX
    gt_candidate = candidates[nearest_idx] if gt_in_pool else None

    gt_candidate_rank: Optional[int] = None
    gt_candidate_score: Optional[float] = None
    if gt_candidate is not None:
        for rank_i, c in enumerate(ranked):
            if c is gt_candidate:
                gt_candidate_rank = rank_i + 1
                gt_candidate_score = float(c.score)
                break

    winner_is_gt_candidate = gt_in_pool and (gt_candidate is winner)
    top2_margin = float(ranked[0].score - ranked[1].score) if len(ranked) > 1 else float("nan")

    failure_location = _classify_failure(
        final_error_px=final_error_px, gt_in_pool=gt_in_pool,
        winner_is_gt_candidate=winner_is_gt_candidate, coarse_error_px=coarse_error_px,
        winner_score=float(winner.score), gt_candidate_score=gt_candidate_score,
    )

    return {
        "pred_x": refined_x, "pred_y": refined_y, "error_px": final_error_px,
        "coarse_error_px": coarse_error_px, "refinement_shift_px": _dist(winner.x, winner.y, refined_x, refined_y),
        "runtime_s": runtime_s,
        "num_candidates_raw": len(raw_candidates), "num_candidates_dedup": len(candidates),
        "winner_score": float(winner.score), "winner_scale": winner.scale, "winner_rotation_deg": winner.rotation_deg,
        "top2_score_margin": top2_margin,
        "gt_in_pool": gt_in_pool, "gt_nearest_candidate_dist_px": nearest_dist,
        "gt_candidate_rank": gt_candidate_rank, "gt_candidate_score": gt_candidate_score,
        "winner_is_gt_candidate": winner_is_gt_candidate,
        "failure_location": failure_location,
    }


def _classify_failure(*, final_error_px: float, gt_in_pool: bool, winner_is_gt_candidate: bool,
                       coarse_error_px: float, winner_score: float,
                       gt_candidate_score: Optional[float]) -> str:
    """Five-way taxonomy from the Phase 3 brief: where did a failed pair
    actually go wrong? Only meaningful when final_error_px exceeds the
    project's @5px tolerance; successes are labeled explicitly rather than
    left blank so downstream grouping never has to special-case them."""
    if final_error_px <= GT_TOLERANCE_PX:
        return "success"
    if not gt_in_pool:
        return "candidate_generation"
    if not winner_is_gt_candidate:
        if gt_candidate_score is not None and (winner_score - gt_candidate_score) < AMBIGUITY_MARGIN:
            return "genuine_ambiguity"
        return "candidate_ranking"
    if coarse_error_px <= GT_TOLERANCE_PX < final_error_px:
        return "refinement"
    return "unexplained"


def generate_and_localize(pair_index: int, seed: int, family: dict) -> dict:
    """Generate one pair with `generator.dataset_generator.generate_pair`
    (no images written to disk - forensics sweeps are analysis artifacts,
    reproducible on demand from (seed, pair_index, family) alone) and run
    the instrumented pipeline on it. Returns one flat record combining
    generator metadata and pipeline diagnostics."""
    ref_img, search_img, meta = dataset_generator.generate_pair(pair_index, seed, family)
    diag = instrumented_localize(ref_img, search_img, meta["gt_x"], meta["gt_y"])
    dp = meta["degradation_params"]
    record = {
        "pair_id": meta["pair_id"], "structural_family": family["name"], "crop_mode": family["crop_mode"],
        "gt_x": meta["gt_x"], "gt_y": meta["gt_y"],
        "num_mats": meta["num_mats"], "presets": ";".join(meta["presets"]),
        "crosses_mat_boundary": meta["crosses_mat_boundary"], "crosses_strip_boundary": meta["crosses_strip_boundary"],
        "same_preset_boundary": meta["same_preset_boundary"],
        "periodicity_score": meta["periodicity_score"], "uniqueness_score": meta["uniqueness_score"],
        "rotation_deg": dp["rotation_deg"], "extra_scale": dp["extra_scale"], "dose_search": dp["dose_search"],
        "dose_reference": dp["dose_reference"], "shear_amplitude_px": dp["shear_amplitude_px"],
        "jitter_std_px": dp["jitter_std_px"], "blur_search_effective_px": dp["blur_search_effective_px"],
        "collapse_enabled": dp["collapse_enabled"], "collapse_threshold_nm": dp["collapse_threshold_nm"],
        "linewidth_bias_nm": dp["linewidth_bias_nm"], "corner_rounding_px": dp["corner_rounding_px"],
        "barrel_k": dp["barrel_k"], "vignette_strength": dp["vignette_strength"], "gamma": dp["gamma"],
        "charging_prob": dp["charging_prob"], "speckle_sigma": dp["speckle_sigma"],
        "salt_pepper_amount": dp["salt_pepper_amount"], "astigmatism_ratio": dp["astigmatism_ratio"],
    }
    record.update(diag)
    return record


def make_family(name: str, crop_mode: str, overrides: dict) -> dict:
    return {"name": name, "split": "forensics", "crop_mode": crop_mode, "overrides": overrides}
