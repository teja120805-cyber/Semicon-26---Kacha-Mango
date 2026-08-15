"""Instrumented evaluation for the finer_hypothesis_grid experiment:
same failure-location diagnostics as experiments/accuracy_forensics/harness.py
(candidate-in-pool, candidate rank/score, failure category), parameterized by
an explicit scale/rotation hypothesis grid so the identical instrumentation
can be applied to both the production 25-hypothesis grid and the candidate
81-hypothesis grid for a like-for-like comparison.

Never reads ground truth inside the pipeline itself - GT is used here only
to label the result afterward, exactly like evaluation/evaluate.py does.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from pipeline import candidate_generation, feature_extraction, ranking, refinement

GT_TOLERANCE_PX = 5.0
AMBIGUITY_MARGIN = 0.02


def _dist(x1, y1, x2, y2) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def _classify_failure(*, final_error_px, gt_in_pool, winner_is_gt_candidate, coarse_error_px,
                       winner_score, gt_candidate_score) -> str:
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


def instrumented_localize(reference_img: np.ndarray, search_img: np.ndarray, gt_x: float, gt_y: float,
                           scale_hypotheses: Optional[tuple] = None,
                           rotation_hypotheses: Optional[tuple] = None) -> dict:
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)
    kwargs = {}
    if scale_hypotheses is not None:
        kwargs["scale_hypotheses"] = scale_hypotheses
    if rotation_hypotheses is not None:
        kwargs["rotation_hypotheses"] = rotation_hypotheses

    t0 = time.perf_counter()
    raw_candidates = candidate_generation.build_candidate_pool(reference, search, **kwargs)
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

    gt_candidate_rank = None
    gt_candidate_score = None
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
        "coarse_error_px": coarse_error_px, "runtime_s": runtime_s,
        "num_candidates_raw": len(raw_candidates), "num_candidates_dedup": len(candidates),
        "winner_score": float(winner.score), "top2_score_margin": top2_margin,
        "gt_in_pool": gt_in_pool, "gt_nearest_candidate_dist_px": nearest_dist,
        "gt_candidate_rank": gt_candidate_rank, "gt_candidate_score": gt_candidate_score,
        "winner_is_gt_candidate": winner_is_gt_candidate, "failure_location": failure_location,
    }
