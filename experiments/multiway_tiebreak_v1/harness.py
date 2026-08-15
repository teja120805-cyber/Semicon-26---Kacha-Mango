"""The candidate A6 tie-break mechanism and its evaluation harness.

Design rationale (see reports/HACKATHON_COMPLIANCE_CHECKLIST.md item A6 and
experiments/center_tiebreak_v2/REPORT.md for the two rejected precursors):

Both prior attempts widened the shipped tie-break's trigger (`TIE_SCORE_EPSILON`,
1e-6, provably inert) using ONLY a pairwise top1/top2 score-gap threshold -
one as a fixed absolute margin (0.001), one considered as a relative
margin. Both were tested against real per-pair data
(experiments/center_tiebreak_v2/outputs/*.csv, analyzed directly in this
conversation) and both failed for the same underlying reason: ZNCC score
alone cannot distinguish a genuine multi-way tie (several structurally
near-identical locations - the periodicity pileups documented in
reports/ACCURACY_FORENSICS.md, e.g. dev_dense_periodic_* pairs with
0.002-0.007 score margins across MANY candidates) from a coincidental
pairwise closeness between two otherwise-unrelated locations, one of which
is simply wrong (a candidate_generation failure, or an isolated
candidate_ranking failure). Both harmed and rescued cases in the rejected
experiment showed the identical "isolated pair, big gap to 3rd place"
score signature - the score alone carries no further information to
split them.

This version adds two structural conditions ON TOP OF a widened score-gap
epsilon, using signals a coincidental pairwise near-tie does NOT share with
a genuine periodicity pileup:

  1. `min_group_size`: require >= N candidates within `tie_score_epsilon` of
     the top score (not just 2). A genuine periodic lattice produces many
     near-identical peaks; a coincidental pairwise closeness produces
     exactly 2. This directly targets the failure mode identified in this
     conversation's own analysis of the rejected v2 experiment - its 0.001
     threshold never even reached this regime (all 10 of its flips were
     pairwise, tie_len==2 by construction), so this condition is genuinely
     untested by that experiment, not re-litigating a settled question.
  2. `max_spread_px`: even among >=N tied-by-score candidates, reject the
     group (do nothing - fall through to the untouched classical ranking)
     if the tied group's spatial extent exceeds this bound. The single
     worst catastrophic regression in the rejected v2 experiment
     (ho_vignette_gamma_005, +497px) had a top1/top2 distance of 524.9px -
     a clear outlier against the 40-115px range of every other flip in
     that run. A real periodic lattice repeats within a bounded local
     neighborhood, not across the whole 1000px canvas, so this bound
     targets exactly the failure mode that produced that experiment's
     single worst outcome without needing to know in advance which
     specific pair will trigger it.

Both conditions are applied to the SAME widened `tie_score_epsilon` used by
the (rejected) v2 experiment's sweep range, so this is testing a genuinely
different hypothesis, not a different threshold value for the same
mechanism.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from pipeline import candidate_generation, refinement
from pipeline.candidate_generation import Candidate
from pipeline import ranking

GT_TOLERANCE_PX = 5.0
AMBIGUITY_MARGIN = 0.02  # same definition as ACCURACY_FORENSICS.md / center_tiebreak_v2/harness.py


def _dist(x1, y1, x2, y2) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def candidate_from_dict(d: dict) -> Candidate:
    return Candidate(x=d["x"], y=d["y"], score=d["score"], scale=d["scale"],
                      rotation_deg=d["rotation_deg"], template_size=d["template_size"])


def apply_multiway_tiebreak(ranked: list[Candidate], search_shape: tuple[int, int], *,
                             tie_score_epsilon: float, min_group_size: int = 3,
                             max_spread_px: float = 200.0) -> tuple[list[Candidate], bool]:
    """Like ranking.apply_center_tiebreak, but gated on group size AND
    spatial spread, not score alone. Returns (ranked_or_reordered, fired).

    Deliberately NOT a modification of pipeline/ranking.py -
    reports/V2_ARCHITECTURE_PLAN.md section 10: an experiment never edits
    pipeline/ in place. This is new experiment-local code that consumes
    the same `ranked` list `ranking.rank_classical` produces.
    """
    if len(ranked) < min_group_size or ranked[0].score <= 0:
        return ranked, False
    best_score = ranked[0].score
    tie_len = 1
    for c in ranked[1:]:
        if best_score - c.score < tie_score_epsilon:
            tie_len += 1
        else:
            break
    if tie_len < min_group_size:
        return ranked, False

    tied_group = ranked[:tie_len]
    max_pairwise = 0.0
    for i in range(len(tied_group)):
        for j in range(i + 1, len(tied_group)):
            d = _dist(tied_group[i].x, tied_group[i].y, tied_group[j].x, tied_group[j].y)
            max_pairwise = max(max_pairwise, d)
    if max_pairwise > max_spread_px:
        return ranked, False

    height, width = search_shape
    center_x, center_y = width / 2.0, height / 2.0
    reordered = sorted(tied_group, key=lambda c: (c.x - center_x) ** 2 + (c.y - center_y) ** 2)
    return reordered + ranked[tie_len:], True


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


def evaluate_cached_pair(cached: dict, *, tie_score_epsilon: float, min_group_size: int,
                          max_spread_px: float, image_root: str) -> dict:
    """Applies one tie-break config to one cached ranked-candidate pool,
    re-running only pipeline.refinement.refine() (cheap - one extra
    correlation for the winning hypothesis) since the winner may have
    changed. Everything upstream (candidate_generation, dedup, classical
    ranking) is reused verbatim from the cache built by build_cache.py.
    """
    import os
    ranked = [candidate_from_dict(d) for d in cached["ranked_top"]]
    search_shape = (cached["search_height"], cached["search_width"])
    pre_tiebreak_winner = ranked[0]

    reordered, fired = apply_multiway_tiebreak(
        ranked, search_shape, tie_score_epsilon=tie_score_epsilon,
        min_group_size=min_group_size, max_spread_px=max_spread_px,
    )
    winner = reordered[0]
    tiebreak_changed_winner = winner is not pre_tiebreak_winner

    ref = cv2.imread(os.path.join(image_root, cached["reference_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32)
    search = cv2.imread(os.path.join(image_root, cached["search_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32)
    coarse_error_px = _dist(winner.x, winner.y, cached["gt_x"], cached["gt_y"])
    refined_x, refined_y = refinement.refine(ref, search, winner)
    final_error_px = _dist(refined_x, refined_y, cached["gt_x"], cached["gt_y"])

    gt_x, gt_y = cached["gt_x"], cached["gt_y"]
    dists = sorted(((_dist(c.x, c.y, gt_x, gt_y), i) for i, c in enumerate(ranked)), key=lambda t: t[0])
    nearest_dist, nearest_idx = dists[0]
    gt_in_pool = nearest_dist <= GT_TOLERANCE_PX
    gt_candidate = ranked[nearest_idx] if gt_in_pool else None

    gt_candidate_score = None
    if gt_candidate is not None:
        for c in reordered:
            if c is gt_candidate:
                gt_candidate_score = float(c.score)
                break

    winner_is_gt_candidate = gt_in_pool and (gt_candidate is winner)
    failure_location = _classify_failure(
        final_error_px=final_error_px, gt_in_pool=gt_in_pool,
        winner_is_gt_candidate=winner_is_gt_candidate, coarse_error_px=coarse_error_px,
        winner_score=float(winner.score), gt_candidate_score=gt_candidate_score,
    )

    return {
        "pair_id": cached["pair_id"], "split": cached["split"],
        "structural_family": cached["structural_family"],
        "gt_x": gt_x, "gt_y": gt_y,
        "pred_x": refined_x, "pred_y": refined_y, "error_px": final_error_px,
        "confidence": float(winner.score), "runtime_s": 0.0,  # sweep timing not meaningful per-pair; see run_experiment.py for real timing
        "tiebreak_fired": fired, "tiebreak_changed_winner": tiebreak_changed_winner,
        "failure_location": failure_location, "gt_in_pool": gt_in_pool,
    }


def instrumented_localize_full(reference_img: np.ndarray, search_img: np.ndarray, gt_x: float, gt_y: float, *,
                                tie_score_epsilon: float, min_group_size: int, max_spread_px: float) -> dict:
    """Full from-scratch pipeline run (candidate_generation -> dedup ->
    rank_classical -> tie-break -> refine), timed end-to-end with
    time.perf_counter() - used for the final gate-quality confirmation run
    (honest runtime_s for gate criterion 6), exactly mirroring
    center_tiebreak_v2/harness.py::instrumented_localize. The cheaper
    cache-based evaluate_cached_pair() above is only for the exploratory
    sweep (candidate_generation reused from a precomputed cache), never for
    numbers that go into a gate decision.
    """
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)

    t0 = time.perf_counter()
    raw_candidates = candidate_generation.build_candidate_pool(reference, search)
    candidates = candidate_generation.deduplicate_by_location(raw_candidates)
    ranked = ranking.rank_classical(candidates)
    pre_tiebreak_winner = ranked[0]

    reordered, fired = apply_multiway_tiebreak(
        ranked, search.shape, tie_score_epsilon=tie_score_epsilon,
        min_group_size=min_group_size, max_spread_px=max_spread_px,
    )
    winner = reordered[0]
    tiebreak_changed_winner = winner is not pre_tiebreak_winner
    coarse_error_px = _dist(winner.x, winner.y, gt_x, gt_y)
    refined_x, refined_y = refinement.refine(reference, search, winner)
    runtime_s = time.perf_counter() - t0
    final_error_px = _dist(refined_x, refined_y, gt_x, gt_y)

    dists = sorted(((_dist(c.x, c.y, gt_x, gt_y), i) for i, c in enumerate(candidates)), key=lambda t: t[0])
    nearest_dist, nearest_idx = dists[0]
    gt_in_pool = nearest_dist <= GT_TOLERANCE_PX
    gt_candidate = candidates[nearest_idx] if gt_in_pool else None

    gt_candidate_score = None
    if gt_candidate is not None:
        for c in reordered:
            if c is gt_candidate:
                gt_candidate_score = float(c.score)
                break

    winner_is_gt_candidate = gt_in_pool and (gt_candidate is winner)
    failure_location = _classify_failure(
        final_error_px=final_error_px, gt_in_pool=gt_in_pool,
        winner_is_gt_candidate=winner_is_gt_candidate, coarse_error_px=coarse_error_px,
        winner_score=float(winner.score), gt_candidate_score=gt_candidate_score,
    )

    return {
        "pred_x": refined_x, "pred_y": refined_y, "error_px": final_error_px,
        "coarse_error_px": coarse_error_px, "runtime_s": runtime_s,
        "confidence": float(winner.score),
        "tiebreak_fired": fired, "tiebreak_changed_winner": tiebreak_changed_winner,
        "failure_location": failure_location, "gt_in_pool": gt_in_pool,
    }
