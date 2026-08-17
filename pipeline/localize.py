"""End-to-end V2 localization pipeline:

Reference -> preprocessing -> scale/rotation hypothesis handling ->
candidate generation -> feature extraction -> candidate scoring -> ranking
(classical by default; learned only if explicitly requested) -> subpixel
refinement -> final coordinate -> confidence/ambiguity score.

Never reads ground truth. Used identically by evaluation/evaluate.py
(offline benchmarking) and app/app.py (interactive use) - there is exactly
one localization code path, never a Streamlit-only reimplementation (see
reports/V2_ARCHITECTURE_PLAN.md section 5).

Note on "candidate scoring" vs. "feature extraction" in the pipeline
diagram: candidate scoring IS the ZNCC score already computed during
candidate generation (matching.py); feature_extraction.py's role here is
computing confidence/ambiguity features from the pool's score distribution
and (for the learned path only) producing normalized patches. This is
stated plainly rather than implying a fancier hand-crafted scoring stage
that doesn't exist - see the "don't add sophistication that isn't earning
its keep" principle in reports/V2_ARCHITECTURE_PLAN.md section 7.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import candidate_generation, feature_extraction, ranking, refinement

# Second-best/best ZNCC ratio at or above this => flagged ambiguous.
#
# Recalibrated 0.92 -> 0.990 (2026-08-17, experiments/psr_confidence/).
#
# 0.92 sat far below the statistic's actual operating range. Measured,
# ambiguity_ratio spans 0.816-0.999 with a median of 0.985, so a 0.92 cut
# fired on 85-91% of all pairs at ~32% precision - independently reproducing
# the "128/156 at 31%" figure already recorded in reports/PROJECT_STATUS.md.
# A flag that is on almost always carries no information.
#
# The important correction to that recorded diagnosis: the STATISTIC is not
# the problem. ambiguity_ratio separates correct from wrong pairs at AUC
# 0.933-0.949, comparable to the pool-gap statistic's 0.941-0.964 and far
# better than PSR's 0.577-0.765 (which reports/RESEARCH_SURVEY_SCORING.md
# section P4 proposed as the principled replacement, and which was tested and
# rejected - see experiments/psr_confidence/REPORT.md section 2). Only the
# constant was wrong.
#
# Fitted on development + a freshly generated degraded surface (n=64), then
# evaluated ONCE on a held-back independently-seeded surface (n=40, seed
# 271828) after the selection rule was fixed in code. Flag precision
# 0.324 -> 0.750; the pipeline answers 70.0% of pairs at 92.9% accuracy.
# Cost, stated plainly: failure recall falls 1.000 -> 0.818, so roughly one
# failure in five is no longer flagged. At 0.92 the flag caught every failure
# but fired on nine pairs in ten, which is not a usable trade.
#
# REPORTING ONLY. `ambiguous` is written to output and never read to make a
# decision anywhere in this repository - verified by grep across pipeline/,
# evaluation/, scripts/ and app/, not assumed. Predicted coordinates,
# accuracy@5px, catastrophic rate and runtime are bit-identical either way;
# this was re-verified per pair after the change, not inferred.
#
# Deliberately NOT the tie-break bar: see ranking.TIE_SCORE_EPSILON's comment
# for why that is a numerical-equality test and this is a reporting one. That
# distinction is unaffected by this recalibration.
AMBIGUITY_THRESHOLD = 0.990

# --- PSF-matched dual-arm candidate generation (integrated 2026-08-16) -------
#
# The Reference and Search images travel different optical/resampling paths,
# leaving the template ~16x sharper than the image it is correlated against
# (see matching.build_template). Blurring the template by PSF_MATCH_SIGMA
# closes that gap. It is a large win on clean-optics and periodic-ambiguity
# cases and a real LOSS on acquisitions carrying non-stationary, non-Gaussian
# corruption (barrel warp, spatially-varying vignette/gamma, impulse noise),
# so it must not be applied unconditionally.
#
# Rather than trying to DETECT those acquisitions (a spectral blur estimator
# was tried and provably cannot separate them - the families that gain span
# estimated sigma 0.36-1.06 and those that lose span 0.34-1.03; see
# experiments/psf_matched_adaptive/), the pool is built BOTH ways and the more
# decisive arm wins, judged by the same top-vs-runner-up gap that predicts
# correctness generally (experiments/oracle_ceiling_diagnostic/: correct pairs
# median 0.0188, wrong pairs 0.0026). No threshold, no tuned constant, no
# ground truth, and the harmed families revert to baseline byte-for-byte
# because the unblurred arm simply wins on them.
#
# Evidence: pooled 0.7436 -> 0.7756 on the frozen benchmark and 0.6618 ->
# 0.6985 on an independently-seeded one; 14 rescued / 4 broken across both
# (sign test p = 0.031). Cost is 2x candidate generation. Integrated as a
# documented gate exception (5/7 criteria) - see reports/GATE_EXCEPTIONS.md
# exception 3 and experiments/psf_gated_selection/REPORT.md.
PSF_MATCH_SIGMA = 1.6
PSF_GAP_DISTINCT_PX = 10.0  # same distinctness radius as deduplicate_by_location


def _decisiveness(candidates: list) -> float:
    """Gap between the best candidate and the best candidate at a genuinely
    DIFFERENT location. Larger means the winner is more clearly separated
    from its nearest real rival.

    Returns -inf when no distinct rival exists, so such an arm never wins the
    comparison - matching the evaluated rule exactly.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = ordered[0]
    for c in ordered[1:]:
        if (c.x - top.x) ** 2 + (c.y - top.y) ** 2 > PSF_GAP_DISTINCT_PX ** 2:
            return float(top.score - c.score)
    return float("-inf")


@dataclass
class LocalizationResult:
    x: float
    y: float
    confidence: float
    ambiguous: bool
    ambiguity_ratio: float
    runtime_s: float
    ranking_mode: str
    num_candidates: int
    top_candidates: list = field(default_factory=list)
    # Which PSF hypothesis won the dual-arm comparison for this pair: 0.0 =
    # the historical sharp template, PSF_MATCH_SIGMA = the passband-matched
    # one. Reported so the choice is visible per pair rather than hidden.
    psf_sigma: float = 0.0
    psf_decisiveness: float = 0.0


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Minimal preprocessing: cast to float32 only. ZNCC (used throughout)
    is already invariant to per-image affine intensity shifts, so
    additional contrast/denoising preprocessing was deliberately not added
    - it would risk destroying the fine structural detail the matcher
    depends on, for no measurable benefit."""
    return img.astype(np.float32)


def localize(reference_img: np.ndarray, search_img: np.ndarray, *,
             ranking_mode: str = "classical", model=None, device: str = "cpu",
             scale_hypotheses: tuple[float, ...] = candidate_generation.DEFAULT_SCALE_HYPOTHESES,
             rotation_hypotheses: tuple[float, ...] = candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
             top_k_report: int = 5, psf_selection: bool = True) -> LocalizationResult:
    """Run the full pipeline on one (Reference, Search) pair and return the
    predicted location, confidence, and ambiguity status. `ranking_mode`
    defaults to "classical" (the production pipeline); pass
    ranking_mode="learned" with a trained `model` to evaluate the candidate
    re-ranker - this never happens implicitly.
    """
    t0 = time.perf_counter()

    reference = _preprocess(reference_img)
    search = _preprocess(search_img)

    # Build the pool under each PSF hypothesis and keep the more decisive
    # one. sigma=0.0 is the historical single-arm behaviour and is always
    # evaluated, so this can only differ from it when the PSF-matched arm is
    # strictly more decisive. Ties go to sigma=0.0 (first in the tuple).
    psf_sigmas = (0.0, PSF_MATCH_SIGMA) if psf_selection else (0.0,)
    best = None
    for sigma in psf_sigmas:
        raw_candidates = candidate_generation.build_candidate_pool(
            reference, search, scale_hypotheses=scale_hypotheses,
            rotation_hypotheses=rotation_hypotheses, psf_sigma=sigma,
        )
        if not raw_candidates:
            raise RuntimeError("Candidate generation produced no candidates")
        # Collapse redundant same-location detections from neighboring
        # scale/rotation hypotheses before ranking/ambiguity - see
        # candidate_generation.deduplicate_by_location docstring.
        pool = candidate_generation.deduplicate_by_location(raw_candidates)
        gap = _decisiveness(pool)
        if best is None or gap > best[1]:
            best = (sigma, gap, pool)
    psf_sigma, _, candidates = best

    if ranking_mode == "classical":
        ranked = ranking.rank_classical(candidates)
    elif ranking_mode == "learned":
        if model is None:
            raise ValueError("ranking_mode='learned' requires a model")
        ranked = ranking.rank_with_model(candidates, reference, search, model, device=device)
    else:
        raise ValueError(f"Unknown ranking_mode '{ranking_mode}'")

    # Applied Materials spec: among genuinely tied/equivalent top candidates,
    # prefer the one closest to the Search image's centre. A no-op whenever
    # there is a single clear-best candidate - see ranking.apply_center_tiebreak
    # and ranking.TIE_SCORE_EPSILON for why this is a tight numerical-equality
    # bar, not the (much looser) AMBIGUITY_THRESHOLD used below for reporting.
    ranked = ranking.apply_center_tiebreak(ranked, search.shape)

    winner = ranked[0]
    # Refine on the same correlation surface the winner was selected on.
    refined_x, refined_y = refinement.refine(reference, search, winner, psf_sigma)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    runtime_s = time.perf_counter() - t0
    return LocalizationResult(
        x=refined_x, y=refined_y, confidence=float(winner.score),
        ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb_ratio,
        runtime_s=runtime_s, ranking_mode=ranking_mode, num_candidates=len(candidates),
        top_candidates=[
            {"x": c.x, "y": c.y, "score": c.score, "scale": c.scale, "rotation_deg": c.rotation_deg}
            for c in ranked[:top_k_report]
        ],
        psf_sigma=float(psf_sigma), psf_decisiveness=float(best[1]),
    )
