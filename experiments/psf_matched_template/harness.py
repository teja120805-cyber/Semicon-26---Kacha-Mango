"""Standalone localize, structurally identical to pipeline.localize.localize
except the template used for candidate generation and subpixel refinement is
PSF-matched to the Search image's passband.

Ranking (rank_classical), dedup, center tiebreak, ambiguity, and the result
contract are the UNMODIFIED production functions, called directly.
"""
from __future__ import annotations

import time

from pipeline import candidate_generation, feature_extraction, ranking
from pipeline.localize import AMBIGUITY_THRESHOLD, LocalizationResult

from psf_match import build_candidate_pool_psf, refine_psf


def localize_psf(reference_img, search_img, *, sigma_extra: float,
                 scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                 rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                 top_k_report: int = 5) -> LocalizationResult:
    t0 = time.perf_counter()
    reference = reference_img.astype("float32")
    search = search_img.astype("float32")

    raw = build_candidate_pool_psf(reference, search, sigma_extra=sigma_extra,
                                   scale_hypotheses=scale_hypotheses,
                                   rotation_hypotheses=rotation_hypotheses)
    if not raw:
        raise RuntimeError("Candidate generation produced no candidates")
    candidates = candidate_generation.deduplicate_by_location(raw)

    ranked = ranking.rank_classical(candidates)
    ranked = ranking.apply_center_tiebreak(ranked, search.shape)

    winner = ranked[0]
    refined_x, refined_y = refine_psf(reference, search, winner, sigma_extra)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    return LocalizationResult(
        x=refined_x, y=refined_y, confidence=float(winner.score),
        ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb_ratio,
        runtime_s=time.perf_counter() - t0, ranking_mode="psf_matched_template",
        num_candidates=len(candidates),
        top_candidates=[{"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                         "rotation_deg": c.rotation_deg} for c in ranked[:top_k_report]],
    )
