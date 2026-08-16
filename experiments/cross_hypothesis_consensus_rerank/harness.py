"""Standalone localize function, structurally identical to
pipeline.localize.localize except the ranking step is swapped for
rank_consensus - every other stage (candidate generation, dedup, center
tiebreak, subpixel refinement, ambiguity reporting) is the UNMODIFIED
production function, called directly, exactly the same way
pipeline.localize.localize itself calls them. This mirrors how
pipeline.ranking.rank_with_model is already wired in as an alternative to
rank_classical - a new ranking function is a first-class swap point in this
architecture, not a fork of the whole pipeline.
"""
from __future__ import annotations

import time

from pipeline import candidate_generation, feature_extraction, ranking, refinement
from pipeline.localize import AMBIGUITY_THRESHOLD, LocalizationResult

from rerank import rank_consensus


def localize_consensus(reference_img, search_img, *, alpha: float, radius_px: float = 15.0,
                        scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                        rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                        top_k_report: int = 5) -> LocalizationResult:
    t0 = time.perf_counter()
    reference = reference_img.astype("float32")
    search = search_img.astype("float32")

    raw_candidates = candidate_generation.build_candidate_pool(
        reference, search, scale_hypotheses=scale_hypotheses, rotation_hypotheses=rotation_hypotheses,
    )
    if not raw_candidates:
        raise RuntimeError("Candidate generation produced no candidates")
    candidates = candidate_generation.deduplicate_by_location(raw_candidates)

    ranked = rank_consensus(candidates, raw_candidates, alpha=alpha, radius_px=radius_px)
    ranked = ranking.apply_center_tiebreak(ranked, search.shape)

    winner = ranked[0]
    refined_x, refined_y = refinement.refine(reference, search, winner)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    runtime_s = time.perf_counter() - t0
    return LocalizationResult(
        x=refined_x, y=refined_y, confidence=float(winner.score),
        ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb_ratio,
        runtime_s=runtime_s, ranking_mode="consensus", num_candidates=len(candidates),
        top_candidates=[
            {"x": c.x, "y": c.y, "score": c.score, "scale": c.scale, "rotation_deg": c.rotation_deg}
            for c in ranked[:top_k_report]
        ],
    )
