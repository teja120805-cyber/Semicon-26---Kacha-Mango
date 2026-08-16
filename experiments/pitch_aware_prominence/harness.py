"""Standalone localize function, structurally identical to
pipeline.localize.localize except the ranking step is swapped for
pitch_aware.rank_pitch_aware_prominence - candidate generation, dedup,
center tiebreak, and refinement are all the UNMODIFIED production
functions, called directly.
"""
from __future__ import annotations

import time

from pipeline import candidate_generation, feature_extraction, ranking, refinement
from pipeline.localize import AMBIGUITY_THRESHOLD, LocalizationResult

from pitch_aware import rank_pitch_aware_prominence


def localize_pitch_aware(reference_img, search_img, *,
                          scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                          rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                          top_k: int = 8, gamma: float = 1.0,
                          search_tolerance_px: float = 2.0, num_multiples: int = 2,
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

    ranked = rank_pitch_aware_prominence(candidates, reference, search, top_k=top_k, gamma=gamma,
                                          search_tolerance_px=search_tolerance_px, num_multiples=num_multiples)
    ranked = ranking.apply_center_tiebreak(ranked, search.shape)

    winner = ranked[0]
    refined_x, refined_y = refinement.refine(reference, search, winner)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    runtime_s = time.perf_counter() - t0
    return LocalizationResult(
        x=refined_x, y=refined_y, confidence=float(winner.score),
        ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb_ratio,
        runtime_s=runtime_s, ranking_mode="pitch_aware_prominence", num_candidates=len(candidates),
        top_candidates=[
            {"x": c.x, "y": c.y, "score": c.score, "scale": c.scale, "rotation_deg": c.rotation_deg}
            for c in ranked[:top_k_report]
        ],
    )
