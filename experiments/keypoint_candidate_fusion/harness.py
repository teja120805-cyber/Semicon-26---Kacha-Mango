"""Standalone localize function, structurally identical to
pipeline.localize.localize except one additional candidate - proposed by
ORB keypoint matching + RANSAC (keypoint_gen.generate_keypoint_candidate) -
is merged into the raw candidate pool BEFORE deduplication. Candidate
generation's own 99 grid hypotheses, dedup, classical ranking, center
tiebreak, and refinement are all the UNMODIFIED production functions,
called directly.
"""
from __future__ import annotations

import time

from pipeline import candidate_generation, feature_extraction, ranking, refinement
from pipeline.localize import AMBIGUITY_THRESHOLD, LocalizationResult

from keypoint_gen import generate_keypoint_candidate


def localize_keypoint_fusion(reference_img, search_img, *,
                              scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                              rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                              ratio_threshold: float = 0.75,
                              min_inliers: int = 6,
                              top_k_report: int = 5) -> LocalizationResult:
    t0 = time.perf_counter()
    reference = reference_img.astype("float32")
    search = search_img.astype("float32")

    raw_candidates = candidate_generation.build_candidate_pool(
        reference, search, scale_hypotheses=scale_hypotheses, rotation_hypotheses=rotation_hypotheses,
    )
    if not raw_candidates:
        raise RuntimeError("Candidate generation produced no candidates")

    kp_candidate = generate_keypoint_candidate(
        reference, search, ratio_threshold=ratio_threshold, min_inliers=min_inliers,
    )
    fused_raw = list(raw_candidates)
    if kp_candidate is not None:
        fused_raw.append(kp_candidate)

    candidates = candidate_generation.deduplicate_by_location(fused_raw)

    ranked = ranking.rank_classical(candidates)
    ranked = ranking.apply_center_tiebreak(ranked, search.shape)

    winner = ranked[0]
    refined_x, refined_y = refinement.refine(reference, search, winner)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    runtime_s = time.perf_counter() - t0
    return LocalizationResult(
        x=refined_x, y=refined_y, confidence=float(winner.score),
        ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb_ratio,
        runtime_s=runtime_s, ranking_mode="classical+keypoint_fusion", num_candidates=len(candidates),
        top_candidates=[
            {"x": c.x, "y": c.y, "score": c.score, "scale": c.scale, "rotation_deg": c.rotation_deg}
            for c in ranked[:top_k_report]
        ],
    )
