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

AMBIGUITY_THRESHOLD = 0.92  # second-best/best ZNCC ratio at or above this => flagged ambiguous


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
             top_k_report: int = 5) -> LocalizationResult:
    """Run the full pipeline on one (Reference, Search) pair and return the
    predicted location, confidence, and ambiguity status. `ranking_mode`
    defaults to "classical" (the production pipeline); pass
    ranking_mode="learned" with a trained `model` to evaluate the candidate
    re-ranker - this never happens implicitly.
    """
    t0 = time.perf_counter()

    reference = _preprocess(reference_img)
    search = _preprocess(search_img)

    raw_candidates = candidate_generation.build_candidate_pool(
        reference, search, scale_hypotheses=scale_hypotheses, rotation_hypotheses=rotation_hypotheses,
    )
    if not raw_candidates:
        raise RuntimeError("Candidate generation produced no candidates")
    # Collapse redundant same-location detections from neighboring
    # scale/rotation hypotheses before ranking/ambiguity - see
    # candidate_generation.deduplicate_by_location docstring.
    candidates = candidate_generation.deduplicate_by_location(raw_candidates)

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
    refined_x, refined_y = refinement.refine(reference, search, winner)

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
    )
