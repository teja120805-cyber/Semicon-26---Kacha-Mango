"""Periodicity-aware candidate generation/scoring variants.

Forensics (reports/ACCURACY_FORENSICS.md) diagnosed candidate-generation
failures on periodic mats as a genuine scoring problem, not a pruning
problem: the true location's ZNCC score, computed on raw intensity, is
sometimes not the highest in the correlation landscape (confirmed by
experiments/wider_candidate_pool - retaining more candidates never changes
the arg-max winner). This module tests whether a DIFFERENT correlation
representation changes which location scores highest, without touching
`pipeline/` - it duplicates `pipeline/matching.py`'s exact structure
(same template construction, same top-k/NMS peak extraction) so the only
variable is the score map's representation.

Two variants:
  - "gradient": correlate Sobel-gradient-magnitude images instead of raw
    intensity - edge/jitter/collapse structure may carry information raw
    intensity ZNCC doesn't exploit.
  - "ensemble": average the intensity and gradient score maps per
    hypothesis, requiring a candidate to look good under both
    representations rather than either alone.

No ground truth is used by any scoring function here - exactly the same
"pipeline never reads GT" discipline as production candidate_generation.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline import matching
from pipeline.candidate_generation import (
    DEFAULT_ROTATION_HYPOTHESES, DEFAULT_SCALE_HYPOTHESES, PEAKS_PER_HYPOTHESIS, SUPPRESSION_RADIUS_PX,
)


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


@dataclass
class Candidate:
    x: float
    y: float
    score: float
    scale: float
    rotation_deg: float
    template_size: int


def _score_map_for_hypothesis(reference: np.ndarray, search: np.ndarray, search_grad: np.ndarray,
                               scale: float, rotation: float, mode: str) -> tuple[np.ndarray, int]:
    template = matching.build_template(reference, scale, rotation)
    if mode == "intensity":
        score_map = matching.correlate(search, template)
    elif mode == "gradient":
        template_grad = sobel_magnitude(template)
        score_map = matching.correlate(search_grad, template_grad)
    elif mode == "ensemble":
        intensity_map = matching.correlate(search, template)
        template_grad = sobel_magnitude(template)
        gradient_map = matching.correlate(search_grad, template_grad)
        score_map = (intensity_map + gradient_map) / 2.0
    else:
        raise ValueError(f"unknown mode '{mode}'")
    return score_map, template.shape[0]


def build_candidate_pool(reference: np.ndarray, search: np.ndarray, mode: str, *,
                          scale_hypotheses: tuple[float, ...] = DEFAULT_SCALE_HYPOTHESES,
                          rotation_hypotheses: tuple[float, ...] = DEFAULT_ROTATION_HYPOTHESES,
                          peaks_per_hypothesis: int = PEAKS_PER_HYPOTHESIS,
                          suppression_radius_px: int = SUPPRESSION_RADIUS_PX) -> list[Candidate]:
    search_grad = sobel_magnitude(search) if mode in ("gradient", "ensemble") else None
    candidates: list[Candidate] = []
    for scale in scale_hypotheses:
        for rotation in rotation_hypotheses:
            score_map, template_size = _score_map_for_hypothesis(
                reference, search, search_grad, scale, rotation, mode
            )
            for px, py, score in matching.top_k_peaks(score_map, peaks_per_hypothesis, suppression_radius_px):
                cx = px + template_size / 2.0
                cy = py + template_size / 2.0
                candidates.append(Candidate(x=cx, y=cy, score=score, scale=scale,
                                             rotation_deg=rotation, template_size=template_size))
    return candidates


def deduplicate_by_location(candidates: list[Candidate], radius_px: float = 10.0) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    radius_sq = radius_px ** 2
    for c in ordered:
        if all((c.x - k.x) ** 2 + (c.y - k.y) ** 2 > radius_sq for k in kept):
            kept.append(c)
    return kept
