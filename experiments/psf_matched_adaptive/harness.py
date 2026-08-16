"""Standalone localize with a PER-PAIR, blindly-estimated PSF-matching blur.

Structurally identical to pipeline.localize.localize except the template
used for candidate generation and subpixel refinement is convolved with a
Gaussian whose width is estimated from the image pair itself
(spectral_sigma.estimate_sigma). Ranking, dedup, center tiebreak,
ambiguity and the result contract are the UNMODIFIED production functions.

Self-contained: the pool builder below mirrors
candidate_generation.build_candidate_pool exactly (same grid, same peak
count, same suppression radius, same center convention) rather than
importing the sibling psf_matched_template experiment.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from pipeline import candidate_generation, feature_extraction, matching, ranking
from pipeline.candidate_generation import Candidate
from pipeline.localize import AMBIGUITY_THRESHOLD, LocalizationResult
from pipeline.refinement import _parabolic_offset

from spectral_sigma import estimate_sigma


def _template(reference, scale, rotation_deg, sigma):
    t = matching.build_template(reference, scale, rotation_deg)
    if sigma and sigma > 0:
        t = cv2.GaussianBlur(t, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    return t.astype(np.float32)


def _pool(reference, search, sigma, scale_hypotheses, rotation_hypotheses):
    out: list[Candidate] = []
    for scale in scale_hypotheses:
        for rotation in rotation_hypotheses:
            t = _template(reference, scale, rotation, sigma)
            smap = matching.correlate(search, t)
            for px, py, score in matching.top_k_peaks(
                    smap, candidate_generation.PEAKS_PER_HYPOTHESIS,
                    candidate_generation.SUPPRESSION_RADIUS_PX):
                out.append(Candidate(x=px + t.shape[1] / 2.0, y=py + t.shape[0] / 2.0,
                                     score=score, scale=scale, rotation_deg=rotation,
                                     template_size=t.shape[0]))
    return out


def localize_adaptive(reference_img, search_img, *,
                      scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                      rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                      sigma_override: float | None = None,
                      top_k_report: int = 5):
    t0 = time.perf_counter()
    reference = reference_img.astype("float32")
    search = search_img.astype("float32")

    sigma = estimate_sigma(reference, search) if sigma_override is None else float(sigma_override)

    raw = _pool(reference, search, sigma, scale_hypotheses, rotation_hypotheses)
    if not raw:
        raise RuntimeError("Candidate generation produced no candidates")
    candidates = candidate_generation.deduplicate_by_location(raw)

    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(candidates), search.shape)
    winner = ranked[0]

    t = _template(reference, winner.scale, winner.rotation_deg, sigma)
    smap = matching.correlate(search, t)
    h, w = smap.shape
    px = int(np.clip(round(winner.x - t.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(winner.y - t.shape[0] / 2.0), 1, h - 2))
    dx = _parabolic_offset(smap[py, px - 1], smap[py, px], smap[py, px + 1])
    dy = _parabolic_offset(smap[py - 1, px], smap[py, px], smap[py + 1, px])
    rx, ry = px + dx + t.shape[1] / 2.0, py + dy + t.shape[0] / 2.0

    amb = feature_extraction.ambiguity_ratio(sorted((c.score for c in candidates), reverse=True))
    res = LocalizationResult(
        x=float(rx), y=float(ry), confidence=float(winner.score),
        ambiguous=amb >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb,
        runtime_s=time.perf_counter() - t0, ranking_mode="psf_matched_adaptive",
        num_candidates=len(candidates),
        top_candidates=[{"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                         "rotation_deg": c.rotation_deg} for c in ranked[:top_k_report]])
    return res, sigma
