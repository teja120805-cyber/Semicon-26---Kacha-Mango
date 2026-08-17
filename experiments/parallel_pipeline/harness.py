"""Localization with a pluggable spectral backend.

Structurally identical to pipeline.localize.localize: same hypothesis grid,
same peaks-per-hypothesis, same suppression radius, same centre convention,
the unmodified rank_classical / apply_center_tiebreak, and the same parabolic
subpixel step. The ONLY substitution is that candidate generation and
refinement correlate against spectrally-filtered images.

The production PSF-matched dual-arm behaviour is deliberately DISABLED here
(single arm, psf_sigma configurable) so that any measured effect is
attributable to the spectral filter alone rather than interacting with the
arm-selection rule. Composing the two is a separate question, taken up only
if the filter proves out on its own.
"""
from __future__ import annotations

import time

import numpy as np

from pipeline import candidate_generation, feature_extraction, matching, ranking
from pipeline.candidate_generation import Candidate
from pipeline.localize import AMBIGUITY_THRESHOLD, LocalizationResult
from pipeline.refinement import _parabolic_offset

from spectral import (apply_filter, build_notch_filter, build_whitening_filter,
                      peak_to_sidelobe_ratio)


def make_filter(search: np.ndarray, backend: str, **kw) -> np.ndarray:
    if backend == "zncc":
        return np.ones(search.shape, dtype=np.float32)
    if backend == "prewhiten":
        return build_whitening_filter(search, rho=kw.get("rho", 0.0),
                                      lam_rel=kw.get("lam_rel", 1e-2))
    if backend == "notch":
        return build_notch_filter(search, depth=kw.get("depth", 0.0),
                                  n_harmonics=kw.get("n_harmonics", 24))
    raise ValueError(f"unknown backend {backend!r}")


def localize_spectral(reference_img, search_img, *, backend: str = "zncc",
                      psf_sigma: float = 0.0,
                      scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                      rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                      top_k_report: int = 5, **backend_kw):
    t0 = time.perf_counter()
    reference = reference_img.astype("float32")
    search = search_img.astype("float32")

    filt = make_filter(search, backend, **backend_kw)
    is_identity = backend == "zncc" or (backend == "prewhiten" and backend_kw.get("rho", 0.0) == 0.0) \
        or (backend == "notch" and backend_kw.get("depth", 0.0) <= 0.0)

    # Filter the Search image once; templates are filtered per hypothesis, with
    # the resampled filter cached per template size (only ~11 distinct sizes
    # across the scale grid, so this is 11 resizes rather than 99).
    search_f = search if is_identity else apply_filter(search, filt)
    tmpl_filter_cache: dict[int, np.ndarray] = {}

    def _template(scale, rot):
        t = matching.build_template(reference, scale, rot, psf_sigma)
        if is_identity:
            return t
        size = t.shape[0]
        if size not in tmpl_filter_cache:
            import cv2
            tmpl_filter_cache[size] = cv2.resize(filt, (size, size),
                                                 interpolation=cv2.INTER_AREA)
        return apply_filter(t, tmpl_filter_cache[size])

    candidates: list[Candidate] = []
    best_map = None
    best_peak = None
    best_score = -np.inf
    for scale in scale_hypotheses:
        for rot in rotation_hypotheses:
            t = _template(scale, rot)
            smap = matching.correlate(search_f, t)
            for px, py, sc in matching.top_k_peaks(
                    smap, candidate_generation.PEAKS_PER_HYPOTHESIS,
                    candidate_generation.SUPPRESSION_RADIUS_PX):
                candidates.append(Candidate(x=px + t.shape[1] / 2.0, y=py + t.shape[0] / 2.0,
                                            score=sc, scale=scale, rotation_deg=rot,
                                            template_size=t.shape[0]))
                if sc > best_score:
                    best_score, best_map, best_peak = sc, smap, (px, py)

    if not candidates:
        raise RuntimeError("Candidate generation produced no candidates")
    pool = candidate_generation.deduplicate_by_location(candidates)
    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(pool), search.shape)
    winner = ranked[0]

    t = _template(winner.scale, winner.rotation_deg)
    smap = matching.correlate(search_f, t)
    h, w = smap.shape
    px = int(np.clip(round(winner.x - t.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(winner.y - t.shape[0] / 2.0), 1, h - 2))
    dx = _parabolic_offset(smap[py, px - 1], smap[py, px], smap[py, px + 1])
    dy = _parabolic_offset(smap[py - 1, px], smap[py, px], smap[py + 1, px])

    psr = peak_to_sidelobe_ratio(best_map, best_peak) if best_map is not None else float("nan")
    amb = feature_extraction.ambiguity_ratio(sorted((c.score for c in pool), reverse=True))

    res = LocalizationResult(
        x=float(px + dx + t.shape[1] / 2.0), y=float(py + dy + t.shape[0] / 2.0),
        confidence=float(winner.score), ambiguous=amb >= AMBIGUITY_THRESHOLD,
        ambiguity_ratio=amb, runtime_s=time.perf_counter() - t0,
        ranking_mode=f"spectral:{backend}", num_candidates=len(pool),
        top_candidates=[{"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                         "rotation_deg": c.rotation_deg} for c in ranked[:top_k_report]])
    return res, psr
