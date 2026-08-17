"""Near-tie arbitration by a spectrally-whitened score.

WHY THIS SHAPE, after the dense variants failed.

The first attempt used the whitened/notched image as a REPLACEMENT for the
raw image throughout candidate generation. Measured on 10 failing and 8
correct pairs, that is net-harmful and wildly unstable: it produces genuine
rescues (30.77px -> 0.36px) but equally large breaks, including on pairs that
were already correct (0.42px -> 274.8px). The reason is now clear - the
periodic array carries most of the image ENERGY, so removing it leaves the
aperiodic boundary signal and the sensor noise at comparable amplitude, and a
correlation over that residual is noise-dominated.

So the filtered representation is informative but too noisy to lead. This
module uses it only where it can help and cannot hurt:

  - Candidate generation, ranking and refinement stay EXACTLY as production.
  - Only when the top candidates are within `tie_eps` of each other in raw
    ZNCC is the whitened score consulted, to choose among them.

That bound is what makes it safe, and our own measurements say it costs
nothing in coverage: every failing pair we have measured is a near-tie, with
the true location scoring within 0.05 ZNCC of the chosen one and never worse
than a 1.048x ratio (experiments/oracle_ceiling_diagnostic/REPORT.md section
2). A clear winner is left untouched by construction, so pairs that currently
work cannot be destabilised.

This is NOT the re-ranking that the nine rejected campaign experiments tried.
Those re-ranked *the same ZNCC scores* by auxiliary geometric criteria. This
arbitrates by a genuinely different scoring representation, which is what the
diagnosis in experiments/crop_uniqueness_ceiling/REPORT.md called for.

tie_eps = 0 disables arbitration entirely and is bit-identical to production.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from pipeline import candidate_generation, feature_extraction, matching, ranking
from pipeline.localize import (AMBIGUITY_THRESHOLD, LocalizationResult,
                               PSF_MATCH_SIGMA, _decisiveness)
from pipeline.refinement import _parabolic_offset

from spectral import apply_filter, build_whitening_filter


def _pool(reference, search, sigma, scales, rots):
    out = []
    for scale in scales:
        for rot in rots:
            t = matching.build_template(reference, scale, rot, sigma)
            smap = matching.correlate(search, t)
            for px, py, sc in matching.top_k_peaks(
                    smap, candidate_generation.PEAKS_PER_HYPOTHESIS,
                    candidate_generation.SUPPRESSION_RADIUS_PX):
                out.append(candidate_generation.Candidate(
                    x=px + t.shape[1] / 2.0, y=py + t.shape[0] / 2.0, score=sc,
                    scale=scale, rotation_deg=rot, template_size=t.shape[0]))
    return out


def localize_arbitrated(reference_img, search_img, *, tie_eps: float = 0.0,
                        rho: float = 0.3, max_tied: int = 6,
                        psf_selection: bool = True,
                        scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                        rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                        top_k_report: int = 5):
    t0 = time.perf_counter()
    reference = reference_img.astype("float32")
    search = search_img.astype("float32")

    # --- production candidate generation, unchanged (incl. dual-arm PSF) ---
    sigmas = (0.0, PSF_MATCH_SIGMA) if psf_selection else (0.0,)
    best = None
    for sigma in sigmas:
        pool = candidate_generation.deduplicate_by_location(
            _pool(reference, search, sigma, scale_hypotheses, rotation_hypotheses))
        gap = _decisiveness(pool)
        if best is None or gap > best[1]:
            best = (sigma, gap, pool)
    psf_sigma, gap, candidates = best

    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(candidates), search.shape)
    winner = ranked[0]
    arbitrated = False
    n_tied = 0

    # --- arbitration, only among genuine near-ties ---
    if tie_eps > 0.0:
        top = ranked[0].score
        tied = [c for c in ranked[:max_tied] if (top - c.score) <= tie_eps]
        n_tied = len(tied)
        if len(tied) > 1:
            filt = build_whitening_filter(search, rho=rho)
            search_w = apply_filter(search, filt)
            cache: dict[int, np.ndarray] = {}
            best_w, best_c = -np.inf, winner
            for c in tied:
                t = matching.build_template(reference, c.scale, c.rotation_deg, psf_sigma)
                size = t.shape[0]
                if size not in cache:
                    cache[size] = cv2.resize(filt, (size, size), interpolation=cv2.INTER_AREA)
                tw = apply_filter(t, cache[size])
                smap = matching.correlate(search_w, tw)
                # score the whitened surface AT this candidate's location only
                px = int(np.clip(round(c.x - size / 2.0), 0, smap.shape[1] - 1))
                py = int(np.clip(round(c.y - size / 2.0), 0, smap.shape[0] - 1))
                sw = float(smap[py, px])
                if sw > best_w:
                    best_w, best_c = sw, c
            arbitrated = best_c is not winner
            winner = best_c

    t = matching.build_template(reference, winner.scale, winner.rotation_deg, psf_sigma)
    smap = matching.correlate(search, t)
    h, w = smap.shape
    px = int(np.clip(round(winner.x - t.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(winner.y - t.shape[0] / 2.0), 1, h - 2))
    dx = _parabolic_offset(smap[py, px - 1], smap[py, px], smap[py, px + 1])
    dy = _parabolic_offset(smap[py - 1, px], smap[py, px], smap[py + 1, px])

    amb = feature_extraction.ambiguity_ratio(sorted((c.score for c in candidates), reverse=True))
    res = LocalizationResult(
        x=float(px + dx + t.shape[1] / 2.0), y=float(py + dy + t.shape[0] / 2.0),
        confidence=float(winner.score), ambiguous=amb >= AMBIGUITY_THRESHOLD,
        ambiguity_ratio=amb, runtime_s=time.perf_counter() - t0,
        ranking_mode="spectral_arbitrated", num_candidates=len(candidates),
        top_candidates=[{"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                         "rotation_deg": c.rotation_deg} for c in ranked[:top_k_report]],
        psf_sigma=float(psf_sigma), psf_decisiveness=float(gap))
    return res, {"arbitrated": arbitrated, "n_tied": n_tied}
