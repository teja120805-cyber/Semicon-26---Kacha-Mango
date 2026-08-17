"""P3 harness: production pipeline, with discriminability-weighted ZNCC
consulted ONLY among near-ties.

Imports pipeline/ unmodified - candidate generation, the dual-arm PSF
selection, ranking, the centre tie-break and refinement are all production
code called with production defaults. The single addition is stage 2 below.

Why bounded to near-ties rather than dense:

  1. experiments/oracle_ceiling_diagnostic/ established that EVERY remaining
     failure is a near-tie - the true location scores within 0.05 ZNCC of the
     chosen one, worst ratio 1.048x. So a tie-bounded rule gives up no
     measurable coverage, while a dense one cannot disturb a clear winner.
  2. experiments/parallel_pipeline/ (P1) found its dense forms broke as many
     pairs as they rescued, and its bounded Form C was the only one that
     stayed interpretable. Starting bounded is the lesson from that work.

Null controls, both exact by construction and both verified per pair in
verify_null.py rather than assumed:

  * tie_eps = 0.0  -> the near-tie group is the winner alone, so stage 2
    cannot change anything, whatever the weights say.
  * alpha = 0.0    -> weights are uniform, at which ZNCC_w IS standard ZNCC
    algebraically, so stage 2 is skipped outright. This is a genuine
    short-circuit, not a dodge: recomputing it would return the same values
    to within float32 accumulation noise (measured max 1.1e-5, see
    verify_null.py check A), and deliberately NOT re-deriving production's
    own numbers through a second code path is what keeps the null exact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline import candidate_generation, feature_extraction, matching, ranking, refinement
from pipeline.localize import (AMBIGUITY_THRESHOLD, PSF_GAP_DISTINCT_PX,
                                PSF_MATCH_SIGMA, _decisiveness)

from .weighted_zncc import extract_patch_at, weighted_zncc_point
from .weights import confuser_variance_weights, lattice_shift_weights


@dataclass
class WeightedResult:
    x: float
    y: float
    confidence: float
    ambiguous: bool
    ambiguity_ratio: float
    runtime_s: float
    num_candidates: int
    psf_sigma: float = 0.0
    # Stage-2 diagnostics, so a change of winner is always visible per pair
    # rather than inferred from the error column moving.
    tie_group_size: int = 1
    rescored: bool = False
    winner_changed: bool = False
    weight_diag: dict = field(default_factory=dict)
    top_candidates: list = field(default_factory=list)


def _build_pool(reference, search, scale_hypotheses, rotation_hypotheses, psf_selection):
    """Byte-for-byte the pool-building half of pipeline.localize.localize."""
    psf_sigmas = (0.0, PSF_MATCH_SIGMA) if psf_selection else (0.0,)
    best = None
    for sigma in psf_sigmas:
        raw = candidate_generation.build_candidate_pool(
            reference, search, scale_hypotheses=scale_hypotheses,
            rotation_hypotheses=rotation_hypotheses, psf_sigma=sigma,
        )
        if not raw:
            raise RuntimeError("Candidate generation produced no candidates")
        pool = candidate_generation.deduplicate_by_location(raw)
        gap = _decisiveness(pool)
        if best is None or gap > best[1]:
            best = (sigma, gap, pool)
    return best


def _tie_group(ranked: list, tie_eps: float) -> list:
    """Candidates within `tie_eps` raw ZNCC of the top one. tie_eps = 0.0
    yields the winner alone (a strict `<=` on a zero gap would still admit
    exact duplicates, but deduplicate_by_location has already removed
    same-location entries, so any survivor is a genuinely distinct site)."""
    if tie_eps <= 0.0 or len(ranked) < 2:
        return ranked[:1]
    top = ranked[0].score
    group = [c for c in ranked if top - c.score <= tie_eps]
    return group if len(group) >= 2 else ranked[:1]


def _weights_for(candidate, template, group, group_patches, scheme, alpha, smooth_sigma):
    if scheme == "lattice_shift":
        return lattice_shift_weights(template, alpha=alpha, smooth_sigma=smooth_sigma)
    if scheme == "confuser_variance":
        # Every rival resampled into THIS candidate's template footprint, so
        # candidates at different scale hypotheses are compared symmetrically
        # rather than through whichever one happened to rank first.
        target = (template.shape[1], template.shape[0])
        resized = [cv2.resize(p, target, interpolation=cv2.INTER_AREA) for p in group_patches]
        return confuser_variance_weights(resized, alpha=alpha, smooth_sigma=smooth_sigma)
    raise ValueError(f"Unknown scheme '{scheme}'")


def localize_weighted(reference_img: np.ndarray, search_img: np.ndarray, *,
                      alpha: float = 0.0, scheme: str = "lattice_shift",
                      tie_eps: float = 0.0, smooth_sigma: float = 1.0,
                      max_group: int = 8, psf_selection: bool = True,
                      scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
                      rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES,
                      top_k_report: int = 5) -> WeightedResult:
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)
    t0 = time.perf_counter()
    psf_sigma, _, candidates = _build_pool(
        reference, search, scale_hypotheses, rotation_hypotheses, psf_selection)
    pool_time = time.perf_counter() - t0
    return finish_from_pool(reference, search, psf_sigma, candidates,
                            alpha=alpha, scheme=scheme, tie_eps=tie_eps,
                            smooth_sigma=smooth_sigma, max_group=max_group,
                            top_k_report=top_k_report, pool_time_s=pool_time)


def finish_from_pool(reference: np.ndarray, search: np.ndarray, psf_sigma: float,
                     candidates: list, *, alpha: float = 0.0,
                     scheme: str = "lattice_shift", tie_eps: float = 0.0,
                     smooth_sigma: float = 1.0, max_group: int = 8,
                     top_k_report: int = 5, pool_time_s: float = 0.0) -> WeightedResult:
    """Everything downstream of candidate generation, given an already-built
    pool. Split out so a parameter sweep can build each pair's pool ONCE
    (it is production code and cannot vary with P3's parameters) and reuse
    it across every configuration, instead of paying ~5s per pair per config
    to recompute an identical result. localize_weighted is the single-shot
    entry point and calls straight through to here, so the two can never
    drift apart."""
    t0 = time.perf_counter()
    ranked = ranking.rank_classical(candidates)
    original_winner = ranked[0]

    tie_group_size = 1
    rescored = False
    weight_diag: dict = {}

    if alpha > 0.0 and tie_eps > 0.0:
        group = _tie_group(ranked, tie_eps)[:max_group]
        tie_group_size = len(group)
        if len(group) >= 2:
            templates, patches = [], []
            for c in group:
                tmpl = matching.build_template(reference, c.scale, c.rotation_deg, psf_sigma)
                tlx = int(round(c.x - tmpl.shape[1] / 2.0))
                tly = int(round(c.y - tmpl.shape[0] / 2.0))
                patch = extract_patch_at(search, tlx, tly, tmpl.shape)
                templates.append(tmpl)
                patches.append(patch)

            # A candidate whose window falls off the image cannot be scored;
            # it keeps its production score and stays in the running rather
            # than being silently dropped or zeroed.
            usable = [i for i, p in enumerate(patches) if p is not None]
            if len(usable) >= 2:
                valid_patches = [patches[i] for i in usable]
                scores = {}
                for i in usable:
                    w, diag = _weights_for(group[i], templates[i], group, valid_patches,
                                           scheme, alpha, smooth_sigma)
                    scores[i] = weighted_zncc_point(templates[i], patches[i], w)
                    if i == 0:
                        weight_diag = diag
                best_i = max(usable, key=lambda i: scores[i])
                rescored = True
                weight_diag["weighted_scores"] = {group[i].scale: round(scores[i], 6) for i in usable}
                if best_i != 0:
                    ranked = [group[best_i]] + [c for c in ranked if c is not group[best_i]]

    winner = ranked[0]
    ranked = ranking.apply_center_tiebreak(ranked, search.shape)
    winner = ranked[0]

    refined_x, refined_y = refinement.refine(reference, search, winner, psf_sigma)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    return WeightedResult(
        x=refined_x, y=refined_y, confidence=float(winner.score),
        ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD, ambiguity_ratio=amb_ratio,
        runtime_s=pool_time_s + (time.perf_counter() - t0), num_candidates=len(candidates),
        psf_sigma=float(psf_sigma), tie_group_size=tie_group_size, rescored=rescored,
        winner_changed=(winner is not original_winner), weight_diag=weight_diag,
        top_candidates=[{"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                          "rotation_deg": c.rotation_deg} for c in ranked[:top_k_report]],
    )
