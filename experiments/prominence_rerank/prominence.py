"""Local peak-prominence re-ranking of the top-K classically-ranked
candidates.

Grounded directly in a mechanism finding confirmed independently by THREE
prior experiments (`cross_hypothesis_consensus_rerank`,
`hough_subpatch_voting`, `pyramid_periodicity_search`): when the classical
winner is wrong, it is essentially never because the true location was
unreachable or badly mis-ranked among near-ties - it is because the wrong
candidate's raw ZNCC score is GENUINELY, substantially higher than the
true location's (e.g. `pyramid_periodicity_search/REPORT.md`: a candidate
0.4px from ground truth scored 0.187 while the classical winner 8.5px away
scored 0.775 - a 4x gap, nowhere near any tie-break epsilon). Raw score
alone conflates two different things: "this location matches distinctively
well" and "this location matches about as well as everywhere else nearby,
including it" - the second is exactly what a periodic lattice produces
everywhere along its repeat direction.

This experiment re-scores the top-K classical candidates by PROMINENCE:
peak score minus the local background score in an annulus around the peak,
sampled from the SAME winning hypothesis's own correlation surface (not
across hypotheses like `cross_hypothesis_consensus_rerank`, and not within
one candidate's own template footprint like `hough_subpatch_voting` - this
looks outward, at OTHER (x, y) locations near the candidate, under the
SAME scale/rotation). A truly distinctive match should stand out from its
immediate spatial neighborhood; a periodic decoy should not, because its
neighbors (the other repeats) score nearly as well.
"""
from __future__ import annotations

import numpy as np

from pipeline import matching
from pipeline.candidate_generation import Candidate
from pipeline.ranking import rank_classical


def compute_prominence(reference: np.ndarray, search: np.ndarray, candidate: Candidate, *,
                        inner_exclude_px: float = 2.0, annulus_outer_px: float = 12.0) -> tuple[float, float]:
    """Returns (prominence, background_score). Recomputes the candidate's
    own winning hypothesis's full correlation surface (same scale/rotation
    that already produced this candidate during generation) and measures
    the MAX score in a near annulus around the peak (excluding a small
    core - the peak's own immediate blob).

    MAX, not median/mean, and a tight default radius (~1-2 periodic pitches
    - measured directly at ~5-7.5px on this dataset, see module docstring):
    a direct radial-profile probe on a real periodic pair showed a
    near-competitive sibling peak (score 0.743 vs. the winner's 0.775 - a
    gap of only 0.03) sitting at radius 4-6px, surrounded on all sides by
    a much lower median/mean (a sibling peak is a small, sparse local
    maximum, not a uniformly elevated region) - a median or mean statistic
    over the same annulus washed this out to near-zero and would have
    barely differentiated any candidate from any other. MAX correctly
    surfaces "is there a competitive periodic repeat immediately nearby",
    which mean/median could not.
    """
    template = matching.build_template(reference, candidate.scale, candidate.rotation_deg)
    score_map = matching.correlate(search, template)
    th, tw = template.shape

    peak_px = candidate.x - tw / 2.0
    peak_py = candidate.y - th / 2.0

    h, w = score_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - peak_px, yy - peak_py)
    annulus_mask = (dist > inner_exclude_px) & (dist <= annulus_outer_px)

    finite = np.isfinite(score_map)
    mask = annulus_mask & finite
    if np.any(mask):
        background = float(np.max(score_map[mask]))
    else:
        background = float(np.nanmax(score_map[finite])) if np.any(finite) else 0.0

    prominence = float(candidate.score) - background
    return prominence, background


def rank_prominence(candidates: list[Candidate], reference: np.ndarray, search: np.ndarray, *,
                     top_k: int = 8, gamma: float = 1.0,
                     inner_exclude_px: float = 2.0, annulus_outer_px: float = 12.0) -> list[Candidate]:
    """Re-ranks the top `top_k` classically-ranked candidates by
    `score + gamma * prominence`; candidates beyond top_k are appended
    unchanged, in their original classical order. gamma=0.0 is
    mathematically identical to rank_classical (built-in sanity check)."""
    ranked = rank_classical(candidates)
    if gamma == 0.0 or len(ranked) < 2:
        return ranked

    head = ranked[:top_k]
    tail = ranked[top_k:]

    scored = []
    for c in head:
        prominence, _bg = compute_prominence(reference, search, c,
                                              inner_exclude_px=inner_exclude_px,
                                              annulus_outer_px=annulus_outer_px)
        scored.append((c.score + gamma * prominence, c))
    scored.sort(key=lambda t: t[0], reverse=True)

    return [c for _, c in scored] + tail
