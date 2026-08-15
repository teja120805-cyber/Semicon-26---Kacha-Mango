"""Coarse-to-fine rotation/scale refinement.

`experiments/finer_hypothesis_grid/` showed that a globally denser
scale/rotation grid (81 vs 25 hypotheses) rescues grid-misalignment
failures (net +5 pairs) but at ~3.2x runtime, since every extra hypothesis
re-correlates the FULL search image. This experiment gets the same benefit
much more cheaply: run the normal (unmodified-cost) coarse 25-hypothesis
search first, then refine only the top-N candidate LOCATIONS with a local
finer rotation/scale search restricted to a small window around each
candidate - local-window correlation is far cheaper than a full-image one,
so refining even a generous number of candidates costs a fraction of a
globally denser grid.

Never reads ground truth; mirrors pipeline/candidate_generation.py and
pipeline/matching.py's own functions (imported, not duplicated in logic)
so scoring stays identical at the coarse stage.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline import candidate_generation, matching

LOCAL_WINDOW_MARGIN_PX = 20
LOCAL_ROTATION_OFFSETS = (-1.25, 0.0, 1.25)  # half of the coarse 2.5deg step
LOCAL_SCALE_OFFSETS = (-0.2, 0.0, 0.2)  # half of the coarse 0.4 absolute-scale step


@dataclass
class RefinedCandidate:
    x: float
    y: float
    score: float
    scale: float
    rotation_deg: float
    template_size: int
    coarse_score: float


def _local_window(search: np.ndarray, cx: float, cy: float, half: int) -> tuple[np.ndarray, int, int]:
    h, w = search.shape
    x0 = int(np.clip(round(cx - half), 0, w))
    x1 = int(np.clip(round(cx + half), 0, w))
    y0 = int(np.clip(round(cy - half), 0, h))
    y1 = int(np.clip(round(cy + half), 0, h))
    return search[y0:y1, x0:x1], x0, y0


def refine_candidate_locally(reference: np.ndarray, search: np.ndarray,
                              candidate: candidate_generation.Candidate) -> RefinedCandidate:
    """Search a small local neighborhood of (rotation, scale) around
    `candidate`'s own coarse hypothesis, restricted to a small crop of
    `search` around its location - cheap because both the template and the
    correlated region stay small, unlike a globally denser grid."""
    half = candidate.template_size // 2 + LOCAL_WINDOW_MARGIN_PX
    window, wx0, wy0 = _local_window(search, candidate.x, candidate.y, half)

    best_score = candidate.score
    best_scale = candidate.scale
    best_rotation = candidate.rotation_deg
    best_x, best_y = candidate.x, candidate.y
    best_template_size = candidate.template_size

    if window.shape[0] < candidate.template_size + 1 or window.shape[1] < candidate.template_size + 1:
        return RefinedCandidate(candidate.x, candidate.y, candidate.score, candidate.scale,
                                 candidate.rotation_deg, candidate.template_size, candidate.score)

    for d_rot in LOCAL_ROTATION_OFFSETS:
        for d_scale in LOCAL_SCALE_OFFSETS:
            if d_rot == 0.0 and d_scale == 0.0:
                continue  # already have the coarse score for this exact hypothesis
            rotation = candidate.rotation_deg + d_rot
            scale = candidate.scale + d_scale
            template = matching.build_template(reference, scale, rotation)
            if template.shape[0] >= window.shape[0] or template.shape[1] >= window.shape[1]:
                continue
            score_map = matching.correlate(window, template)
            idx = int(np.argmax(score_map))
            py, px = divmod(idx, score_map.shape[1])
            score = float(score_map[py, px])
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_scale, best_rotation = scale, rotation
                best_x = wx0 + px + template.shape[1] / 2.0
                best_y = wy0 + py + template.shape[0] / 2.0
                best_template_size = template.shape[0]

    return RefinedCandidate(best_x, best_y, best_score, best_scale, best_rotation,
                             best_template_size, candidate.score)


def coarse_to_fine_localize(reference: np.ndarray, search: np.ndarray, top_n: int = 8
                             ) -> tuple[RefinedCandidate, list[RefinedCandidate], int]:
    """Full coarse-to-fine pipeline: unmodified 25-hypothesis coarse search
    -> dedup -> locally refine the top `top_n` coarse candidates -> re-rank
    by refined score. Returns (winner, all refined candidates considered,
    number of local correlations performed)."""
    raw = candidate_generation.build_candidate_pool(reference, search)
    coarse_candidates = candidate_generation.deduplicate_by_location(raw)
    coarse_ranked = sorted(coarse_candidates, key=lambda c: c.score, reverse=True)[:top_n]

    refined = [refine_candidate_locally(reference, search, c) for c in coarse_ranked]
    refined.sort(key=lambda r: r.score, reverse=True)
    n_local_correlations = len(coarse_ranked) * (len(LOCAL_ROTATION_OFFSETS) * len(LOCAL_SCALE_OFFSETS) - 1)
    return refined[0], refined, n_local_correlations
