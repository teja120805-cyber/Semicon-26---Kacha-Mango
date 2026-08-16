"""Sub-patch geometric-consistency re-ranking (a lightweight, RANSAC/Hough-
style verification pass over the classical ranking's own top-K candidates).

For each of the top-K deduplicated candidates (already ranked by
pipeline.ranking.rank_classical), the reference template rendered at that
candidate's (scale, rotation) hypothesis is split into several overlapping
sub-patches. Each sub-patch is independently, LOCALLY re-matched against a
small window of the search image around where it should fall if the
candidate's whole-template match is genuinely correct. A location whose
sub-patches all independently confirm the same alignment (real structure)
should out-score one whose high whole-template score came from periodic/
textural similarity that doesn't hold up sub-patch by sub-patch.

This only re-ranks candidates the classical search ALREADY proposed - unlike
the project's prior periodicity/ experiment (rejected: alternative
whole-template scoring functions) or the learned_candidate_generator
experiment (rejected: candidate recall too low to help), this targets
candidate_ranking / genuine_ambiguity failures specifically
(reports/ACCURACY_FORENSICS.md's failure taxonomy - GT already in the pool
but outscored, or a near-tie) - it structurally cannot rescue
candidate_generation failures where the truth never entered the pool at all.

Never modifies pipeline/ - imports matching.build_template/correlate and
Candidate unmodified.
"""
from __future__ import annotations

import numpy as np

from pipeline import matching
from pipeline.candidate_generation import Candidate

# 4 corner quadrants + center, each a fraction of the full template -
# overlapping, not a strict partition, so every sub-patch still contains
# enough structure to correlate meaningfully even for small templates.
SUBPATCH_FRACTION = 0.55
SUBPATCH_CENTERS = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5))
LOCAL_SEARCH_MARGIN_PX = 6


def subpatch_consistency(reference: np.ndarray, search: np.ndarray, candidate: Candidate) -> float:
    template = matching.build_template(reference, candidate.scale, candidate.rotation_deg)
    t_h, t_w = template.shape
    sub_size = max(8, int(round(min(t_h, t_w) * SUBPATCH_FRACTION)))

    top_left_x = candidate.x - t_w / 2.0
    top_left_y = candidate.y - t_h / 2.0

    scores = []
    for fx, fy in SUBPATCH_CENTERS:
        sx0 = int(np.clip(round(fx * t_w - sub_size / 2.0), 0, max(0, t_w - sub_size)))
        sy0 = int(np.clip(round(fy * t_h - sub_size / 2.0), 0, max(0, t_h - sub_size)))
        sub_template = template[sy0:sy0 + sub_size, sx0:sx0 + sub_size]
        if sub_template.shape != (sub_size, sub_size):
            continue

        expected_x = int(round(top_left_x + sx0))
        expected_y = int(round(top_left_y + sy0))
        m = LOCAL_SEARCH_MARGIN_PX
        h, w = search.shape
        wy0 = int(np.clip(expected_y - m, 0, max(0, h - sub_size)))
        wx0 = int(np.clip(expected_x - m, 0, max(0, w - sub_size)))
        wy1 = int(np.clip(expected_y + sub_size + m, sub_size, h))
        wx1 = int(np.clip(expected_x + sub_size + m, sub_size, w))
        window = search[wy0:wy1, wx0:wx1]
        if window.shape[0] < sub_size or window.shape[1] < sub_size:
            continue

        local_map = matching.correlate(window, sub_template)
        if local_map.size == 0:
            continue
        scores.append(float(np.max(local_map)))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def rank_subpatch(ranked_candidates: list[Candidate], reference: np.ndarray, search: np.ndarray, *,
                   top_k: int, beta: float) -> list[Candidate]:
    """Re-scores the top_k already-(classically)ranked candidates using
    sub-patch consistency; candidates below top_k are left untouched and
    stay below the re-scored head (a re-ranking pass, not a full
    re-ordering of the whole pool)."""
    if not ranked_candidates:
        return ranked_candidates
    head = ranked_candidates[:top_k]
    tail = ranked_candidates[top_k:]

    consistency = [subpatch_consistency(reference, search, c) for c in head]
    combined = [c.score + beta * cons for c, cons in zip(head, consistency)]
    order = sorted(range(len(head)), key=lambda i: combined[i], reverse=True)
    new_head = [head[i] for i in order]
    return new_head + tail
