"""Coarse-to-fine (image-pyramid) candidate proposal, targeting
`candidate_generation`-stage periodicity failures via spatial low-pass
filtering rather than an alternative scoring function
(`experiments/periodicity/`, both variants already rejected) or a
re-ranking pass (`cross_hypothesis_consensus_rerank/`,
`hough_subpatch_voting/`, both already rejected).

Grounded measurement (not assumed): direct autocorrelation of a
`dev_dense_periodic` reference template showed a word/bit periodic pitch of
roughly 5-7.5px in the SAME pixel space `pipeline/matching.correlate`
operates in (verified via `cv2.matchTemplate` self-correlation peak
spacing). That pitch is small relative to macro structural features like
mat boundaries (which span tens to hundreds of pixels). The hypothesis:
area-averaging (`cv2.INTER_AREA`) downsampling by a factor large enough to
push that ~5-7.5px pitch below/near the coarse image's Nyquist limit should
average away the periodic repeat while a boundary-scale feature is still
clearly resolved at the coarse resolution - giving the coarse pass a
correlation surface that isn't fooled by the SAME periodic-decoy ambiguity
the full-resolution 99-hypothesis grid suffers from.

Mechanism: downsample both Reference and Search by the SAME factor (so the
scale/rotation hypothesis grid stays physically valid - the two images
still land in the same relative-scale relationship the grid assumes), run
the unmodified `candidate_generation.build_candidate_pool` at the coarse
resolution to get a handful of coarse proposal locations, then run a full
99-hypothesis correlation AGAIN at full resolution but confined to a small
local window around each coarse proposal (cheap - windows are ~100-150px,
not the full ~1000px search image). The resulting full-resolution,
locally-refined candidates are handed back to the caller to be MERGED into
the standard full-image candidate pool (never replacing it) - so this can
only ever ADD a competing candidate, never remove one the standard grid
already found.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import candidate_generation, matching
from pipeline.candidate_generation import Candidate

COARSE_PEAKS_PER_HYPOTHESIS = 2
COARSE_SUPPRESSION_RADIUS_PX = 3


def _downsample(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape
    new_w, new_h = max(8, int(round(w / factor))), max(8, int(round(h / factor)))
    return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_coarse_to_fine_candidates(reference: np.ndarray, search: np.ndarray, *,
                                     downsample_factor: float,
                                     top_k_coarse: int,
                                     window_margin_px: int,
                                     scale_hypotheses: tuple[float, ...],
                                     rotation_hypotheses: tuple[float, ...]) -> list[Candidate]:
    ref_coarse = _downsample(reference, downsample_factor)
    search_coarse = _downsample(search, downsample_factor)

    coarse_candidates = candidate_generation.build_candidate_pool(
        ref_coarse, search_coarse,
        scale_hypotheses=scale_hypotheses, rotation_hypotheses=rotation_hypotheses,
        peaks_per_hypothesis=COARSE_PEAKS_PER_HYPOTHESIS,
        suppression_radius_px=COARSE_SUPPRESSION_RADIUS_PX,
    )
    if not coarse_candidates:
        return []

    coarse_dedup_radius = max(3.0, 10.0 / downsample_factor)
    coarse_deduped = candidate_generation.deduplicate_by_location(coarse_candidates, radius_px=coarse_dedup_radius)
    coarse_ranked = sorted(coarse_deduped, key=lambda c: c.score, reverse=True)[:top_k_coarse]

    sh, sw = search.shape
    local_candidates: list[Candidate] = []
    for cc in coarse_ranked:
        fx, fy = cc.x * downsample_factor, cc.y * downsample_factor
        for scale in scale_hypotheses:
            for rotation in rotation_hypotheses:
                template = matching.build_template(reference, scale, rotation)
                th, tw = template.shape
                x0 = int(max(0, fx - tw / 2.0 - window_margin_px))
                x1 = int(min(sw, fx + tw / 2.0 + window_margin_px))
                y0 = int(max(0, fy - th / 2.0 - window_margin_px))
                y1 = int(min(sh, fy + th / 2.0 + window_margin_px))
                if x1 - x0 < tw or y1 - y0 < th:
                    continue
                window = search[y0:y1, x0:x1]
                score_map = matching.correlate(window, template)
                if score_map.size == 0:
                    continue
                idx = int(np.argmax(score_map))
                wy, wx = divmod(idx, score_map.shape[1])
                score = float(score_map[wy, wx])
                if not np.isfinite(score):
                    continue
                cand_x = x0 + wx + tw / 2.0
                cand_y = y0 + wy + th / 2.0
                local_candidates.append(Candidate(x=cand_x, y=cand_y, score=score, scale=scale,
                                                   rotation_deg=rotation, template_size=tw))
    return local_candidates
