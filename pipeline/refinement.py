"""Subpixel refinement: quadratic (parabolic) interpolation of the winning
candidate's correlation peak, independently along each axis. Recomputes one
extra correlation for the winning hypothesis rather than reusing the pool's
score map, so this module stays independently testable/callable without
threading score maps through every other stage.
"""
from __future__ import annotations

import numpy as np

from . import matching
from .candidate_generation import Candidate


def _parabolic_offset(f_minus: float, f_0: float, f_plus: float) -> float:
    denom = f_minus - 2.0 * f_0 + f_plus
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (f_minus - f_plus) / denom, -1.0, 1.0))


def refine(reference: np.ndarray, search: np.ndarray, candidate: Candidate,
           psf_sigma: float = 0.0) -> tuple[float, float]:
    """Returns a subpixel-refined (x, y) center for `candidate`.

    `psf_sigma` must match the value the winning candidate was SELECTED
    under, so the parabola is fitted to the same correlation surface the
    winner was chosen on rather than a differently-filtered one."""
    template = matching.build_template(reference, candidate.scale, candidate.rotation_deg, psf_sigma)
    score_map = matching.correlate(search, template)
    h, w = score_map.shape

    px = int(np.clip(round(candidate.x - template.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(candidate.y - template.shape[0] / 2.0), 1, h - 2))

    dx = _parabolic_offset(score_map[py, px - 1], score_map[py, px], score_map[py, px + 1])
    dy = _parabolic_offset(score_map[py - 1, px], score_map[py, px], score_map[py + 1, px])

    refined_x = px + dx + template.shape[1] / 2.0
    refined_y = py + dy + template.shape[0] / 2.0
    return float(refined_x), float(refined_y)
