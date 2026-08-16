"""Joint scale/rotation/position subpixel refinement.

pipeline/refinement.py already does parabolic subpixel interpolation
independently along x and y at the winning hypothesis's own correlation
peak. The SCALE and ROTATION axes stay coarse (a fixed grid,
pipeline/candidate_generation.py::DEFAULT_SCALE_HYPOTHESES /
DEFAULT_ROTATION_HYPOTHESES) - reports/ACCURACY_FORENSICS.md Finding 2
shows a real, measured "sawtooth" accuracy pattern that tracks distance to
the nearest tested scale/rotation hypothesis, not drift magnitude. That
finding predates the grid being densified twice since
(experiments/finer_hypothesis_grid/, experiments/scale_range_v1/), so the
absolute gaps are smaller now, but the mechanism (quantization to a finite
grid) is structural, not eliminated by density alone.

This experiment extends the SAME parabolic-interpolation idea
refinement.py already uses for x/y to the scale and rotation axes too: at
the winning candidate's (x, y) location, sample the correlation score under
the winning hypothesis's immediate scale/rotation neighbors in the grid,
parabolically interpolate a continuous scale and rotation estimate,
re-render one template at that continuous estimate, and finish with the
same x/y parabolic refinement pipeline/refinement.py already does.

Never modifies pipeline/ - imports matching.build_template/correlate and
candidate_generation's hypothesis tuples/Candidate unmodified. Costs at
most 5 extra correlation calls per pair (2 scale neighbors + 2 rotation
neighbors + 1 final), negligible next to the ~99 hypothesis correlations
candidate generation itself already performs.
"""
from __future__ import annotations

import numpy as np

from pipeline import matching
from pipeline.candidate_generation import (
    DEFAULT_ROTATION_HYPOTHESES,
    DEFAULT_SCALE_HYPOTHESES,
    Candidate,
)


def _parabolic_offset(f_minus: float, f_0: float, f_plus: float) -> float:
    denom = f_minus - 2.0 * f_0 + f_plus
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (f_minus - f_plus) / denom, -1.0, 1.0))


def _score_at_location(search: np.ndarray, template: np.ndarray, x: float, y: float) -> float:
    score_map = matching.correlate(search, template)
    h, w = score_map.shape
    px = int(np.clip(round(x - template.shape[1] / 2.0), 0, w - 1))
    py = int(np.clip(round(y - template.shape[0] / 2.0), 0, h - 1))
    return float(score_map[py, px])


def _nearest_index(hypotheses: tuple, value: float) -> int:
    arr = np.array(hypotheses)
    return int(np.argmin(np.abs(arr - value)))


def refine_joint(reference: np.ndarray, search: np.ndarray, candidate: Candidate, *,
                  scale_hypotheses: tuple = DEFAULT_SCALE_HYPOTHESES,
                  rotation_hypotheses: tuple = DEFAULT_ROTATION_HYPOTHESES) -> tuple[float, float, float, float]:
    """Returns (refined_x, refined_y, refined_scale, refined_rotation_deg)."""
    scale_idx = _nearest_index(scale_hypotheses, candidate.scale)
    rot_idx = _nearest_index(rotation_hypotheses, candidate.rotation_deg)

    if 0 < scale_idx < len(scale_hypotheses) - 1:
        s_minus, s_0, s_plus = (scale_hypotheses[scale_idx - 1], scale_hypotheses[scale_idx],
                                 scale_hypotheses[scale_idx + 1])
        f_minus = _score_at_location(search, matching.build_template(reference, s_minus, candidate.rotation_deg),
                                      candidate.x, candidate.y)
        f_0 = _score_at_location(search, matching.build_template(reference, s_0, candidate.rotation_deg),
                                  candidate.x, candidate.y)
        f_plus = _score_at_location(search, matching.build_template(reference, s_plus, candidate.rotation_deg),
                                     candidate.x, candidate.y)
        offset = _parabolic_offset(f_minus, f_0, f_plus)
        scale_step = s_plus - s_0
        refined_scale = s_0 + offset * scale_step
    else:
        refined_scale = candidate.scale

    if 0 < rot_idx < len(rotation_hypotheses) - 1:
        r_minus, r_0, r_plus = (rotation_hypotheses[rot_idx - 1], rotation_hypotheses[rot_idx],
                                 rotation_hypotheses[rot_idx + 1])
        f_minus = _score_at_location(search, matching.build_template(reference, refined_scale, r_minus),
                                      candidate.x, candidate.y)
        f_0 = _score_at_location(search, matching.build_template(reference, refined_scale, r_0),
                                  candidate.x, candidate.y)
        f_plus = _score_at_location(search, matching.build_template(reference, refined_scale, r_plus),
                                     candidate.x, candidate.y)
        offset = _parabolic_offset(f_minus, f_0, f_plus)
        rot_step = r_plus - r_0
        refined_rotation = r_0 + offset * rot_step
    else:
        refined_rotation = candidate.rotation_deg

    final_template = matching.build_template(reference, refined_scale, refined_rotation)
    score_map = matching.correlate(search, final_template)
    h, w = score_map.shape
    px = int(np.clip(round(candidate.x - final_template.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(candidate.y - final_template.shape[0] / 2.0), 1, h - 2))
    dx = _parabolic_offset(score_map[py, px - 1], score_map[py, px], score_map[py, px + 1])
    dy = _parabolic_offset(score_map[py - 1, px], score_map[py, px], score_map[py + 1, px])
    refined_x = px + dx + final_template.shape[1] / 2.0
    refined_y = py + dy + final_template.shape[0] / 2.0
    return float(refined_x), float(refined_y), float(refined_scale), float(refined_rotation)
