"""Candidate generation: enumerate scale x rotation hypotheses and collect
the strongest template-match peaks under each, producing the pool every
downstream stage (feature extraction, ranking, refinement) operates on.

Never reads ground truth - candidates are produced purely from the
Reference/Search pixel content.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import matching

# Absolute Reference:Search pixel-size ratio hypotheses. The generator's
# true base ratio is exactly 10 (see generator/dataset_generator.py); this
# grid spans +-10% around it - the literal 9:1-11:1 "robustness tests may
# span" range the Applied Materials help doc and sponsor pptx both state -
# to cover the residual scale-drift range ([0.90, 1.10]) that some
# structural families sample, with a small margin.
# Density doubled (9 vs. 5 steps, same span) after experiments/finer_hypothesis_grid/
# and experiments/finer_grid_validation/ showed this corrects hypothesis-grid
# misalignment failures (net rescue +16/+6 on two independent targeted
# validation sets, +4 on the frozen benchmark, 0-2 breaks throughout) without
# affecting any case that doesn't involve rotation/scale drift. Span widened
# from +-8% to the literal +-10% (2026-08-15, experiments/scale_range_v1/,
# same step size - 11 points instead of 9): exact tie on the frozen
# benchmark (widening the grid alone, on data that never needed the extra
# reach, is a genuine no-op - every split/family identical), and on a fresh
# dataset whose scale-drift families were also widened to the literal edge,
# pooled 70.5% -> 71.4%@5px with zero regressions anywhere
# (experiments/scale_range_v1/REPORT.md; integrated as a documented gate
# exception - see reports/GATE_EXCEPTIONS.md).
DEFAULT_SCALE_HYPOTHESES: tuple[float, ...] = (
    9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
    10.2, 10.4, 10.6, 10.8, 11.0,
)

# Degrees. Spans the +-4deg residual rotation-drift range some structural
# families sample, with margin - never just [0.0] (see module docstring).
# Density doubled - see DEFAULT_SCALE_HYPOTHESES comment above.
DEFAULT_ROTATION_HYPOTHESES: tuple[float, ...] = (
    -5.0, -3.75, -2.5, -1.25, 0.0,
    1.25, 2.5, 3.75, 5.0,
)

PEAKS_PER_HYPOTHESIS = 2
SUPPRESSION_RADIUS_PX = 8


@dataclass
class Candidate:
    x: float             # match center, Search-image pixel coordinates
    y: float
    score: float          # ZNCC score at this peak
    scale: float           # absolute Reference:Search ratio tested
    rotation_deg: float
    template_size: int


def build_candidate_pool(reference: np.ndarray, search: np.ndarray, *,
                          scale_hypotheses: tuple[float, ...] = DEFAULT_SCALE_HYPOTHESES,
                          rotation_hypotheses: tuple[float, ...] = DEFAULT_ROTATION_HYPOTHESES,
                          peaks_per_hypothesis: int = PEAKS_PER_HYPOTHESIS,
                          suppression_radius_px: int = SUPPRESSION_RADIUS_PX,
                          psf_sigma: float = 0.0) -> list[Candidate]:
    """`psf_sigma` is forwarded to matching.build_template - see its
    docstring, and pipeline/localize.py's PSF_MATCH_SIGMA, for why the
    template may be blurred into the Search image's passband. 0.0 (the
    default) is the pre-2026-08-16 behaviour, bit-identical."""
    candidates: list[Candidate] = []
    for scale in scale_hypotheses:
        for rotation in rotation_hypotheses:
            template = matching.build_template(reference, scale, rotation, psf_sigma)
            score_map = matching.correlate(search, template)
            for px, py, score in matching.top_k_peaks(score_map, peaks_per_hypothesis, suppression_radius_px):
                cx = px + template.shape[1] / 2.0
                cy = py + template.shape[0] / 2.0
                candidates.append(Candidate(x=cx, y=cy, score=score, scale=scale,
                                             rotation_deg=rotation, template_size=template.shape[0]))
    return candidates


def deduplicate_by_location(candidates: list[Candidate], radius_px: float = 10.0) -> list[Candidate]:
    """Collapse near-duplicate candidates that different scale/rotation
    hypotheses independently found at essentially the same (x, y) into one
    (keeping the highest-scoring instance).

    This matters beyond tidiness: a genuinely correct match is often found
    by several nearby scale/rotation hypotheses with similar scores, since
    the true structure is somewhat tolerant of a slightly-wrong scale/
    rotation guess. Without deduplication, those redundant detections of
    the SAME correct location would be mistaken for a second, DIFFERENT
    plausible location when computing the ambiguity ratio - inverting the
    signal (robust agreement across hypotheses would look like ambiguity
    instead of confidence). Ambiguity should mean "two different places
    look similarly good", not "one place was found five times".
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    radius_sq = radius_px ** 2
    for c in ordered:
        if all((c.x - k.x) ** 2 + (c.y - k.y) ** 2 > radius_sq for k in kept):
            kept.append(c)
    return kept
