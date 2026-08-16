"""PSF-matched template construction.

Mechanism (measured in experiments/crop_uniqueness_ceiling, §3):

The Reference and Search images travel different optical/resampling paths
before they are ever compared.

  Reference: blurred at REFERENCE resolution with sigma `blur_sigma_ref_px`
             (default 0.6 px at 1000px scale), then the pipeline shrinks it
             10x to build the template -> the blur it carries into template
             space is 0.6/10 = 0.06 Search px.
  Search:    blurred at FINE canvas resolution with sigma
             `blur_search_effective_px * 10`, then area-averaged 10x ->
             an effective 1.0 Search px of blur.

So the production template is roughly 16x sharper than the image it is
correlated against. ZNCC then compares a crisp template to a soft image,
and the mismatch costs fidelity everywhere: template-vs-Search@truth peaks
at only ~0.85. Since decoys also sit at ~0.85, the true location has no
margin - even though the underlying Search content at the true location and
at the decoy differ at ZNCC 0.73.

The fix is one line of physics: convolve the template with the Gaussian
that closes the gap,

    sigma_extra = sqrt(sigma_search^2 - (sigma_ref / scale)^2)  ~= 1.0 px

so both sides of the correlation sit in the same passband.

IMPORTANT - this module never reads generator parameters. `sigma_extra` is
a hyperparameter selected by a sweep on the `development` split only (see
run_experiment.py), exactly like every other tuned constant in this
project. The derivation above explains *why* the swept optimum should land
near 1.0; it is not used to set the value.

Imports pipeline/ unmodified; never writes to it.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching
from pipeline.candidate_generation import (
    DEFAULT_ROTATION_HYPOTHESES,
    DEFAULT_SCALE_HYPOTHESES,
    PEAKS_PER_HYPOTHESIS,
    SUPPRESSION_RADIUS_PX,
    Candidate,
)


def build_template_psf_matched(reference: np.ndarray, scale_factor: float,
                               rotation_deg: float, sigma_extra: float) -> np.ndarray:
    """pipeline.matching.build_template followed by the PSF-matching blur.

    sigma_extra <= 0 reproduces the production template exactly, so the
    sigma=0 arm of any sweep is provably identical to baseline.
    """
    template = matching.build_template(reference, scale_factor, rotation_deg)
    if sigma_extra and sigma_extra > 0:
        template = cv2.GaussianBlur(template, (0, 0), sigma_extra,
                                    borderType=cv2.BORDER_REPLICATE)
    return template.astype(np.float32)


def build_candidate_pool_psf(reference: np.ndarray, search: np.ndarray, *,
                             sigma_extra: float,
                             scale_hypotheses: tuple[float, ...] = DEFAULT_SCALE_HYPOTHESES,
                             rotation_hypotheses: tuple[float, ...] = DEFAULT_ROTATION_HYPOTHESES,
                             peaks_per_hypothesis: int = PEAKS_PER_HYPOTHESIS,
                             suppression_radius_px: int = SUPPRESSION_RADIUS_PX) -> list[Candidate]:
    """Structurally identical to candidate_generation.build_candidate_pool -
    same hypothesis grid, same peak count, same suppression radius, same
    center convention - with the single substitution of the PSF-matched
    template. Nothing else about candidate generation changes."""
    candidates: list[Candidate] = []
    for scale in scale_hypotheses:
        for rotation in rotation_hypotheses:
            template = build_template_psf_matched(reference, scale, rotation, sigma_extra)
            score_map = matching.correlate(search, template)
            for px, py, score in matching.top_k_peaks(score_map, peaks_per_hypothesis,
                                                      suppression_radius_px):
                candidates.append(Candidate(
                    x=px + template.shape[1] / 2.0, y=py + template.shape[0] / 2.0,
                    score=score, scale=scale, rotation_deg=rotation,
                    template_size=template.shape[0]))
    return candidates


def refine_psf(reference: np.ndarray, search: np.ndarray, candidate: Candidate,
               sigma_extra: float) -> tuple[float, float]:
    """pipeline.refinement.refine with the PSF-matched template, so the
    subpixel step interpolates the same score surface the winner was chosen
    on rather than a differently-filtered one."""
    from pipeline.refinement import _parabolic_offset

    template = build_template_psf_matched(reference, candidate.scale,
                                          candidate.rotation_deg, sigma_extra)
    score_map = matching.correlate(search, template)
    h, w = score_map.shape
    px = int(np.clip(round(candidate.x - template.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(candidate.y - template.shape[0] / 2.0), 1, h - 2))
    dx = _parabolic_offset(score_map[py, px - 1], score_map[py, px], score_map[py, px + 1])
    dy = _parabolic_offset(score_map[py - 1, px], score_map[py, px], score_map[py + 1, px])
    return (float(px + dx + template.shape[1] / 2.0),
            float(py + dy + template.shape[0] / 2.0))
