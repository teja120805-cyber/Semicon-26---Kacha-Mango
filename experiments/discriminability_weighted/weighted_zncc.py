"""Weighted ZNCC primitives (P3 from reports/RESEARCH_SURVEY_SCORING.md).

Standard ZNCC treats every template pixel as equally informative. On a DRAM
crop that is provably wrong: ~90% of the template is a periodic cell array
that is identical at the true location AND at every lattice-shifted decoy,
so it contributes to sigma (the denominator) without contributing anything
to the SEPARATION between them. The ~10% aperiodic boundary content carries
all the discriminating information and is diluted accordingly - measured at
a 0.0098 template-side separation against a 0.732 Search-side difference
(experiments/crop_uniqueness_ceiling/REPORT.md section 3).

With per-pixel weights w, sum(w) = 1:

    mu_w(I)  = sum(w * I)
    sig_w(I) = sqrt(sum(w * (I - mu_w(I))^2))
    ZNCC_w   = sum(w * (T - mu_w(T)) * (I - mu_w(I))) / (sig_w(T) * sig_w(I))

w = uniform recovers standard ZNCC exactly, which is this module's null
control. That identity is VERIFIED numerically against OpenCV's
TM_CCOEFF_NORMED in verify_null.py rather than asserted from the algebra.

Dense evaluation stays FFT-fast. Expanding the numerator with sum(w) = 1:

    sum(w*(T-muT)*(I-muI)) = sum((w*T)*I) - muT * sum(w*I)

and the search-side variance is sum(w*I^2) - (sum(w*I))^2, so a dense map
needs exactly three cross-correlations of the Search image - with (w*T),
with w, and (of I^2) with w - instead of the one standard ZNCC needs.
Three, not four as the survey estimated, because sum(w*I) is reused.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-12


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Force sum(w) = 1, which every formula in this module assumes."""
    w = np.asarray(w, dtype=np.float32)
    total = float(w.sum())
    if total <= 0.0:
        return np.full(w.shape, 1.0 / w.size, dtype=np.float32)
    return (w / total).astype(np.float32)


def uniform_weights(shape: tuple[int, int]) -> np.ndarray:
    n = int(shape[0]) * int(shape[1])
    return np.full(shape, 1.0 / n, dtype=np.float32)


def weighted_zncc_point(template: np.ndarray, patch: np.ndarray, w: np.ndarray) -> float:
    """ZNCC_w between a template and one equally-sized Search patch.

    Used when only a handful of locations need scoring (the top-K
    re-scoring path), where a dense map would be pure waste.
    """
    t = template.astype(np.float64)
    p = patch.astype(np.float64)
    wd = w.astype(np.float64)

    mu_t = float((wd * t).sum())
    mu_p = float((wd * p).sum())
    dt = t - mu_t
    dp = p - mu_p
    sig_t = float(np.sqrt(max((wd * dt * dt).sum(), EPS)))
    sig_p = float(np.sqrt(max((wd * dp * dp).sum(), EPS)))
    return float((wd * dt * dp).sum() / (sig_t * sig_p))


def weighted_zncc_map(search: np.ndarray, template: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Dense ZNCC_w score map, same shape/convention as
    pipeline.matching.correlate (valid-mode, top-left indexed)."""
    search = search.astype(np.float32)
    template = template.astype(np.float32)
    w = normalize_weights(w)

    wt = (w * template).astype(np.float32)
    mu_t = float((w * template).sum())
    sig_t = float(np.sqrt(max(float((w * (template - mu_t) ** 2).sum()), EPS)))

    # sum(w*T*I), sum(w*I), sum(w*I^2) over every valid window position.
    a = cv2.matchTemplate(search, wt, cv2.TM_CCORR)
    b = cv2.matchTemplate(search, w, cv2.TM_CCORR)
    c = cv2.matchTemplate((search * search).astype(np.float32), w, cv2.TM_CCORR)

    numerator = a - mu_t * b
    var_i = np.maximum(c - b * b, EPS)
    return (numerator / (sig_t * np.sqrt(var_i))).astype(np.float32)


def extract_patch_at(search: np.ndarray, top_left_x: int, top_left_y: int,
                     shape: tuple[int, int]) -> np.ndarray | None:
    """Search patch with the same footprint a template at this top-left
    would cover. Returns None when the window falls outside the image -
    callers must treat that as "cannot score", never as a zero score."""
    h, w = int(shape[0]), int(shape[1])
    if top_left_y < 0 or top_left_x < 0:
        return None
    if top_left_y + h > search.shape[0] or top_left_x + w > search.shape[1]:
        return None
    return search[top_left_y:top_left_y + h, top_left_x:top_left_x + w]
