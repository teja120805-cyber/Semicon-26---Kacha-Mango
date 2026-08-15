"""Classical template-matching primitives: normalized cross-correlation
evaluated over an explicit grid of scale and rotation hypotheses.

Testing only a single 0-degree rotation hypothesis over the whole image is a
likely cause of a sharp accuracy collapse whenever rotation drift is present.
V2's candidate generation instead always enumerates a small grid of rotation
AND scale hypotheses, so "assume 0 degrees" can never be silently baked into
the design (see reports/V2_ARCHITECTURE_PLAN.md section 2).
"""
from __future__ import annotations

import cv2
import numpy as np


def build_template(reference: np.ndarray, scale_factor: float, rotation_deg: float) -> np.ndarray:
    """Resize the Reference down by `scale_factor` (a hypothesis for the
    true Reference:Search pixel-size ratio, nominally 10) then rotate by
    `rotation_deg`, producing one candidate template to correlate against
    the Search image."""
    h, w = reference.shape
    size = max(8, int(round(w / scale_factor)))
    template = cv2.resize(reference.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    if rotation_deg != 0.0:
        matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), rotation_deg, 1.0)
        template = cv2.warpAffine(template, matrix, (size, size), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return template.astype(np.float32)


def correlate(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalized cross-correlation score map. OpenCV's
    TM_CCOEFF_NORMED is exactly ZNCC (zero-mean normalized cross-
    correlation): invariant to per-image affine intensity shifts, which is
    why no contrast-normalization preprocessing is needed upstream."""
    return cv2.matchTemplate(search.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)


def top_k_peaks(score_map: np.ndarray, k: int, suppression_radius: int) -> list[tuple[int, int, float]]:
    """Greedy non-maximum suppression: repeatedly take the global max, then
    suppress a `suppression_radius` neighborhood around it, up to k times.
    Returns (top_left_x, top_left_y, score) tuples in score-map coordinates
    (the caller adds half the template size to get a center point)."""
    working = score_map.copy()
    peaks: list[tuple[int, int, float]] = []
    h, w = working.shape
    for _ in range(k):
        idx = int(np.argmax(working))
        y, x = divmod(idx, w)
        score = float(working[y, x])
        if not np.isfinite(score):
            break
        peaks.append((x, y, score))
        y0, y1 = max(0, y - suppression_radius), min(h, y + suppression_radius + 1)
        x0, x1 = max(0, x - suppression_radius), min(w, x + suppression_radius + 1)
        working[y0:y1, x0:x1] = -np.inf
    return peaks
