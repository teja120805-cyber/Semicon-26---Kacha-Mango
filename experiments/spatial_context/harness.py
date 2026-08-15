"""Experiment C: does a wider spatial context window around a candidate
location contain genuinely separating information between the true site
and a periodic decoy?

Method: measure each candidate's own **periodicity strength** directly from
search-image pixel content (no ground truth, no metadata, no injected
fingerprints) via 2D autocorrelation - the same information a real
inference-time algorithm could compute from the search image alone. A
location deep inside a periodic mat autocorrelates strongly with itself at
a nonzero lag (the pitch); a location near a boundary or unique structure
does not, because the boundary/strip breaks the repeating pattern.

If wider context helps distinguish a true site from a decoy, periodicity
strength should measurably diverge between them as the window grows (the
true site, if boundary-adjacent, becomes visibly less "purely periodic";
the decoy, if deep in periodic territory, stays high). If both stay
equally periodic at every tested window size, that is evidence of a real
information ceiling, not an algorithmic failure to exploit context.
"""
from __future__ import annotations

import numpy as np


def _autocorrelation_2d(crop: np.ndarray) -> np.ndarray:
    """Normalized 2D autocorrelation via FFT (Wiener-Khinchin), zero-mean
    input so the zero-lag peak reflects pattern repetition, not DC offset."""
    x = crop.astype(np.float64) - crop.mean()
    f = np.fft.fft2(x, s=(2 * x.shape[0], 2 * x.shape[1]))
    power = f * np.conj(f)
    ac = np.fft.ifft2(power).real
    ac = np.fft.fftshift(ac)
    norm = ac.max()
    if norm <= 1e-9:
        return np.zeros_like(ac)
    return ac / norm


def periodicity_strength(crop: np.ndarray, min_lag_px: int = 3) -> float:
    """Height of the strongest non-zero-lag autocorrelation peak, excluding
    a `min_lag_px` neighborhood around the trivial zero-lag peak. Close to
    1.0 = strongly, repeatably periodic (deep mat interior, no boundary in
    view); close to 0.0 = little self-repetition (boundary/strip/unique
    structure present)."""
    ac = _autocorrelation_2d(crop)
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    ac_masked = ac.copy()
    ac_masked[max(0, cy - min_lag_px):cy + min_lag_px + 1, max(0, cx - min_lag_px):cx + min_lag_px + 1] = -np.inf
    return float(np.max(ac_masked))


def extract_crop(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray | None:
    h, w = img.shape
    half = size / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1]
