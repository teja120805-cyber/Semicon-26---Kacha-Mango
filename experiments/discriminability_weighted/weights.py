"""Two ways to set the per-pixel discriminability weights for P3.

Both return a weight map the size of the template, normalized to sum 1, and
both are written so that `alpha = 0.0` returns exactly uniform weights -
the built-in null control. `alpha` mixes on the normalized maps:

    w = (1 - alpha) * uniform + alpha * discriminability

so alpha is a genuine continuous knob between production ZNCC (0.0) and
fully discriminability-driven scoring (1.0), never an on/off switch.

Scheme A - lattice-shift dissimilarity (template-only)
    w(x) ~ mean_over_delta |T(x) - T(x + delta)|^2 for lattice vectors
    delta read off the template's own autocorrelation. Array pixels map onto
    themselves under a lattice shift and score ~0; boundary pixels do not and
    score large. Needs no knowledge of the Search image, so it can be applied
    densely during candidate generation.

Scheme B - confuser variance (data-driven, no lattice estimation)
    w(x) ~ Var_k[I_k(x)] across the Search patches at the top-K candidate
    locations. Pixels that look identical at every rival location carry no
    information about which rival is right and get ~0 weight. This is
    Fisher-style discriminant weighting and needs a candidate pool first, so
    it is a re-scoring stage rather than a dense one.

Note on scheme B and the ACCURACY_90_CAMPAIGN rejections: those nine
experiments re-ranked using functions OF the existing ZNCC scores. Scheme B
does not read the scores at all - it recomputes a different similarity from
pixels. That is a different class of change, but the distinction is narrow
enough to be worth stating rather than assuming.
"""
from __future__ import annotations

import cv2
import numpy as np

from .weighted_zncc import normalize_weights, uniform_weights

# Lattice vectors shorter than this are almost certainly autocorrelation
# noise around the origin rather than a real cell pitch.
MIN_LATTICE_PX = 3.0
# ...and longer than this are a sizeable fraction of the template, where
# "shift and compare" leaves too little overlap to mean anything.
MAX_LATTICE_FRACTION = 0.4
MAX_LATTICE_VECTORS = 4


def estimate_lattice_vectors(template: np.ndarray,
                             max_vectors: int = MAX_LATTICE_VECTORS) -> list[tuple[int, int]]:
    """Dominant periodic shift vectors, read off the template's own
    normalized autocorrelation via FFT. Returns (dy, dx) integer offsets,
    strongest first; an empty list means no usable periodicity was found
    (an aperiodic crop), which callers must handle rather than assume."""
    t = template.astype(np.float32)
    t = t - float(t.mean())
    denom = float((t * t).sum())
    if denom <= 0.0:
        return []

    spectrum = np.fft.rfft2(t)
    acf = np.fft.irfft2(spectrum * np.conj(spectrum), s=t.shape)
    acf = np.fft.fftshift(acf) / denom

    h, w = acf.shape
    cy, cx = h // 2, w // 2
    max_r = MAX_LATTICE_FRACTION * min(h, w)

    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - cy, xx - cx)
    searchable = (radius >= MIN_LATTICE_PX) & (radius <= max_r)

    working = np.where(searchable, acf, -np.inf)
    vectors: list[tuple[int, int]] = []
    for _ in range(max_vectors):
        idx = int(np.argmax(working))
        py, px = divmod(idx, w)
        if not np.isfinite(working[py, px]) or working[py, px] <= 0.10:
            break
        dy, dx = py - cy, px - cx
        vectors.append((int(dy), int(dx)))
        # Suppress this peak, its negation, and a small neighbourhood, so
        # the next pick is a genuinely different lattice direction rather
        # than the same one re-found one pixel over.
        for sy, sx in ((py, px), (2 * cy - py, 2 * cx - px)):
            y0, y1 = max(0, sy - 3), min(h, sy + 4)
            x0, x1 = max(0, sx - 3), min(w, sx + 4)
            working[y0:y1, x0:x1] = -np.inf
    return vectors


def _shift_dissimilarity(template: np.ndarray, vectors: list[tuple[int, int]]) -> np.ndarray:
    """mean over delta of |T(x) - T(x+delta)|^2, with edge handling by
    replication so shifted-out regions do not manufacture fake structure."""
    t = template.astype(np.float32)
    acc = np.zeros_like(t)
    used = 0
    for dy, dx in vectors:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(t, matrix, (t.shape[1], t.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        acc += (t - shifted) ** 2
        used += 1
    if used == 0:
        return np.zeros_like(t)
    return acc / float(used)


def lattice_shift_weights(template: np.ndarray, alpha: float = 1.0,
                          smooth_sigma: float = 1.0) -> tuple[np.ndarray, dict]:
    """Scheme A. Returns (weights, diagnostics)."""
    shape = template.shape
    uniform = uniform_weights(shape)
    if alpha <= 0.0:
        return uniform, {"scheme": "lattice_shift", "alpha": 0.0, "n_vectors": 0, "fell_back": False}

    vectors = estimate_lattice_vectors(template)
    if not vectors:
        # Aperiodic crop: there is no lattice to down-weight, so uniform IS
        # the right answer here. Reported, not silently swallowed.
        return uniform, {"scheme": "lattice_shift", "alpha": float(alpha),
                          "n_vectors": 0, "fell_back": True}

    dissim = _shift_dissimilarity(template, vectors)
    if smooth_sigma > 0.0:
        dissim = cv2.GaussianBlur(dissim, (0, 0), smooth_sigma, borderType=cv2.BORDER_REPLICATE)

    discriminability = normalize_weights(dissim)
    mixed = (1.0 - alpha) * uniform + alpha * discriminability
    return normalize_weights(mixed), {"scheme": "lattice_shift", "alpha": float(alpha),
                                       "n_vectors": len(vectors), "fell_back": False,
                                       "vectors": vectors}


def confuser_variance_weights(patches: list[np.ndarray], alpha: float = 1.0,
                              smooth_sigma: float = 1.0) -> tuple[np.ndarray, dict]:
    """Scheme B. `patches` are the Search windows at the rival candidate
    locations, all the same shape as the template."""
    shape = patches[0].shape
    uniform = uniform_weights(shape)
    if alpha <= 0.0 or len(patches) < 2:
        return uniform, {"scheme": "confuser_variance", "alpha": float(alpha),
                          "n_patches": len(patches), "fell_back": len(patches) < 2}

    stack = np.stack([p.astype(np.float32) for p in patches], axis=0)
    # Each rival is z-normalized first, so a rival that is merely brighter or
    # higher-contrast overall does not masquerade as per-pixel disagreement.
    means = stack.mean(axis=(1, 2), keepdims=True)
    stds = stack.std(axis=(1, 2), keepdims=True)
    stack = (stack - means) / np.maximum(stds, 1e-6)

    variance = stack.var(axis=0)
    if smooth_sigma > 0.0:
        variance = cv2.GaussianBlur(variance, (0, 0), smooth_sigma, borderType=cv2.BORDER_REPLICATE)

    discriminability = normalize_weights(variance)
    mixed = (1.0 - alpha) * uniform + alpha * discriminability
    return normalize_weights(mixed), {"scheme": "confuser_variance", "alpha": float(alpha),
                                       "n_patches": len(patches), "fell_back": False}
