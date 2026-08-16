"""Blind, per-pair estimation of the template/Search passband mismatch.

`psf_matched_template` showed that blurring the template to match the
Search image's passband is a real effect (+2.56pp pooled) but that a single
GLOBAL sigma is a compromise: clean-optics families want ~1.6, while
degraded-acquisition families (barrel/charging, vignette/gamma,
speckle/salt-pepper, worst-case) are damaged by it and regress.

This module estimates sigma per pair, from the two images alone.

Derivation. Convolving with a Gaussian of width sigma multiplies the power
spectrum by exp(-4 pi^2 sigma^2 f^2). So if the Search image is the
template's content seen through sigma extra pixels of blur,

    log P_template(f) - log P_search(f) = 4 pi^2 sigma^2 f^2 + c

which is a straight line in f^2. Fitting that slope gives

    sigma = sqrt( slope / (4 pi^2) )

in closed form - no search, no iteration, no tuned constant.

Why this is the right adaptive quantity. No structural family in this
dataset overrides the blur parameters, so the true OPTICAL mismatch really
is ~1.0 px everywhere. What varies between families is noise: dose,
read noise, speckle, salt-and-pepper, charging. Additive noise raises the
Search image's high-frequency power, which flattens the fitted slope and
therefore lowers the estimated sigma. The estimator consequently applies
less blur exactly where the Search image's high frequencies are
noise-dominated rather than signal - which is the behaviour the per-family
regression table in psf_matched_template/REPORT.md asks for. It is best
described as matching *usable bandwidth*, not pure PSF estimation.

DELIBERATELY PARAMETER-FREE. The fit band and clamp below are fixed a
priori from sampling geometry, not swept: `development` contains no
degraded-acquisition family (documented in psf_matched_template/REPORT.md
§3), so any dev-tuned constant here would inherit that blind spot. With
nothing tuned, there is nothing for an unrepresentative split to overfit.

Reads no ground truth and no generator parameters. Imports pipeline/
unmodified; never writes to it.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching

# Fit band in cycles/pixel. Lower bound excludes DC and the macro-structure
# scales where a 100px crop and the full Search image legitimately differ;
# upper bound stays clear of Nyquist (0.5). Fixed from sampling geometry.
F_LO, F_HI = 0.06, 0.30
SIGMA_MIN, SIGMA_MAX = 0.0, 2.0   # non-negative; capped below the ~2.2px template feature scale
NOMINAL_SCALE = 10.0              # generator's base Reference:Search ratio
PATCH_GRID = 5                    # 5x5 = 25 Search patches averaged


def _radial_power(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum of one square patch.

    Hann-windowed and mean-subtracted so patch edges do not inject a
    broadband cross that would masquerade as high-frequency signal.
    """
    n = patch.shape[0]
    w = np.hanning(n)
    win = np.outer(w, w)
    x = (patch - patch.mean()) * win
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2

    k = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(k, k)
    f = np.sqrt(fx ** 2 + fy ** 2).ravel()
    p = p.ravel()

    nbins = n // 2
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(f, edges) - 1, 0, nbins - 1)
    sums = np.bincount(idx, weights=p, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return centers, mean_p


def _search_reference_spectrum(search: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Average radial spectrum over a grid of Search patches, so the
    estimate reflects the image's typical content rather than one
    arbitrary window."""
    h, w = search.shape
    ys = np.linspace(0, h - size, PATCH_GRID).astype(int)
    xs = np.linspace(0, w - size, PATCH_GRID).astype(int)
    acc, centers = None, None
    for y in ys:
        for x in xs:
            c, p = _radial_power(search[y:y + size, x:x + size].astype(np.float32))
            centers = c
            acc = p if acc is None else acc + p
    return centers, acc / (PATCH_GRID ** 2)


def estimate_sigma(reference: np.ndarray, search: np.ndarray) -> float:
    """Blind estimate of the extra Gaussian blur (in Search pixels) that
    brings the template into the Search image's passband."""
    template = matching.build_template(reference, NOMINAL_SCALE, 0.0)
    size = template.shape[0]
    if size > min(search.shape):
        return 0.0

    f_t, p_t = _radial_power(template.astype(np.float32))
    f_s, p_s = _search_reference_spectrum(search, size)

    band = (f_t >= F_LO) & (f_t <= F_HI) & np.isfinite(p_t) & np.isfinite(p_s) & (p_t > 0) & (p_s > 0)
    if band.sum() < 4:
        return 0.0

    f2 = f_t[band] ** 2
    d = np.log(p_t[band]) - np.log(p_s[band])
    slope = float(np.polyfit(f2, d, 1)[0])       # d = slope * f^2 + c
    if not np.isfinite(slope) or slope <= 0:
        return 0.0
    sigma = float(np.sqrt(slope / (4.0 * np.pi ** 2)))
    return float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
