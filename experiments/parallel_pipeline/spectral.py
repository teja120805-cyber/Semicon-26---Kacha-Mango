"""Spectral scoring backends: prewhitening (P1) and lattice notching (P2).

THE ARGUMENT (reports/RESEARCH_SURVEY_SCORING.md section 1)

ZNCC *is* the matched filter, and the matched filter is optimal only when the
competing background is WHITE. Ours is the opposite extreme - a DRAM cell
array puts almost all of its energy into a handful of lattice harmonics. The
textbook correction under coloured interference is to PREWHITEN: divide by
the interference power spectrum before correlating.

Why that targets our measured failure specifically. We measured that ~90% of
the template is periodic array that matches equally well in many places, and
~10% is aperiodic boundary/routing structure carrying all the discriminating
information - diluted by the averaging in ZNCC down to a 0.02-0.05 score
contribution, which is exactly the margin our failures lose by. The array's
energy is concentrated at lattice harmonics; the boundary's is broadband and
low-energy. Dividing by |F|^2rho suppresses the former and relatively
amplifies the latter.

IMPLEMENTATION CHOICE

Rather than replace the correlation, this module whitens the IMAGES and then
runs the unmodified production correlation on them. That keeps local
normalization, peak finding, dedup, ranking and refinement exactly as they
are, so any measured difference is attributable to the representation and
nothing else. rho=0 makes the filter identically 1, which is production
bit-for-bit.

The filter is derived from the SEARCH image (the interference we are trying
to suppress) and applied to both sides, so the correlation stays meaningful.
Template and Search have the same pixel pitch but different sizes, hence
different frequency-grid resolutions; the filter is built once on the Search
grid and resampled onto each template size. Because fftshift places DC at the
centre and both grids span the same normalized range [-0.5, 0.5] cycles/px,
resampling the shifted filter with INTER_AREA is exactly the right sampling -
and the averaging is desirable, since the coarser template bins genuinely
integrate a wider frequency band.

Never modifies pipeline/, generator/, or model/.
"""
from __future__ import annotations

import cv2
import numpy as np

# Light smoothing of the periodogram before inversion. A single image's power
# spectrum is chi-squared with 2 DOF - very noisy - so some smoothing is
# standard. Kept small deliberately: the lattice harmonics we want to suppress
# are strong and span several bins, so light smoothing preserves them while
# damping estimator noise. Larger values would blur away the very structure
# this method exists to exploit.
SPECTRUM_SMOOTH_BINS = 1.0


def _power_spectrum(img: np.ndarray, smooth_bins: float = SPECTRUM_SMOOTH_BINS) -> np.ndarray:
    """Smoothed, fftshifted power spectrum of a zero-meaned image."""
    x = img.astype(np.float32)
    x = x - x.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    if smooth_bins and smooth_bins > 0:
        p = cv2.GaussianBlur(p.astype(np.float32), (0, 0), smooth_bins,
                             borderType=cv2.BORDER_REPLICATE)
    return p.astype(np.float32)


def build_whitening_filter(search: np.ndarray, rho: float, lam_rel: float = 1e-2,
                           smooth_bins: float = SPECTRUM_SMOOTH_BINS) -> np.ndarray:
    """Fftshifted filter W(f) = 1 / (|F_search(f)|^2 + lam)^rho, normalized to
    unit mean so downstream score magnitudes stay comparable across rho.

    `lam_rel` is relative to the spectrum's mean power, so the regularizer
    does not need retuning per image. rho=0 returns exactly ones.
    """
    if rho == 0.0:
        return np.ones(search.shape, dtype=np.float32)
    p = _power_spectrum(search, smooth_bins)
    lam = lam_rel * float(p.mean())
    w = 1.0 / np.power(p + lam, rho, dtype=np.float64)
    w = w / w.mean()
    return w.astype(np.float32)


def build_notch_filter(search: np.ndarray, depth: float, n_harmonics: int = 24,
                       radius_bins: float = 2.0, min_freq: float = 0.02,
                       smooth_bins: float = SPECTRUM_SMOOTH_BINS) -> np.ndarray:
    """P2 - targeted lattice notching.

    Instead of attenuating the whole spectrum by a power law, find the
    strongest discrete harmonics (excluding a low-frequency disc, which
    carries the macro structure we WANT) and suppress only those, by factor
    (1 - depth). depth=0 returns exactly ones.

    This is the surgical counterpart to prewhitening: it deletes the periodic
    carrier outright rather than reshaping the whole spectrum, and A/B'ing the
    two says whether broad reshaping or targeted deletion does the work.
    """
    if depth <= 0.0:
        return np.ones(search.shape, dtype=np.float32)
    p = _power_spectrum(search, smooth_bins)
    h, w_ = p.shape
    cy, cx = h // 2, w_ // 2
    ky = (np.arange(h) - cy) / float(h)
    kx = (np.arange(w_) - cx) / float(w_)
    fy, fx = np.meshgrid(ky, kx, indexing="ij")
    freq = np.sqrt(fx ** 2 + fy ** 2)

    work = p.copy()
    work[freq < min_freq] = -np.inf          # protect DC / macro structure
    filt = np.ones((h, w_), dtype=np.float32)
    rad = int(np.ceil(radius_bins))
    for _ in range(n_harmonics):
        idx = int(np.argmax(work))
        y, x = divmod(idx, w_)
        if not np.isfinite(work[y, x]):
            break
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        x0, x1 = max(0, x - rad), min(w_, x + rad + 1)
        filt[y0:y1, x0:x1] *= (1.0 - depth)
        work[y0:y1, x0:x1] = -np.inf
    return filt


def apply_filter(img: np.ndarray, filt_shifted: np.ndarray) -> np.ndarray:
    """Apply a fftshifted frequency filter to an image, returning a real
    image of the same shape. The filter is resampled to the image's own
    frequency grid when sizes differ (see module docstring)."""
    if filt_shifted.shape != img.shape:
        filt_shifted = cv2.resize(filt_shifted, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32)
    x = x - x.mean()
    f = np.fft.fftshift(np.fft.fft2(x))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(f * filt_shifted)))
    return out.astype(np.float32)


def peak_to_sidelobe_ratio(score_map: np.ndarray, peak_xy: tuple[int, int],
                           exclude_px: int = 11) -> float:
    """P4 - PSR = (peak - mu_sidelobe) / sigma_sidelobe, with a window around
    the peak excluded from the sidelobe statistics (Bolme et al., MOSSE).

    Our own gap statistic is a cruder version of this: it compares the peak
    against one rival, where PSR compares it against the whole distribution of
    rivals. Reported here so it can be evaluated as a replacement for both the
    dual-arm selector and AMBIGUITY_THRESHOLD.
    """
    x, y = peak_xy
    h, w = score_map.shape
    mask = np.ones((h, w), dtype=bool)
    y0, y1 = max(0, y - exclude_px), min(h, y + exclude_px + 1)
    x0, x1 = max(0, x - exclude_px), min(w, x + exclude_px + 1)
    mask[y0:y1, x0:x1] = False
    side = score_map[mask]
    if side.size < 16:
        return float("nan")
    sd = float(side.std())
    if sd < 1e-12:
        return float("nan")
    return float((float(score_map[y, x]) - float(side.mean())) / sd)
