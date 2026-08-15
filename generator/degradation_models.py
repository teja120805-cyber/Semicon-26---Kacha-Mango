"""Imaging/acquisition degradation models.

Every function takes a float32 image (arbitrary intensity range, not yet
clipped/cast) and an explicit `rng` where randomness is involved - never
module-global or system-entropy randomness - so every degradation is
independently seed-controlled and independently toggleable. A disabled
degradation is simply never called; see `image_reference`/`image_search`
below. This means `metadata.py` only ever records parameters for
degradations that were actually applied to a given pair, never a value
that happens to be a no-op left in the schema (the failure mode documented
in reports/DEGRADATION_COVERAGE.md, row 17).

Composition order (see reports/V2_ARCHITECTURE_PLAN.md section 4) follows the
physical acquisition chain: geometric effects (finite beam spot, scan
distortion/drift) happen before the noise floor is added at the detector,
and noise is added before any post-hoc radiometric reshaping
(vignette/gamma). Structural effects (pattern collapse, corner rounding)
are generation-time, not acquisition-time, and live in pattern_renderer.py
instead of here.

Citations for the physically-motivated choices:
- Poisson shot noise as the dominant SEM noise source: Foi, Trimeche,
  Katkovnik & Egiazarian (2008), "Practical Poisson-Gaussian Noise Modeling
  and Fitting for Single-Image Raw-Data".
- Cumulative overlay/placement drift: Orji et al. (2018), "Metrology for
  the next generation of semiconductor devices", Nature Electronics.
"""
from __future__ import annotations

import cv2
import numpy as np


def gaussian_psf_blur(img: np.ndarray, spot_sigma_px: float, astigmatism_ratio: float = 1.0) -> np.ndarray:
    """Finite electron-beam spot size -> Gaussian point-spread function.

    `astigmatism_ratio != 1.0` makes the blur axis-locked-elliptical
    (sharper along one scan axis than the other), a common artifact when
    the beam is imperfectly stigmated. Locked to the scan axes rather than
    an arbitrary angle: there is no physical mechanism on a raster-scanned
    instrument that would rotate an astigmatism axis relative to the scan
    directions (see architecture plan section 3).
    """
    sx = max(float(spot_sigma_px) * astigmatism_ratio, 1e-3)
    sy = max(float(spot_sigma_px) / astigmatism_ratio, 1e-3)
    ksx = max(3, int(2 * round(3 * sx) + 1))
    ksy = max(3, int(2 * round(3 * sy) + 1))
    kx = cv2.getGaussianKernel(ksx, sx).astype(np.float32).flatten()
    ky = cv2.getGaussianKernel(ksy, sy).astype(np.float32).flatten()
    return cv2.sepFilter2D(img.astype(np.float32), -1, kx, ky)


def poisson_shot_noise(img: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Electron-counting statistics: signal-dependent shot noise.

    `dose` is a relative electrons-per-pixel scale. The image is mapped
    into a dose-electron domain, Poisson-sampled, and mapped back, so
    `dose` controls SNR directly (higher dose = less relative noise)
    independent of the 0-255 display range.
    """
    scale = max(dose, 1e-6) / 255.0
    lam = np.clip(img, 0, None).astype(np.float64) * scale
    sampled = rng.poisson(lam).astype(np.float32)
    return sampled / np.float32(scale)


def gaussian_read_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Amplifier/detector electronic read noise - additive, signal-independent."""
    return img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)


def apply_speckle_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative detector-gain variation ("speckle"): out = img*(1+N(0,sigma)).

    Distinguished from read noise by scaling with local signal rather than
    being purely additive - a different physical origin (gain variation
    across the detector) from electronic read noise.
    """
    return img * (1.0 + rng.normal(0.0, sigma, size=img.shape).astype(np.float32))


def apply_salt_and_pepper(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Impulse noise from dead/hot detector pixels and discrete discharge
    events: a small fraction of pixels driven to 0 or 255 independent of
    local signal."""
    out = img.copy()
    hit = rng.random(img.shape) < amount
    to_white = rng.random(img.shape) < 0.5
    out[hit & to_white] = 255.0
    out[hit & ~to_white] = 0.0
    return out


def apply_charging_streaks(img: np.ndarray, prob: float, intensity: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Local sample charging on insulating regions -> transient bright
    streaks along the slow (row) scan axis.

    Streak rows arrive as a Poisson-like process along image height (rare,
    independent events per row); each brightens its full row by
    `intensity`. Physically most relevant on insulating strip/routing
    material rather than conductive metal lines, but modeled here as a
    whole-row effect for simplicity, consistent with it being a slow-axis
    (per-scan-line) charging event rather than a localized one.
    """
    out = img.copy()
    h = img.shape[0]
    hits = rng.random(h) < prob
    if hits.any():
        out[hits, :] = out[hits, :] + intensity
    return out


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial illumination/collection-efficiency falloff toward the field edge."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (np.sqrt(cy ** 2 + cx ** 2) + 1e-9)
    falloff = 1.0 - strength * (r_norm ** 2)
    return img * falloff


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Detector-gain nonlinearity, applied in normalized [0,1] space."""
    x = np.clip(img, 0, 255) / 255.0
    return (x ** gamma) * 255.0


def apply_barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Imperfect beam-scan linearity / lens calibration -> radial (barrel
    for k>0, pincushion for k<0) distortion, strongest toward the field
    edge - the same reason a physically larger field of view (the Search
    image, at 10x the Reference's linear FOV) is more affected than the
    Reference for the same `k`.
    """
    if k == 0:
        return img
    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny, nx = (yy - cy) / cy, (xx - cx) / cx
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + k * r2
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_raster_shear_drift(img: np.ndarray, shear_amplitude_px: float, jitter_std_px: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Progressive row-to-row scan drift (linear shear across the slow
    axis) plus per-row vibration jitter - both consequences of finite
    scan-stabilization bandwidth during a slow raster acquisition.
    """
    h, w = img.shape
    row_idx = np.arange(h, dtype=np.float32)
    shear = shear_amplitude_px * (row_idx / max(h - 1, 1) - 0.5)
    jitter = rng.normal(0.0, jitter_std_px, size=h).astype(np.float32)
    shift = shear + jitter
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + shift[:, None]
    ys = np.tile(row_idx[:, None], (1, w))
    return cv2.remap(img.astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)


def rotation_scale_matrix(size: int, rotation_deg: float, extra_scale: float) -> np.ndarray:
    """Single source of truth for the residual-drift affine transform,
    shared by `apply_rotation_scale` (warps the image) and
    `transform_point` (must move ground truth through the exact same
    transform, or GT silently goes wrong for any point away from the image
    center - see `transform_point` docstring)."""
    center = (size / 2.0, size / 2.0)
    return cv2.getRotationMatrix2D(center, rotation_deg, extra_scale)


def apply_rotation_scale(img: np.ndarray, rotation_deg: float, extra_scale: float) -> np.ndarray:
    """Residual stage-rotation and magnification-calibration drift, applied
    AFTER the exact base 10x downsample. Models a hardware/calibration
    imperfection layered on top of a fixed nominal magnification - NOT the
    magnification itself being a random variable (see architecture plan
    section 4 for the full rationale).
    """
    if rotation_deg == 0 and extra_scale == 1.0:
        return img
    h, w = img.shape
    matrix = rotation_scale_matrix(w, rotation_deg, extra_scale)
    return cv2.warpAffine(img.astype(np.float32), matrix, (w, h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def transform_point(x: float, y: float, rotation_deg: float, extra_scale: float, size: int) -> tuple[float, float]:
    """Map a ground-truth point through the same rotation/scale transform
    applied to the image by `apply_rotation_scale` - NOT optional
    bookkeeping. `cv2.warpAffine` maps a source-image point p to
    destination position M @ [p; 1] (OpenCV's forward-matrix convention);
    a point 350px from the image center moves by roughly
    350 * sin(4 deg) =~ 24px under just a 4-degree rotation. Without this
    correction, ground truth for every rotation/scale-drift family would
    be wrong by tens of pixels for any crop whose true location isn't near
    the image center, silently invalidating that family's entire accuracy
    measurement - caught by `test_dataset_validation.py`'s on-disk
    GT-matches-Reference check during this generator's own Phase 0
    validation, not assumed away.
    """
    if rotation_deg == 0 and extra_scale == 1.0:
        return x, y
    matrix = rotation_scale_matrix(size, rotation_deg, extra_scale)
    vec = matrix @ np.array([x, y, 1.0])
    return float(vec[0]), float(vec[1])


def downsample_area_average(img: np.ndarray, factor: int) -> np.ndarray:
    """Physical 10:1 magnification ratio between Reference and Search
    fields of view, realized as area-average resampling - each Search
    pixel integrates the corresponding factor x factor patch of
    Reference-resolution signal, matching how a real detector pixel
    integrates incident signal over its finite area rather than
    point-sampling it.
    """
    h, w = img.shape
    return cv2.resize(img.astype(np.float32), (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def image_reference(crop: np.ndarray, rng: np.random.Generator, p: dict) -> np.ndarray:
    """Compose the Reference-image acquisition pipeline: mild blur + high
    dose + low read noise, and (only if enabled) a small fraction of the
    Search-side radiometric/geometric effects - a real Reference
    acquisition is typically slower/higher-dose than a production Search
    scan, so its degradation should be present but proportionally milder,
    not absent.
    """
    img = crop.astype(np.float32)
    img = gaussian_psf_blur(img, p["blur_sigma_ref_px"], p.get("astigmatism_ratio", 1.0))
    img = poisson_shot_noise(img, p["dose_reference"], rng)
    img = gaussian_read_noise(img, p["read_noise_sigma_ref"], rng)
    if p.get("vignette_strength", 0.0) > 0:
        img = apply_vignette(img, p["vignette_strength"] * 0.5)
    if p.get("gamma", 1.0) != 1.0:
        img = apply_gamma(img, p["gamma"])
    if p.get("barrel_k", 0.0) != 0.0:
        img = apply_barrel_distortion(img, p["barrel_k"] * 0.3)
    return np.clip(img, 0, 255).astype(np.uint8)


def image_search(fine_canvas: np.ndarray, rng: np.random.Generator, p: dict) -> np.ndarray:
    """Compose the full Search-image acquisition pipeline from the shared
    fine canvas: blur at fine resolution -> exact 10x area-average
    downsample -> optional residual rotation/scale drift -> raster
    shear/drift -> optional barrel distortion -> shot+read noise ->
    optional speckle/salt-pepper/charging -> optional vignette/gamma.
    """
    img = fine_canvas.astype(np.float32)
    img = gaussian_psf_blur(img, p["blur_sigma_search_fine_px"], p.get("astigmatism_ratio", 1.0))
    img = downsample_area_average(img, p["scale_factor"])
    if p.get("rotation_deg", 0.0) != 0.0 or p.get("extra_scale", 1.0) != 1.0:
        img = apply_rotation_scale(img, p.get("rotation_deg", 0.0), p.get("extra_scale", 1.0))
    img = apply_raster_shear_drift(img, p["shear_amplitude_px"], p["jitter_std_px"], rng)
    if p.get("barrel_k", 0.0) != 0.0:
        img = apply_barrel_distortion(img, p["barrel_k"])
    img = poisson_shot_noise(img, p["dose_search"], rng)
    img = gaussian_read_noise(img, p["read_noise_sigma_search"], rng)
    if p.get("speckle_sigma", 0.0) > 0:
        img = apply_speckle_noise(img, p["speckle_sigma"], rng)
    if p.get("salt_pepper_amount", 0.0) > 0:
        img = apply_salt_and_pepper(img, p["salt_pepper_amount"], rng)
    if p.get("charging_prob", 0.0) > 0:
        img = apply_charging_streaks(img, p["charging_prob"], p["charging_intensity"], rng)
    if p.get("vignette_strength", 0.0) > 0:
        img = apply_vignette(img, p["vignette_strength"])
    if p.get("gamma", 1.0) != 1.0:
        img = apply_gamma(img, p["gamma"])
    return np.clip(img, 0, 255).astype(np.uint8)
