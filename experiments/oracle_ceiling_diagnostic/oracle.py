"""Oracle warp sweep: what is the BEST ZNCC score achievable at a given
location, under any (scale, rotation) warp in a generous continuous range?

This is a DIAGNOSTIC, not a pipeline candidate - it reads ground truth, so
it can never be a localization method. Its only job is to decide, per
failing pair, whether the classical winner beat the true location because:

  (a) the hypothesis grid never built the right template  -> GEOMETRIC
      (a denser/continuous warp search would fix it; ZNCC is fine), or
  (b) even under a perfect warp the decoy still scores higher -> PHOTOMETRIC
      /genuine self-similarity (no ZNCC-based method can fix it).

Imports pipeline.matching unmodified; never writes to pipeline/.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching


def build_template_cached(reference_resized: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Rotation half of pipeline.matching.build_template, applied to an
    already-resized reference. Verified bit-identical to calling
    build_template(reference, scale, rotation) directly - see
    verify_equivalence() below, which is asserted at run start."""
    size = reference_resized.shape[0]
    if rotation_deg == 0.0:
        return reference_resized.astype(np.float32)
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), rotation_deg, 1.0)
    template = cv2.warpAffine(reference_resized, matrix, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return template.astype(np.float32)


def resize_reference(reference: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resize half of pipeline.matching.build_template."""
    w = reference.shape[1]
    size = max(8, int(round(w / scale_factor)))
    return cv2.resize(reference.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def verify_equivalence(reference: np.ndarray) -> None:
    """Assert the cached decomposition reproduces the production function
    exactly, so no oracle number depends on a shortcut."""
    for scale in (9.0, 9.73, 10.4, 11.0):
        for rot in (0.0, -2.5, 1.3, 5.0):
            direct = matching.build_template(reference, scale, rot)
            cached = build_template_cached(resize_reference(reference, scale), rot)
            if not np.array_equal(direct, cached):
                raise AssertionError(f"cached template != production build_template at {scale},{rot}")


def score_at_location(search: np.ndarray, template: np.ndarray,
                      cx: float, cy: float, tolerance_px: float) -> float:
    """Max ZNCC over template placements whose CENTER lands within
    tolerance_px (Chebyshev) of (cx, cy).

    Cropping is exact, not an approximation: TM_CCOEFF_NORMED normalizes
    within each template-sized window independently, so scores computed on
    a crop equal the corresponding entries of the full-image score map.
    """
    th, tw = template.shape
    tol = int(np.ceil(tolerance_px))
    # Desired top-left range for the template.
    tl_x = cx - tw / 2.0
    tl_y = cy - th / 2.0
    x0 = int(np.floor(tl_x)) - tol
    y0 = int(np.floor(tl_y)) - tol
    x1 = int(np.ceil(tl_x)) + tol + tw   # exclusive crop edge
    y1 = int(np.ceil(tl_y)) + tol + th
    H, W = search.shape
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if (x1c - x0c) < tw or (y1c - y0c) < th:
        return float("nan")  # location too close to the edge to evaluate
    crop = search[y0c:y1c, x0c:x1c]
    smap = matching.correlate(crop, template)
    # Map score-map indices back to absolute template-center coordinates and
    # keep only placements genuinely within tolerance of the target.
    ys, xs = np.mgrid[0:smap.shape[0], 0:smap.shape[1]]
    centers_x = xs + x0c + tw / 2.0
    centers_y = ys + y0c + th / 2.0
    within = (np.abs(centers_x - cx) <= tolerance_px) & (np.abs(centers_y - cy) <= tolerance_px)
    if not within.any():
        return float("nan")
    return float(np.nanmax(np.where(within, smap, -np.inf)))


def sweep_best(reference: np.ndarray, search: np.ndarray, targets: dict[str, tuple[float, float]],
               scales: np.ndarray, rotations: np.ndarray, tolerance_px: float) -> dict[str, dict]:
    """For each named target location, the best ZNCC over the full
    (scales x rotations) warp set, plus the warp that achieved it.

    Resizes are computed once per scale and shared across rotations and
    across all target locations - the expensive 1000x1000 INTER_AREA
    resample would otherwise dominate runtime.
    """
    best = {name: {"score": -np.inf, "scale": np.nan, "rotation_deg": np.nan} for name in targets}
    for scale in scales:
        resized = resize_reference(reference, float(scale))
        for rot in rotations:
            template = build_template_cached(resized, float(rot))
            for name, (cx, cy) in targets.items():
                s = score_at_location(search, template, cx, cy, tolerance_px)
                if np.isfinite(s) and s > best[name]["score"]:
                    best[name] = {"score": s, "scale": float(scale), "rotation_deg": float(rot)}
    for name in best:
        if not np.isfinite(best[name]["score"]):
            best[name]["score"] = float("nan")
    return best
