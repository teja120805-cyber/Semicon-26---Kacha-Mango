"""ORB keypoint-based candidate generation: an independent proposal
mechanism for the Search-image location of the Reference, fused into the
existing scale/rotation-hypothesis candidate pool.

Rationale (grounded in reports/ACCURACY_FORENSICS.md): whole-template ZNCC
correlation (pipeline/candidate_generation.py) enumerates 99 scale/rotation
hypotheses and keeps each hypothesis's strongest peak(s) - but on
`candidate_generation`-stage failures (31/156, the single largest failure
category), the TRUE location is never a competitive peak under ANY tested
hypothesis, most often because periodic/aliasing content pulls the whole-
template correlation surface's global maxima toward a repeat-pitch decoy
instead. The already-rejected `wider_candidate_pool` showed more
peaks-per-hypothesis doesn't help this (the true location isn't even a
LOCAL max on that surface, so keeping more peaks from the SAME surface
can't surface it), and the already-rejected `periodicity/` showed
alternative whole-template SCORING functions (gradient-domain,
gradient+intensity ensemble) don't help either.

This experiment tries something structurally different: sparse, locally-
distinctive ORB keypoint correspondence (corners/blobs, not periodic bulk
texture) between the Reference and the Search image, robustly fit via
RANSAC to a similarity transform, used to PROPOSE an additional candidate
location outside the fixed 99-hypothesis grid entirely. The proposal is
then scored with the exact same ZNCC metric as every other candidate (via
a small local correlation at the transform-estimated continuous
scale/rotation) so it competes fairly under the unmodified rank_classical
arg-max - this experiment adds a candidate SOURCE, not a new scoring
function or ranking rule.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching
from pipeline.candidate_generation import Candidate

ORB_N_FEATURES = 1000
RATIO_TEST_THRESHOLD = 0.75
MIN_RAW_MATCHES = 6
MIN_RANSAC_INLIERS = 6
LOCAL_SEARCH_MARGIN_PX = 12  # local correlation window half-size around the keypoint-estimated location

# Default ORB thresholds (edgeThreshold=31, fastThreshold=20) barely detect
# any keypoints on this dataset's smooth, low-texture synthetic SEM imagery
# (~19 keypoints on a 100x100 template - verified directly). Loosened
# detector thresholds (smaller edgeThreshold/patchSize/fastThreshold, more
# pyramid levels) bring that up to ~970-1000, which is what makes any
# downstream matching possible at all on this data.
_ORB_KWARGS = dict(scaleFactor=1.1, nlevels=12, edgeThreshold=5, patchSize=15, fastThreshold=5)


def _to_u8(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _orb_detect(img: np.ndarray, n_features: int = ORB_N_FEATURES):
    orb = cv2.ORB_create(nfeatures=n_features, **_ORB_KWARGS)
    kp, desc = orb.detectAndCompute(_to_u8(img), None)
    return kp, desc


def generate_keypoint_candidate(reference: np.ndarray, search: np.ndarray, *,
                                 ratio_threshold: float = RATIO_TEST_THRESHOLD,
                                 min_raw_matches: int = MIN_RAW_MATCHES,
                                 min_inliers: int = MIN_RANSAC_INLIERS,
                                 nominal_scale: float = 10.0,
                                 local_margin_px: int = LOCAL_SEARCH_MARGIN_PX) -> Candidate | None:
    """Returns one additional Candidate proposed via ORB keypoint matching
    + RANSAC similarity-transform estimation, or None if the mechanism
    can't produce a confident proposal on this pair (too few keypoints /
    matches / inliers) - in which case the pool falls back to the
    unmodified classical grid alone, exactly as if this function didn't
    exist.
    """
    template = matching.build_template(reference, nominal_scale, 0.0)

    kp_t, desc_t = _orb_detect(template)
    kp_s, desc_s = _orb_detect(search)
    if desc_t is None or desc_s is None or len(kp_t) < 4 or len(kp_s) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc_t, desc_s, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    if len(good) < min_raw_matches:
        return None

    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        pts_t, pts_s, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000, confidence=0.99,
    )
    if transform is None or inlier_mask is None:
        return None
    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_inliers:
        return None

    # Decompose the estimated similarity transform for an implied
    # scale/rotation (informational only - the actual candidate score comes
    # from a fresh local correlation below, not from this decomposition).
    a, b = float(transform[0, 0]), float(transform[0, 1])
    implied_scale_ratio = float(np.hypot(a, b))  # template-pixel -> search-pixel scale factor
    implied_rotation_deg = float(np.degrees(np.arctan2(-b, a)))

    th, tw = template.shape
    center_t = np.array([[tw / 2.0, th / 2.0]], dtype=np.float32).reshape(-1, 1, 2)
    center_s = cv2.transform(center_t, transform).reshape(2)
    prop_x, prop_y = float(center_s[0]), float(center_s[1])

    # Guard against a degenerate/out-of-range implied scale (e.g. a bad
    # RANSAC fit skewing the template far outside plausible sizes) - fall
    # back to the nominal scale ratio for building the verification
    # template rather than trusting an outlier decomposition. 7-13 gives
    # generous margin around the production 9-11 grid.
    absolute_scale = nominal_scale / implied_scale_ratio if implied_scale_ratio > 1e-3 else nominal_scale
    if not (7.0 <= absolute_scale <= 13.0):
        absolute_scale = nominal_scale
    rotation_for_template = implied_rotation_deg if abs(implied_rotation_deg) <= 8.0 else 0.0

    verify_template = matching.build_template(reference, absolute_scale, rotation_for_template)
    vh, vw = verify_template.shape

    sh, sw = search.shape
    x0 = int(max(0, prop_x - vw / 2.0 - local_margin_px))
    x1 = int(min(sw, prop_x + vw / 2.0 + local_margin_px))
    y0 = int(max(0, prop_y - vh / 2.0 - local_margin_px))
    y1 = int(min(sh, prop_y + vh / 2.0 + local_margin_px))
    if x1 - x0 < vw or y1 - y0 < vh:
        return None
    window = search[y0:y1, x0:x1]

    score_map = matching.correlate(window, verify_template)
    if score_map.size == 0:
        return None
    idx = int(np.argmax(score_map))
    wy, wx = divmod(idx, score_map.shape[1])
    score = float(score_map[wy, wx])
    if not np.isfinite(score):
        return None

    cand_x = x0 + wx + vw / 2.0
    cand_y = y0 + wy + vh / 2.0

    return Candidate(x=cand_x, y=cand_y, score=score, scale=absolute_scale,
                      rotation_deg=rotation_for_template, template_size=vw)
