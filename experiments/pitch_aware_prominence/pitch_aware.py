"""Pitch-aware prominence re-ranking: a surgical follow-up to
`experiments/prominence_rerank/`, which found a real signal (peak
prominence relative to a generic local annulus) that was net HARMFUL
because it penalized non-periodic true matches whenever *any* other
reasonably-similar structure happened to sit within a fixed 12px radius -
not specifically a periodic repeat.

This version replaces the generic annulus with a per-pair, MEASURED
periodic pitch: each candidate's own template (at its own winning
scale/rotation hypothesis) is self-correlated to detect whether it shows
significant axis-aligned periodicity at all (DRAM word/bit lines are
axis-aligned by construction in generator/pattern_renderer.py - see
`reports/ACCURACY_FORENSICS.md` and the direct pitch measurement in
`pyramid_periodicity_search/REPORT.md`, ~5-7.5px on a `dev_dense_periodic`
pair). If the template shows NO significant periodicity, prominence is
exactly 0.0 - no discount is ever applied, which specifically avoids
`prominence_rerank`'s false-positive mechanism on non-periodic families
(`strip_anchor`, `single_mat` without a dense preset). If it does, the
candidate is only checked for a competing peak at exact integer multiples
of the MEASURED pitch (+/- a small registration tolerance), not a generic
radius - a much narrower, better-targeted discriminator.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching
from pipeline.candidate_generation import Candidate
from pipeline.ranking import rank_classical

MIN_AUTOCORR_PEAK = 0.5
MIN_AUTOCORR_AMPLITUDE = 0.25  # peak-minus-preceding-trough - rejects gentle decay-tail ripples
MIN_PITCH_PX = 3.0
MAX_PITCH_PX = 60.0
PITCH_SEARCH_TOLERANCE_PX = 2.0
NUM_PITCH_MULTIPLES = 2


def _first_genuine_pitch(profile: np.ndarray, center: int) -> float | None:
    """Walks outward from the zero-lag center looking for a genuine
    oscillation: a local minimum (trough) followed by a local maximum
    (peak) whose absolute height AND rise-from-trough both clear their
    thresholds. Requiring both - not just an absolute-height local max -
    is what separates real periodicity (dev_dense_periodic/dev_single_mat:
    trough ~0.22-0.24, peak ~0.92-0.96, amplitude ~0.70) from a
    non-periodic pair's gentle decay-tail ripple (dev_strip_anchor:
    trough ~0.25, "peak" ~0.42, amplitude ~0.17) - verified directly by
    inspecting both families' radial profiles side by side; a threshold
    on absolute peak height alone (first attempt, MIN_AUTOCORR_PEAK=0.3,
    no amplitude check) incorrectly fired on every family, including ones
    the dataset's own family descriptions call non-periodic."""
    n = len(profile)
    lo = int(MIN_PITCH_PX)
    hi = int(min(MAX_PITCH_PX, n - center - 2))
    trough_val = None
    for lag in range(lo, hi):
        idx = center + lag
        v = profile[idx]
        is_local_min = v <= profile[idx - 1] and v <= profile[idx + 1]
        is_local_max = v >= profile[idx - 1] and v >= profile[idx + 1]
        if trough_val is None:
            if is_local_min:
                trough_val = v
            continue
        if is_local_max:
            if v >= MIN_AUTOCORR_PEAK and (v - trough_val) >= MIN_AUTOCORR_AMPLITUDE:
                return float(lag)
            trough_val = None  # this local max didn't qualify - keep scanning for the next trough/peak pair
    return None


def detect_pitch(template: np.ndarray) -> list[tuple[float, float]]:
    """Self-correlates the template and looks for genuine axis-aligned
    periodicity (word/bit line arrays are axis-aligned by construction -
    see generator/pattern_renderer.py) along the row and column axes
    independently. Returns a list of (dx, dy) pitch vectors - empty if
    neither axis shows a genuine trough-then-peak oscillation."""
    h, w = template.shape
    pad = max(h, w) // 2
    padded = np.pad(template.astype(np.float32), pad, mode="reflect")
    ac = cv2.matchTemplate(padded, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2

    pitches: list[tuple[float, float]] = []
    row_pitch = _first_genuine_pitch(ac[cy], cx)
    if row_pitch is not None:
        pitches.append((row_pitch, 0.0))
    col_pitch = _first_genuine_pitch(ac[:, cx], cy)
    if col_pitch is not None:
        pitches.append((0.0, col_pitch))

    return pitches


def compute_pitch_aware_prominence(reference: np.ndarray, search: np.ndarray, candidate: Candidate, *,
                                    search_tolerance_px: float = PITCH_SEARCH_TOLERANCE_PX,
                                    num_multiples: int = NUM_PITCH_MULTIPLES) -> float:
    template = matching.build_template(reference, candidate.scale, candidate.rotation_deg)
    pitches = detect_pitch(template)
    if not pitches:
        return 0.0  # no measured periodicity on this hypothesis - never discount

    score_map = matching.correlate(search, template)
    th, tw = template.shape
    peak_px = candidate.x - tw / 2.0
    peak_py = candidate.y - th / 2.0
    h, w = score_map.shape

    max_competitor = None
    for (pdx, pdy) in pitches:
        for k in range(1, num_multiples + 1):
            for sign in (1, -1):
                ox = peak_px + sign * k * pdx
                oy = peak_py + sign * k * pdy
                x0, x1 = int(max(0, ox - search_tolerance_px)), int(min(w, ox + search_tolerance_px + 1))
                y0, y1 = int(max(0, oy - search_tolerance_px)), int(min(h, oy + search_tolerance_px + 1))
                if x1 <= x0 or y1 <= y0:
                    continue
                window = score_map[y0:y1, x0:x1]
                finite = window[np.isfinite(window)]
                if finite.size == 0:
                    continue
                local_max = float(finite.max())
                max_competitor = local_max if max_competitor is None else max(max_competitor, local_max)

    if max_competitor is None:
        return 0.0

    return float(candidate.score) - max_competitor


def rank_pitch_aware_prominence(candidates: list[Candidate], reference: np.ndarray, search: np.ndarray, *,
                                 top_k: int = 8, gamma: float = 1.0,
                                 search_tolerance_px: float = PITCH_SEARCH_TOLERANCE_PX,
                                 num_multiples: int = NUM_PITCH_MULTIPLES) -> list[Candidate]:
    ranked = rank_classical(candidates)
    if gamma == 0.0 or len(ranked) < 2:
        return ranked

    head = ranked[:top_k]
    tail = ranked[top_k:]

    scored = []
    for c in head:
        prominence = compute_pitch_aware_prominence(reference, search, c,
                                                      search_tolerance_px=search_tolerance_px,
                                                      num_multiples=num_multiples)
        scored.append((c.score + gamma * prominence, c))
    scored.sort(key=lambda t: t[0], reverse=True)

    return [c for _, c in scored] + tail
