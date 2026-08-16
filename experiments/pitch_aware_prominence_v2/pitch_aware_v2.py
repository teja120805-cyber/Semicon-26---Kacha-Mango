"""Pitch-aware prominence re-ranking, v2: a direct mechanistic fix for
`experiments/pitch_aware_prominence/`'s (v1's) failure mode.

v1 passed 6/7 integration-gate criteria on the production-seed benchmark
(0.7436 -> 0.7500) but failed an independent second-seed validation
(0.6618 -> 0.6324, 0 rescues / 4 breaks). Diagnosing those 4 breaks (see
this experiment's REPORT.md section 1) traced every single one to the SAME
root cause: v1's formula `score + gamma * prominence` treats prominence
SYMMETRICALLY - a candidate with no nearby competing peak gets a POSITIVE
bonus, not just a candidate WITH one getting a penalty. That bonus is what
caused every failure: a wrong, lower-scoring candidate would occasionally
get an outsized prominence value (one observed case: +0.22, vs. a normal
+0.03 to +0.09 range) purely because its own pitch-offset probe happened
to land in a locally low-scoring area - not because it was a genuinely
better match. Re-examining ALL 10 pairs v1 ever changed (6 on the
production seed, 4 on the second seed) found this same bonus-driven
pattern in every single one, with zero exceptions - including v1's own
`ch_combined_acquisition_005` production-seed regression, which was ALSO
bonus-driven.

v2's fix: only ever apply prominence as a PENALTY, never a bonus -
`score + gamma * min(prominence, 0.0)`. A candidate with no nearby
periodic competitor is left at its raw score, exactly as `rank_classical`
would score it; only a candidate WITH a strong nearby competitor (evidence
it might be a decoy) gets discounted. This is also the more
theoretically-justified formulation - "no competing peak nearby" is not by
itself positive evidence of correctness (many correct matches simply don't
have one), so it should never have been rewarded in the first place.

detect_pitch/compute_pitch_aware_prominence are unchanged from
`pitch_aware_prominence/pitch_aware.py` (copied here, not imported, to
keep this folder self-contained per this project's experiment-isolation
convention - see experiments/README.md) - only the ranking function
changes.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching
from pipeline.candidate_generation import Candidate
from pipeline.ranking import rank_classical

MIN_AUTOCORR_PEAK = 0.5
MIN_AUTOCORR_AMPLITUDE = 0.25
MIN_PITCH_PX = 3.0
MAX_PITCH_PX = 60.0
PITCH_SEARCH_TOLERANCE_PX = 2.0
NUM_PITCH_MULTIPLES = 2


def _first_genuine_pitch(profile: np.ndarray, center: int) -> float | None:
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
            trough_val = None
    return None


def detect_pitch(template: np.ndarray) -> list[tuple[float, float]]:
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
        return 0.0

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


def rank_pitch_aware_prominence_v2(candidates: list[Candidate], reference: np.ndarray, search: np.ndarray, *,
                                    top_k: int = 8, gamma: float = 1.0,
                                    search_tolerance_px: float = PITCH_SEARCH_TOLERANCE_PX,
                                    num_multiples: int = NUM_PITCH_MULTIPLES) -> list[Candidate]:
    """Identical to v1's rank_pitch_aware_prominence except the prominence
    term is clipped to <= 0.0 before being applied - a penalty-only
    formulation. gamma=0.0 is still provably identical to rank_classical."""
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
        penalty_only = min(prominence, 0.0)
        scored.append((c.score + gamma * penalty_only, c))
    scored.sort(key=lambda t: t[0], reverse=True)

    return [c for _, c in scored] + tail
