"""Low-level rendering primitives for one DRAM mat's cell array, at 1 nm/px.

Physical basis: a folded-bitline DRAM sub-array is periodic word lines
(rows) and bit lines (columns) with storage-node contacts on a checkerboard
subset of their intersections (folded-bitline sensing shares one bitline
pair per two columns, so only every other cell in a pair connects to a given
sense amp). Real fabricated lines are never on a perfect grid:

- Overlay/placement error accumulates along a scan/exposure field as a slow
  random walk rather than landing independently per line (Orji et al.,
  2018, "Metrology for the next generation of semiconductor devices").
- Each line's drawn width varies independently around its nominal critical
  dimension (line-edge roughness).
- Adjacent lines whose as-drawn spacing falls below a process-dependent
  threshold can bridge together (capillary/etch-induced pattern collapse).

None of this is randomness for its own sake - it is what makes two mats
built from the *same* preset still not be pixel-identical, without an
injected fingerprint.
"""
from __future__ import annotations

import cv2
import numpy as np

POSITION_JITTER_NM = 1.4
WIDTH_JITTER_FRACTION = 0.10

BACKGROUND_VALUE = 40.0
LINE_VALUE = 200.0
CONTACT_VALUE = 235.0


def jittered_line_positions(n_lines: int, pitch_nm: float, rng: np.random.Generator) -> np.ndarray:
    """Nominal-pitch line centers perturbed by a cumulative random walk.

    Each line's offset from its nominal grid position is the running sum of
    independent per-line placement errors (`POSITION_JITTER_NM` std per
    step), matching cumulative overlay drift rather than per-line-
    independent placement error.
    """
    steps = rng.normal(0.0, POSITION_JITTER_NM, size=n_lines)
    walk = np.cumsum(steps)
    walk -= walk.mean()
    nominal = np.arange(n_lines, dtype=np.float64) * pitch_nm
    return nominal + walk


def _rasterize_lines(size_px: int, centers: np.ndarray, width_px: float,
                      width_jitter_frac: float, rng: np.random.Generator,
                      linewidth_bias_px: float = 0.0) -> np.ndarray:
    """Boolean 1D coverage mask along one axis: True where some line (with
    independent per-line width jitter plus an optional deterministic global
    bias, representing a systematic over/under exposure or etch bias) covers
    that position."""
    axis = np.arange(size_px, dtype=np.float64)[:, None]
    widths = width_px + linewidth_bias_px + rng.normal(
        0.0, max(width_px * width_jitter_frac, 1e-6), size=len(centers)
    )
    widths = np.clip(widths, width_px * 0.3, width_px * 2.0)
    lo = (centers - widths / 2.0)[None, :]
    hi = (centers + widths / 2.0)[None, :]
    covered = (axis >= lo) & (axis < hi)
    return covered.any(axis=1)


def bridge_narrow_gaps(mask: np.ndarray, min_gap_px: float, collapse_prob: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Structural pattern-collapse: fill a False-run strictly between two
    True-runs when it is narrower than `min_gap_px`, independently with
    probability `collapse_prob` per gap. Edge runs (touching either end of
    the array) are never bridged - there is no "adjacent line" to bridge to.

    This models real capillary/etch-induced bridging between adjacent lines
    whose as-drawn spacing falls below a process-dependent threshold - it
    silently removes a line-pair boundary a matcher might otherwise rely on,
    which is exactly why it belongs in a difficulty-generating pipeline
    rather than only being cosmetic.
    """
    if min_gap_px <= 0 or not mask.any() or mask.all():
        return mask
    out = mask.copy()
    m = mask.astype(np.int8)
    # Run-length encode without artificial edge padding: a gap run is only
    # "between two lines" if it is neither the first nor the last run, which
    # RLE gives for free (consecutive runs always alternate value) - no
    # padding trick needed, and no risk of a fake boundary line making an
    # edge gap look bridgeable (the bug an earlier padding-based version of
    # this function had).
    change_points = np.flatnonzero(np.diff(m)) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [len(m)]))
    run_values = m[run_starts]
    for i in range(1, len(run_starts) - 1):
        if run_values[i] == 0 and (run_ends[i] - run_starts[i]) < min_gap_px:
            if rng.random() < collapse_prob:
                out[run_starts[i]:run_ends[i]] = True
    return out


def _round_corners(canvas: np.ndarray, radius_px: float) -> np.ndarray:
    """Lithography/etch never produces perfectly sharp corners - a small
    morphological open+close on the binarized structural mask softens
    inside and outside corners symmetrically without shifting line
    centerlines (unlike a Gaussian blur, which is an acquisition effect
    applied later and would also soften genuinely straight edges)."""
    k = max(1, int(round(radius_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    is_structure = (canvas > (BACKGROUND_VALUE + LINE_VALUE) / 2.0).astype(np.uint8)
    opened = cv2.morphologyEx(is_structure, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    out = canvas.copy()
    out[(closed > 0) & (canvas < LINE_VALUE)] = LINE_VALUE
    out[(closed == 0) & (canvas >= LINE_VALUE)] = BACKGROUND_VALUE
    return out


def render_dram_cell_array(size_px: int, word_pitch_nm: float, bit_pitch_nm: float,
                            feature_nm: float, rng: np.random.Generator, *,
                            linewidth_bias_nm: float = 0.0,
                            collapse_enabled: bool = False,
                            collapse_threshold_nm: float = 10.0,
                            collapse_prob: float = 0.65,
                            corner_rounding_px: float = 0.0) -> np.ndarray:
    """Render one size_px x size_px folded-bitline DRAM mat at 1 nm/px.

    Word lines run horizontally (full-width bands), bit lines run
    vertically; storage-node contacts occupy the (word_rank + bit_rank)
    even subset of their intersections. Contrast is deliberately NOT varied
    per preset here - only geometry (pitch, width) is - so a mat's identity
    comes from real structural difference, never an injected brightness
    fingerprint (see reports/V2_ARCHITECTURE_PLAN.md section 2).
    """
    n_words = int(size_px / word_pitch_nm) + 3
    n_bits = int(size_px / bit_pitch_nm) + 3
    word_centers = jittered_line_positions(n_words, word_pitch_nm, rng)
    bit_centers = jittered_line_positions(n_bits, bit_pitch_nm, rng)

    word_mask = _rasterize_lines(size_px, word_centers, feature_nm, WIDTH_JITTER_FRACTION, rng, linewidth_bias_nm)
    bit_mask = _rasterize_lines(size_px, bit_centers, feature_nm, WIDTH_JITTER_FRACTION, rng, linewidth_bias_nm)

    if collapse_enabled:
        word_mask = bridge_narrow_gaps(word_mask, collapse_threshold_nm, collapse_prob, rng)
        bit_mask = bridge_narrow_gaps(bit_mask, collapse_threshold_nm, collapse_prob, rng)

    canvas = np.full((size_px, size_px), BACKGROUND_VALUE, dtype=np.float32)
    canvas[word_mask, :] = LINE_VALUE
    canvas[:, bit_mask] = LINE_VALUE

    contact_half = max(1, int(round(feature_nm / 2.0)))
    word_idx = np.clip(np.round(word_centers).astype(np.int64), 0, size_px - 1)
    bit_idx = np.clip(np.round(bit_centers).astype(np.int64), 0, size_px - 1)
    ii, jj = np.meshgrid(np.arange(len(word_idx)), np.arange(len(bit_idx)), indexing="ij")
    parity_match = (ii + jj) % 2 == 0
    rows = word_idx[ii[parity_match]]
    cols = bit_idx[jj[parity_match]]
    contact_points = np.zeros((size_px, size_px), dtype=np.uint8)
    contact_points[rows, cols] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * contact_half + 1, 2 * contact_half + 1))
    contact_points = cv2.dilate(contact_points, kernel)
    canvas = np.where(contact_points > 0, np.float32(CONTACT_VALUE), canvas).astype(np.float32)

    if corner_rounding_px > 0:
        canvas = _round_corners(canvas, corner_rounding_px)

    return canvas
