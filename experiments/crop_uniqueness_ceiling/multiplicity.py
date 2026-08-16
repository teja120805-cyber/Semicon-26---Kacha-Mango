"""Measure how many places in the Search image genuinely look like the
place the Reference crop actually came from.

This is deliberately ALGORITHM-FREE. It does not build a template, apply a
scale/rotation hypothesis, or run any part of the localization pipeline.
It compares Search-image content against other Search-image content, at
the same resolution and under the same acquisition, so the result is a
property of the DATA - an upper bound no method can exceed - rather than a
property of the matcher.

Two quantities per pair:

  identity(gt, pred) - ZNCC between the Search patch centered on ground
      truth and the Search patch centered on the location the pipeline
      predicted. Near 1.0 means the pipeline picked a spot that is, in the
      image, indistinguishable from the right one.

  multiplicity K - the number of distinct locations whose Search patch
      matches the ground-truth patch above a threshold. K == 1 means the
      crop's origin is uniquely determined by image content. K == n means
      n places are equally valid answers and any method must guess, with
      expected accuracy 1/n.

Never modifies pipeline/, generator/, or model/.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import matching

PATCH_PX = 100          # Reference footprint in Search pixels (1000px ref / 10x ratio)
NMS_RADIUS_PX = 10      # same distinctness radius production dedup uses


def extract_patch(img: np.ndarray, cx: float, cy: float, size: int = PATCH_PX):
    """Patch centered on (cx, cy), or None if it would fall off the edge."""
    half = size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if x0 < 0 or y0 < 0 or y0 + size > img.shape[0] or x0 + size > img.shape[1]:
        return None
    return img[y0:y0 + size, x0:x0 + size].astype(np.float32)


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """ZNCC between two equal-sized patches, via the same OpenCV routine
    the pipeline scores with (a 1x1 score map)."""
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    return float(matching.correlate(a, b)[0, 0])


def multiplicity(search: np.ndarray, gt_patch: np.ndarray, thresholds=(0.85, 0.90, 0.95)):
    """For each threshold, how many distinct locations in `search` match
    `gt_patch` at or above it (greedy NMS at NMS_RADIUS_PX).

    The ground-truth location itself always counts, so K >= 1 whenever the
    patch is valid; K == 1 means genuinely unique.
    """
    smap = matching.correlate(search, gt_patch)
    out, peaks_out = {}, {}
    for thr in thresholds:
        working = smap.copy()
        peaks = []
        while True:
            idx = int(np.argmax(working))
            y, x = divmod(idx, working.shape[1])
            s = float(working[y, x])
            if not np.isfinite(s) or s < thr or len(peaks) >= 200:
                break
            peaks.append((x + gt_patch.shape[1] / 2.0, y + gt_patch.shape[0] / 2.0, s))
            y0, y1 = max(0, y - NMS_RADIUS_PX), min(working.shape[0], y + NMS_RADIUS_PX + 1)
            x0, x1 = max(0, x - NMS_RADIUS_PX), min(working.shape[1], x + NMS_RADIUS_PX + 1)
            working[y0:y1, x0:x1] = -np.inf
        out[thr] = len(peaks)
        peaks_out[thr] = peaks
    return out, peaks_out
