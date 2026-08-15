"""Per-candidate feature extraction: normalized patches for the learned
re-ranking model, and simple score-distribution features for confidence /
ambiguity reporting.
"""
from __future__ import annotations

import numpy as np


def extract_patch(search: np.ndarray, x: float, y: float, size: int = 100) -> np.ndarray:
    """Crop a size x size patch centered at (x, y) from the Search image,
    zero-padding if the requested window falls outside the image bounds."""
    h, w = search.shape
    half = size / 2.0
    x0, y0 = int(round(x - half)), int(round(y - half))
    patch = np.zeros((size, size), dtype=np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = search[sy0:sy1, sx0:sx1]
    return patch


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std float32 patch - the input normalization used by
    both the classical texture checks and the learned embedding model, so
    neither is sensitive to absolute brightness/dose differences between
    Reference and Search acquisitions."""
    p = patch.astype(np.float32)
    std = float(p.std())
    if std < 1e-6:
        return p - p.mean()
    return (p - p.mean()) / std


def ambiguity_ratio(sorted_scores: list[float]) -> float:
    """Second-best / best ZNCC-score ratio across the whole candidate pool.
    Close to 1.0 means the top match is not clearly distinguished from a
    runner-up (e.g. a periodic-repeat or same-preset-mat decoy); close to
    0.0 means the top match stands out clearly."""
    if len(sorted_scores) < 2:
        return 0.0
    best, second = sorted_scores[0], sorted_scores[1]
    if best <= 1e-6:
        return 1.0
    return float(np.clip(second / best, 0.0, 1.0))
