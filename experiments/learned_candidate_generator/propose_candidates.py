"""Dense-grid candidate proposal from a trained embedding network: unlike
`pipeline/ranking.py::rank_with_model` (which only re-ranks the classical
pipeline's own top-12 shortlist), this evaluates learned similarity at a
grid of locations spanning the WHOLE search image, so it can propose
locations the classical ZNCC search never considered competitive at all -
directly targeting the candidate-generation failure mode, not the ranking
one.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from pipeline import feature_extraction, matching

PATCH_SIZE = 100
GRID_STEP_PX = 15
SUPPRESSION_RADIUS_PX = 20
BATCH_SIZE = 256


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def propose_learned_candidates(model, reference: np.ndarray, search: np.ndarray, device: str = "cpu",
                                grid_step: int = GRID_STEP_PX, top_k: int = 20) -> list[dict]:
    """Returns up to `top_k` (x, y, learned_score, ncc_score) candidates,
    deduplicated by location, sorted by learned similarity score."""
    model.eval()
    ref_patch = feature_extraction.normalize_patch(
        cv2.resize(reference.astype(np.float32), (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    )
    ref_tensor = torch.from_numpy(ref_patch).unsqueeze(0).unsqueeze(0).to(device)

    search_f32 = search.astype(np.float32)
    h, w = search_f32.shape
    half = PATCH_SIZE // 2
    xs = list(range(half, w - half, grid_step))
    ys = list(range(half, h - half, grid_step))
    centers = [(x, y) for y in ys for x in xs]

    with torch.no_grad():
        ref_emb = model(ref_tensor)

        all_scores = np.zeros(len(centers), dtype=np.float32)
        for start in range(0, len(centers), BATCH_SIZE):
            batch_centers = centers[start:start + BATCH_SIZE]
            patches = np.stack([
                feature_extraction.normalize_patch(
                    feature_extraction.extract_patch(search_f32, cx, cy, PATCH_SIZE)
                ) for cx, cy in batch_centers
            ])
            batch_tensor = torch.from_numpy(patches).unsqueeze(1).to(device)
            embs = model(batch_tensor)
            sims = torch.nn.functional.cosine_similarity(ref_emb, embs).cpu().numpy()
            all_scores[start:start + len(batch_centers)] = sims

    order = np.argsort(-all_scores)
    kept: list[dict] = []
    for idx in order:
        cx, cy = centers[idx]
        if all(_dist((cx, cy), (k["x"], k["y"])) > SUPPRESSION_RADIUS_PX for k in kept):
            kept.append({"x": float(cx), "y": float(cy), "learned_score": float(all_scores[idx])})
        if len(kept) >= top_k:
            break

    # Score each proposed location classically too (base hypothesis only:
    # scale=10.0, rotation=0.0) so the existing rank_classical stage can
    # compare learned proposals against classical candidates on the same
    # ZNCC scale. Note: this under-scores a learned candidate that's only
    # correct under rotation/scale drift, a known, documented limitation
    # (see REPORT.md) rather than an unstated one.
    template = matching.build_template(reference.astype(np.float32), 10.0, 0.0)
    score_map = matching.correlate(search_f32, template)
    for cand in kept:
        px = int(np.clip(round(cand["x"] - template.shape[1] / 2.0), 0, score_map.shape[1] - 1))
        py = int(np.clip(round(cand["y"] - template.shape[0] / 2.0), 0, score_map.shape[0] - 1))
        cand["ncc_score"] = float(score_map[py, px])
        cand["scale"] = 10.0
        cand["rotation_deg"] = 0.0
        cand["template_size"] = template.shape[0]

    return kept
