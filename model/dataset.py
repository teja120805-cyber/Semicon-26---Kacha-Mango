"""Builds (anchor, positive, negative) triplets for training the embedding
model, using the classical pipeline's own candidate pool to mine hard
negatives - directly targeting the decoys reports/V2_BASELINE_REPORT.md identifies
(periodic repeats, same-preset boundaries), rather than easier random-crop
negatives that would look good in isolation but not transfer.

Training data comes from the `development` split ONLY (see
model/TRAINING_PROTOCOL.md) - validation/held_out/challenge/cross_generator
are never read by this module, so there is no path by which they could
leak into training even by accident.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline import candidate_generation, feature_extraction

PATCH_SIZE = 100
MIN_NEGATIVE_DISTANCE_PX = 20.0


def _load_dev_manifest(data_root: str) -> list[dict]:
    with open(os.path.join(data_root, "development", "ground_truth.json")) as f:
        records = json.load(f)
    # The bonus acquisition-variant demo set (generator/dataset_generator.py
    # ::generate_acquisition_variant_demo) has its own manifest.json under
    # development/acquisition_variants/ and is never mixed into this one -
    # nothing to filter here, this is just documenting why not.
    return records


class TripletPatchDataset(Dataset):
    """One item = (anchor patch from Reference, positive patch from Search
    at GT, negative patch from Search at a hard classical-candidate
    location). Negatives are mined once at construction time (not re-mined
    per epoch) so the dataset is a fixed, fully reproducible object given a
    seed - not a moving target across runs.
    """

    def __init__(self, data_root: str, seed: int = 20260101, negatives_per_pair: int = 3):
        self.data_root = data_root
        self.records = _load_dev_manifest(data_root)
        self.negatives_per_pair = negatives_per_pair
        self.rng = np.random.default_rng(seed)
        self.items = self._build_items()

    def _build_items(self) -> list[dict]:
        items = []
        for r in self.records:
            ref_path = os.path.join(self.data_root, r["reference_path"])
            search_path = os.path.join(self.data_root, r["search_path"])
            ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            search = cv2.imread(search_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            candidates = candidate_generation.build_candidate_pool(ref, search)
            candidates = candidate_generation.deduplicate_by_location(candidates, radius_px=15.0)
            hard_negatives = [
                c for c in candidates
                if np.hypot(c.x - r["gt_x"], c.y - r["gt_y"]) > MIN_NEGATIVE_DISTANCE_PX
            ][: self.negatives_per_pair]
            if not hard_negatives:
                continue

            for neg in hard_negatives:
                items.append({
                    "reference_path": ref_path, "search_path": search_path,
                    "structural_family": r["structural_family"],
                    "gt_x": r["gt_x"], "gt_y": r["gt_y"],
                    "neg_x": neg.x, "neg_y": neg.y,
                })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        ref = cv2.imread(item["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(item["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)

        anchor = feature_extraction.normalize_patch(
            cv2.resize(ref, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        )
        positive = feature_extraction.normalize_patch(
            feature_extraction.extract_patch(search, item["gt_x"], item["gt_y"], PATCH_SIZE)
        )
        negative = feature_extraction.normalize_patch(
            feature_extraction.extract_patch(search, item["neg_x"], item["neg_y"], PATCH_SIZE)
        )
        anchor_t = torch.from_numpy(anchor).unsqueeze(0)
        positive_t = torch.from_numpy(positive).unsqueeze(0)
        negative_t = torch.from_numpy(negative).unsqueeze(0)
        return anchor_t, positive_t, negative_t
