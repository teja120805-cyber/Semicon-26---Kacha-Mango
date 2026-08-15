"""Triplet dataset for the learned candidate generator, mirroring
model/dataset.py::TripletPatchDataset's exact hard-negative-mining logic
(same classical candidate pool used to mine negatives - the actual decoys
the classical matcher itself finds plausible) but pointed at the expanded,
16-family dev_data/ this experiment generates, instead of the 3-family
production development split that caused the prior reranker's overfitting.
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


class ExpandedTripletPatchDataset(Dataset):
    def __init__(self, data_root: str, family_filter: list[str], seed: int = 20260201, negatives_per_pair: int = 3):
        self.data_root = data_root
        with open(os.path.join(data_root, "development", "ground_truth.json")) as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r["structural_family"] in family_filter]
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
