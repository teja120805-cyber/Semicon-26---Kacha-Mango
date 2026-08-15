"""Validates an already-generated dataset directory on disk: dimensions,
grayscale, scale relationship, GT correctness, coordinate convention,
reproducibility, and metadata completeness.

This is deliberately separate from `test_gt_safety.py` (which tests the
*generation functions* in isolation) - this module tests the *files a run
actually wrote*, which is what the brief's Phase 3 validation step ("Only
after the small dataset passes validation should you generate the larger
dataset") is actually asking for.

Usage: `python -m generator.test_dataset_validation [data_root] [--splits a,b]`
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from . import dataset_generator as dg

REQUIRED_METADATA_FIELDS = [
    "pair_id", "seed", "generator_version", "reference_path", "search_path",
    "gt_x", "gt_y", "structural_family", "split", "crop_mode",
    "crop_mode_fallback_used", "crop_mode_tries", "mat_ids", "num_mats",
    "presets", "crosses_mat_boundary", "crosses_strip_boundary", "same_preset_boundary",
    "periodicity_score", "uniqueness_score", "mat_size_nm", "strip_width_nm",
    "degradation_params", "timestamp",
]


def _load_split(data_root: str, split: str) -> list[dict]:
    path = os.path.join(data_root, split, "ground_truth.json")
    with open(path) as f:
        return json.load(f)


def validate_split(data_root: str, split: str) -> None:
    records = _load_split(data_root, split)
    assert records, f"[{split}] manifest is empty"

    pair_ids = set()
    for r in records:
        missing = [k for k in REQUIRED_METADATA_FIELDS if k not in r]
        assert not missing, f"[{split}] {r.get('pair_id')} missing metadata fields: {missing}"

        assert r["pair_id"] not in pair_ids, f"[{split}] duplicate pair_id {r['pair_id']}"
        pair_ids.add(r["pair_id"])

        ref_path = os.path.join(data_root, r["reference_path"])
        search_path = os.path.join(data_root, r["search_path"])
        assert os.path.isfile(ref_path), f"[{split}] missing Reference file {ref_path}"
        assert os.path.isfile(search_path), f"[{split}] missing Search file {search_path}"

        ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        search_img = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
        assert ref_img is not None and search_img is not None, f"[{split}] unreadable PNG for {r['pair_id']}"
        assert ref_img.ndim == 2, f"[{split}] {r['pair_id']} Reference is not single-channel grayscale"
        assert search_img.ndim == 2, f"[{split}] {r['pair_id']} Search is not single-channel grayscale"
        assert ref_img.shape == (dg.REFERENCE_SIZE_PX, dg.REFERENCE_SIZE_PX), (
            f"[{split}] {r['pair_id']} Reference shape {ref_img.shape} != "
            f"({dg.REFERENCE_SIZE_PX},{dg.REFERENCE_SIZE_PX})"
        )
        assert search_img.shape == (dg.REFERENCE_SIZE_PX, dg.REFERENCE_SIZE_PX), (
            f"[{split}] {r['pair_id']} Search shape {search_img.shape} != "
            f"({dg.REFERENCE_SIZE_PX},{dg.REFERENCE_SIZE_PX})"
        )
        assert ref_img.dtype == np.uint8 and search_img.dtype == np.uint8, (
            f"[{split}] {r['pair_id']} images must be uint8"
        )

        # A loose sanity bound only: rotation/scale-drift families move GT
        # through `transform_point`, so it can legitimately land a little
        # outside the pre-warp margin near the image edge - the tight
        # per-pixel correctness check below (GT box vs. Reference) is what
        # actually verifies GT is right, not this bound.
        assert -5.0 <= r["gt_x"] <= dg.REFERENCE_SIZE_PX + 5.0, (
            f"[{split}] {r['pair_id']} gt_x={r['gt_x']} outside plausible Search-image bounds"
        )
        assert -5.0 <= r["gt_y"] <= dg.REFERENCE_SIZE_PX + 5.0, (
            f"[{split}] {r['pair_id']} gt_y={r['gt_y']} outside plausible Search-image bounds"
        )

        # GT correctness + coordinate convention, re-checked against the
        # ACTUAL on-disk files (test_gt_safety.py checks a fresh in-memory
        # generation only).
        #
        # For families with nonzero rotation/scale drift, the comparison
        # patch must be rotated/scaled the same way before comparing: a
        # similarity transform (rotation + uniform scale, no shear) acts
        # identically everywhere in the plane, so a small crop far from the
        # image's global rotation center looks, LOCALLY, exactly like that
        # same crop rotated/scaled about its own center - only the
        # translation differs by position, and that part is already what
        # `transform_point` accounts for. Comparing against a plain
        # (unrotated) downsampled Reference would fail even at the
        # genuinely correct location, because fine periodic structure is
        # very sensitive to a few degrees of phase misalignment.
        #
        # A GT box near the image edge can extend past it (e.g. gt_y a few
        # px from 0 with scale > 1 widening the footprint) - clamping x0/y0
        # into bounds without also shifting the comparison patch by that
        # same clamp would silently compare two boxes centered on different
        # points, which is a validation-logic bug, not a GT bug (confirmed
        # by cross-checking against an edge-padded, unclamped extraction
        # during Phase 4 investigation). Padding the search image instead of
        # clamping keeps the box centered on the true GT in every case.
        box = dg.REFERENCE_SIZE_PX // dg.SCALE_FACTOR
        pad = box
        search_padded = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0 = int(round(r["gt_x"] - box / 2.0)) + pad
        y0 = int(round(r["gt_y"] - box / 2.0)) + pad
        search_patch = search_padded[y0:y0 + box, x0:x0 + box].astype(np.float64)
        expected = cv2.resize(ref_img.astype(np.float32), (box, box), interpolation=cv2.INTER_AREA)
        rotation_deg = r["degradation_params"].get("rotation_deg", 0.0)
        extra_scale = r["degradation_params"].get("extra_scale", 1.0)
        if rotation_deg != 0.0 or extra_scale != 1.0:
            local_matrix = cv2.getRotationMatrix2D((box / 2.0, box / 2.0), rotation_deg, extra_scale)
            expected = cv2.warpAffine(expected, local_matrix, (box, box), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        diff = float(np.mean(np.abs(search_patch - expected.astype(np.float64))))
        barrel_k = r["degradation_params"].get("barrel_k", 0.0)
        if diff >= 50.0 and barrel_k != 0.0:
            # Barrel/pincushion distortion is deliberately NOT analytically
            # corrected in ground truth (see dataset_generator.py's
            # ch_barrel_charging/ch_worst_case family comments) - its radial
            # displacement grows with distance from the image center, so a
            # GT point far from center can end up controllable-but-uncorrected
            # by up to roughly a couple dozen px. Confirm the claimed GT is
            # still the right BALLPARK location (a local search finds a much
            # better match within BARREL_SEARCH_RADIUS_PX) rather than
            # silently accepting an arbitrarily wrong one.
            BARREL_SEARCH_RADIUS_PX = 20
            best_diff, best_dx, best_dy = diff, 0, 0
            for dy in range(-BARREL_SEARCH_RADIUS_PX, BARREL_SEARCH_RADIUS_PX + 1):
                for dx in range(-BARREL_SEARCH_RADIUS_PX, BARREL_SEARCH_RADIUS_PX + 1):
                    xs, ys = x0 + dx, y0 + dy
                    patch = search_padded[ys:ys + box, xs:xs + box].astype(np.float64)
                    d = float(np.mean(np.abs(patch - expected.astype(np.float64))))
                    if d < best_diff:
                        best_diff, best_dx, best_dy = d, dx, dy
            assert best_diff < 50.0, (
                f"[{split}] {r['pair_id']} on-disk GT box does not match Reference even within "
                f"{BARREL_SEARCH_RADIUS_PX}px (best diff={best_diff:.2f})"
            )
            diff = best_diff
        assert diff < 50.0, f"[{split}] {r['pair_id']} on-disk GT box does not match Reference (diff={diff:.2f})"

    # Reproducibility: regenerate the first pair of this split from its
    # family definition and confirm it byte-matches the file on disk.
    first = records[0]
    family = next(f for f in dg.FAMILIES if f["name"] == first["structural_family"])
    pair_index = int(first["pair_id"].rsplit("_", 1)[-1])
    ref_regen, search_regen, meta_regen = dg.generate_pair(pair_index, first["seed"], family)
    ref_on_disk = cv2.imread(os.path.join(data_root, first["reference_path"]), cv2.IMREAD_UNCHANGED)
    search_on_disk = cv2.imread(os.path.join(data_root, first["search_path"]), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(ref_regen, ref_on_disk), f"[{split}] {first['pair_id']} Reference not reproducible"
    assert np.array_equal(search_regen, search_on_disk), f"[{split}] {first['pair_id']} Search not reproducible"
    assert meta_regen["gt_x"] == first["gt_x"] and meta_regen["gt_y"] == first["gt_y"], (
        f"[{split}] {first['pair_id']} GT not reproducible"
    )

    print(f"PASS [{split}]: {len(records)} pairs validated (dimensions, grayscale, GT, metadata, reproducibility)")


def validate_dataset(data_root: str, splits: list[str] | None = None) -> None:
    if splits is None:
        splits = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
            and os.path.isfile(os.path.join(data_root, d, "ground_truth.json"))
            # cross_generator is external (a different generator's output, imported as data
            # files only) and, by design, does not share this project's metadata schema -
            # see reports/DATASET_AUDIT.md section 3. It is validated separately, via
            # evaluation/evaluate.py's own manifest loader, not this internal-schema check.
            and d != "cross_generator"
        )
    assert splits, f"No splits with a ground_truth.json found under {data_root}"
    for split in splits:
        validate_split(data_root, split)
    print(f"\nAll {len(splits)} split(s) passed dataset validation: {splits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an already-generated V2 dataset directory.")
    parser.add_argument("data_root", nargs="?", default="data")
    parser.add_argument("--splits", default=None, help="Comma-separated subset (default: every split present).")
    args = parser.parse_args()
    validate_dataset(args.data_root, args.splits.split(",") if args.splits else None)
