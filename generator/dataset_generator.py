"""Orchestrates one Reference/Search pair (or a full dataset split) from the
macro layout + degradation modules.

Ground-truth-by-construction pipeline, in order:
  1. Build one shared fine canvas (mats + strips) from a layout seed.
  2. Pick a crop origin (x0, y0) according to the family's crop mode.
  3. Compute ground truth from (x0, y0) BEFORE any imaging call.
  4. Crop -> image_reference() (mild degradation).
  5. Whole canvas -> image_search() (full degradation pipeline, exact 10x
     downsample + optional acquisition-stage drift).
  6. Write both PNGs + full metadata.

Every pair's RNG is `default_rng([seed, family_salt(split, family_name),
pair_index])` - fully determined by (seed, split, family_name, pair_index)
alone, never by how much randomness earlier pairs consumed, so regenerating
pair #47 never requires regenerating #0..46 first. The split/family name is
mixed into the seed (not just pair_index) specifically so that two families
- even in different splits - never draw the same underlying macro canvas
merely because they share a pair_index and neither samples extra
rotation/scale randomness; see reports/DATASET_AUDIT.md section 2 for the
cross-split leakage this fixes (generator_version bumped to 2.1.0 here to
mark the RNG scheme change - datasets generated under 2.0.0 are not
reproducible under this code and must be regenerated, not compared byte-wise).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import zlib
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import degradation_models
from . import macro_layout
from . import mat_generator
from . import metadata as metadata_mod

GENERATOR_VERSION = "driftsensev2.1.0"

REFERENCE_SIZE_PX = 1000
SCALE_FACTOR = 10
FINE_CANVAS_SIZE_PX = REFERENCE_SIZE_PX * SCALE_FACTOR  # 10000

MAX_CROP_TRIES = 60
SINGLE_MAT_MARGIN_PX = 60
STRIP_CENTER_JITTER_PX = 150

DEFAULT_PARAMS: dict = {
    # Macro layout / structure
    "mat_size_nm": 2400,
    "strip_width_nm": 300,
    "force_preset": None,
    "feature_size_scale": 1.0,
    "linewidth_bias_nm": 0.0,
    "collapse_enabled": True,
    "collapse_threshold_nm": 10.0,
    "collapse_prob": 0.65,
    "corner_rounding_px": 0.0,
    # Optics
    "blur_sigma_ref_px": 0.6,
    "blur_search_effective_px": 1.0,
    "astigmatism_ratio": 1.0,
    # Shot / read noise
    "dose_reference": 1800.0,
    "dose_search": 220.0,
    "read_noise_sigma_ref": 2.0,
    "read_noise_sigma_search": 6.0,
    # Scan drift
    "shear_amplitude_px": 1.0,
    "jitter_std_px": 0.4,
    # Acquisition-stage residual drift (post-downsample), off unless a family samples it
    "rotation_deg": 0.0,
    "extra_scale": 1.0,
    # Optional degradations, OFF by default (see reports/DEGRADATION_COVERAGE.md)
    "vignette_strength": 0.0,
    "gamma": 1.0,
    "barrel_k": 0.0,
    "speckle_sigma": 0.0,
    "salt_pepper_amount": 0.0,
    "charging_prob": 0.0,
    "charging_intensity": 0.0,
}

# Structural family = a named difficulty/condition category (crop mode +
# degradation overrides). Not to be confused with an "acquisition variant"
# (see generate_acquisition_variant_set below) - see
# reports/V2_ARCHITECTURE_PLAN.md section 12 for why these are kept as distinct terms.
FAMILIES: list[dict] = [
    dict(name="dev_strip_anchor", split="development", crop_mode="strip_center", n=8, overrides={},
         description="Crop centered on a strip - a reliable, non-periodic landmark."),
    dict(name="dev_single_mat", split="development", crop_mode="single_mat", n=8, overrides={},
         description="Deep inside one mat, no boundary - baseline periodic case."),
    dict(name="dev_dense_periodic", split="development", crop_mode="single_mat", n=8,
         overrides={"force_preset": "mat_dense"},
         description="Densest pitch preset, deep in-mat - maximum local ambiguity."),

    dict(name="val_mat_boundary", split="validation", crop_mode="mat_boundary", n=10, overrides={},
         description="Straddles 2+ mats, any presets."),
    dict(name="val_same_preset_boundary", split="validation", crop_mode="same_preset_boundary", n=10,
         overrides={}, description="Straddles 2+ mats sharing the same preset - hardest boundary case."),
    dict(name="val_multi_mat", split="validation", crop_mode="multi_mat", n=10, overrides={},
         description="Touches 3+ mats near a macro grid corner."),
    dict(name="val_linewidth_bias", split="validation", crop_mode="random", n=10,
         overrides={"linewidth_bias_nm": 4.0},
         description="Deterministic global CD/etch bias exercised (see reports/DEGRADATION_COVERAGE.md)."),

    dict(name="ho_heavy_noise", split="held_out", crop_mode="random", n=10,
         overrides={"dose_search": 40.0, "read_noise_sigma_search": 10.0},
         description="Pure acquisition-noise stress."),
    dict(name="ho_rotation_drift", split="held_out", crop_mode="random", n=10,
         overrides={"_rotation_range": (-4.0, 4.0)},
         description="Residual stage-rotation drift on top of the exact 10x base."),
    dict(name="ho_scale_drift", split="held_out", crop_mode="random", n=10,
         # (0.90, 1.10) = literal 9:1-11:1 given the exact 10x base ratio -
         # the Applied Materials help doc/pptx's stated "robustness tests
         # may span" range. Widened from (0.93, 1.07) 2026-08-15 (was ~+-7%,
         # 2pp short of the stated edge) - experiments/scale_range_v1/,
         # integrated as a documented gate exception, see
         # reports/GATE_EXCEPTIONS.md.
         overrides={"_scale_range": (0.90, 1.10)},
         description="Residual magnification-calibration drift on top of the exact 10x base."),
    dict(name="ho_vignette_gamma", split="held_out", crop_mode="random", n=10,
         overrides={"vignette_strength": 0.35, "gamma": 1.3},
         description="Radiometric falloff + detector-gain nonlinearity exercised."),

    dict(name="ch_combined_acquisition", split="challenge", crop_mode="random", n=8,
         # _scale_range widened to the literal 9:1-11:1 edge - see
         # ho_scale_drift's comment above for the full rationale.
         overrides={"dose_search": 60.0, "_rotation_range": (-4.0, 4.0), "_scale_range": (0.90, 1.10)},
         description="Rotation + scale drift + noise combined."),
    dict(name="ch_barrel_charging", split="challenge", crop_mode="random", n=8,
         # barrel_k deliberately kept very small: unlike rotation/scale,
         # barrel distortion displaces points by an amount that depends on
         # the point's own (unknown-until-corrected) radius, which is not
         # analytically inverted here (see degradation_models.transform_point
         # docstring - that only covers rotation/scale). At this k, the
         # induced GT displacement stays a bounded, single-digit-to-low-teens
         # px even for a GT point near a corner (the worst case, farthest
         # from the distortion center) - re-measured directly against on-disk
         # GT-box-vs-Reference matches during the Phase 4 reseeding fix
         # (generator/test_dataset_validation.py's barrel-aware local search
         # enforces this bound at dataset-validation time, not just here).
         # Smaller than the >10px tolerance bucket the benchmark treats as a
         # real failure. This is a documented simplification, not an
         # oversight: see reports/DEGRADATION_COVERAGE.md for why an
         # analytic inverse wasn't built.
         overrides={"barrel_k": 0.003, "charging_prob": 0.015, "charging_intensity": 60.0},
         description="Scan-linearity (barrel) distortion + sample-charging streaks."),
    dict(name="ch_speckle_saltpepper", split="challenge", crop_mode="random", n=8,
         overrides={"speckle_sigma": 0.12, "salt_pepper_amount": 0.01},
         description="Detector-gain speckle + impulse (salt-and-pepper) noise."),
    dict(name="ch_worst_case", split="challenge", crop_mode="same_preset_boundary", n=8,
         # _scale_range widened to the literal 9:1-11:1 edge - see
         # ho_scale_drift's comment above for the full rationale. This is
         # the family experiments/scale_range_v1/ and
         # experiments/multiway_tiebreak_v1/ both showed the clearest
         # measurable gains on (ch_worst_case_006's 118.5px -> 4.6px rescue
         # in particular).
         overrides={"force_preset": "mat_dense", "mat_size_nm": 3200, "dose_search": 45.0,
                    "_rotation_range": (-4.0, 4.0), "_scale_range": (0.90, 1.10),
                    "barrel_k": 0.002, "speckle_sigma": 0.08},
         description="Every hard axis combined on the hardest boundary mode."),
]

ACQUISITION_VARIANTS: dict[str, dict] = {
    "clean": {},
    "low_dose": {"dose_search": 35.0},
    "heavy_drift": {"shear_amplitude_px": 2.5, "jitter_std_px": 1.0},
    "speckle_salt_pepper": {"speckle_sigma": 0.15, "salt_pepper_amount": 0.015},
    "charging": {"charging_prob": 0.02, "charging_intensity": 70.0},
}


def _random_xy(rng: np.random.Generator, size: int) -> tuple[int, int]:
    hi = FINE_CANVAS_SIZE_PX - size
    return int(rng.integers(0, hi + 1)), int(rng.integers(0, hi + 1))


def pick_crop_origin(mode: str, layout: macro_layout.MacroLayout, rng: np.random.Generator,
                      size: int = REFERENCE_SIZE_PX) -> tuple[int, int, bool, int]:
    """Returns (x0, y0, fallback_used, tries). Non-"random" modes retry up
    to MAX_CROP_TRIES against a structural predicate, then fall back to a
    uniform-random crop. The fallback is COUNTED and returned so it ends up
    in metadata rather than silently diluting a family's intended difficulty
    (see reports/V2_ARCHITECTURE_PLAN.md section 4).
    """
    if mode == "random":
        x0, y0 = _random_xy(rng, size)
        return x0, y0, False, 1

    if mode == "single_mat":
        m = SINGLE_MAT_MARGIN_PX
        candidates = [r for r in layout.mat_rects if r["w"] >= size + 2 * m and r["h"] >= size + 2 * m]
        for attempt in range(1, MAX_CROP_TRIES + 1):
            if candidates:
                r = candidates[int(rng.integers(0, len(candidates)))]
                x0 = int(rng.integers(r["x"] + m, r["x"] + r["w"] - size - m + 1))
                y0 = int(rng.integers(r["y"] + m, r["y"] + r["h"] - size - m + 1))
                mem = macro_layout.crop_membership(x0, y0, size, layout)
                if mem["num_mats"] == 1 and not mem["crosses_strip_boundary"]:
                    return x0, y0, False, attempt
        x0, y0 = _random_xy(rng, size)
        return x0, y0, True, MAX_CROP_TRIES

    if mode == "strip_center":
        for attempt in range(1, MAX_CROP_TRIES + 1):
            if layout.strip_rects:
                s = layout.strip_rects[int(rng.integers(0, len(layout.strip_rects)))]
                cx, cy = s["x"] + s["w"] / 2.0, s["y"] + s["h"] / 2.0
                jx = int(rng.integers(-STRIP_CENTER_JITTER_PX, STRIP_CENTER_JITTER_PX + 1))
                jy = int(rng.integers(-STRIP_CENTER_JITTER_PX, STRIP_CENTER_JITTER_PX + 1))
                x0 = int(np.clip(cx - size / 2 + jx, 0, FINE_CANVAS_SIZE_PX - size))
                y0 = int(np.clip(cy - size / 2 + jy, 0, FINE_CANVAS_SIZE_PX - size))
                mem = macro_layout.crop_membership(x0, y0, size, layout)
                if mem["crosses_strip_boundary"]:
                    return x0, y0, False, attempt
        x0, y0 = _random_xy(rng, size)
        return x0, y0, True, MAX_CROP_TRIES

    if mode in ("mat_boundary", "same_preset_boundary", "multi_mat"):
        need_same_preset = mode == "same_preset_boundary"
        need_count = 3 if mode == "multi_mat" else 2
        for attempt in range(1, MAX_CROP_TRIES + 1):
            x0, y0 = _random_xy(rng, size)
            mem = macro_layout.crop_membership(x0, y0, size, layout)
            if mem["num_mats"] >= need_count and (not need_same_preset or mem["same_preset_boundary"]):
                return x0, y0, False, attempt
        x0, y0 = _random_xy(rng, size)
        return x0, y0, True, MAX_CROP_TRIES

    raise ValueError(f"Unknown crop mode '{mode}'")


def _resolve_params(base_overrides: dict, rng: np.random.Generator) -> dict:
    params = dict(DEFAULT_PARAMS)
    overrides = dict(base_overrides)
    rotation_range = overrides.pop("_rotation_range", None)
    scale_range = overrides.pop("_scale_range", None)
    params.update(overrides)
    if rotation_range is not None:
        params["rotation_deg"] = float(rng.uniform(*rotation_range))
    if scale_range is not None:
        params["extra_scale"] = float(rng.uniform(*scale_range))
    return params


def _build_layout(params: dict, rng: np.random.Generator) -> macro_layout.MacroLayout:
    layout_seed = int(rng.integers(0, 2 ** 31 - 1))
    return macro_layout.generate_macro_canvas(
        layout_seed,
        canvas_size_nm=FINE_CANVAS_SIZE_PX,
        mat_size_nm=params["mat_size_nm"],
        strip_width_nm=params["strip_width_nm"],
        force_preset=params["force_preset"],
        feature_size_scale=params["feature_size_scale"],
        linewidth_bias_nm=params["linewidth_bias_nm"],
        collapse_enabled=params["collapse_enabled"],
        collapse_threshold_nm=params["collapse_threshold_nm"],
        collapse_prob=params["collapse_prob"],
        corner_rounding_px=params["corner_rounding_px"],
    )


def _gt_from_origin(x0: int, y0: int) -> tuple[float, float]:
    half_footprint = (REFERENCE_SIZE_PX / SCALE_FACTOR) / 2.0
    return x0 / SCALE_FACTOR + half_footprint, y0 / SCALE_FACTOR + half_footprint


def _touched_word_pitches(membership: dict, force_preset: Optional[str],
                           feature_size_scale: float = 1.0) -> list[float]:
    """As-rendered word pitch (nominal preset pitch x feature_size_scale) -
    periodicity_score must reflect the pitch actually drawn, not the
    unscaled preset table, or a feature_size_scale != 1.0 pair would report
    a structurally-wrong periodicity/difficulty score."""
    if membership["presets"]:
        return [mat_generator.DRAM_MAT_PRESETS[p]["word_pitch_nm"] * feature_size_scale
                for p in membership["presets"]]
    if force_preset:
        return [mat_generator.DRAM_MAT_PRESETS[force_preset]["word_pitch_nm"] * feature_size_scale]
    return []


def _family_salt(split: str, family_name: str) -> int:
    """Stable (not process-randomized - `zlib.crc32`, never Python's
    built-in `hash()`) integer identity for one (split, family_name) pair,
    mixed into the per-pair RNG seed below so two families - even in two
    different splits - never draw the same underlying macro canvas just
    because they happen to share a pair_index and neither one samples extra
    rotation/scale randomness. See reports/DATASET_AUDIT.md section 2."""
    return zlib.crc32(f"{split}:{family_name}".encode("utf-8"))


def _pair_rng(seed: int, split: str, family_name: str, pair_index: int) -> np.random.Generator:
    return np.random.default_rng([seed, _family_salt(split, family_name), pair_index])


def generate_pair(pair_index: int, seed: int, family: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate one (Reference, Search, metadata) tuple. Deterministic given
    (seed, pair_index, family) alone - including family["split"] and
    family["name"], which are folded into the RNG seed (see _pair_rng)."""
    rng = _pair_rng(seed, family["split"], family["name"], pair_index)
    params = _resolve_params(family["overrides"], rng)
    layout = _build_layout(params, rng)

    x0, y0, fallback_used, tries = pick_crop_origin(family["crop_mode"], layout, rng, REFERENCE_SIZE_PX)
    gt_x_pre_warp, gt_y_pre_warp = _gt_from_origin(x0, y0)

    crop = layout.canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]
    reference_img = degradation_models.image_reference(crop, rng, params)

    search_params = dict(params)
    search_params["blur_sigma_search_fine_px"] = params["blur_search_effective_px"] * SCALE_FACTOR
    search_params["scale_factor"] = SCALE_FACTOR
    search_img = degradation_models.image_search(layout.canvas, rng, search_params)

    # Families that sample nonzero rotation/scale drift move every point in
    # the Search image relative to the pre-warp downsampled frame - ground
    # truth must move with it (see degradation_models.transform_point).
    gt_x, gt_y = degradation_models.transform_point(
        gt_x_pre_warp, gt_y_pre_warp, params.get("rotation_deg", 0.0), params.get("extra_scale", 1.0),
        REFERENCE_SIZE_PX,
    )

    membership = macro_layout.crop_membership(x0, y0, REFERENCE_SIZE_PX, layout)
    pitches = _touched_word_pitches(membership, params["force_preset"], params["feature_size_scale"])

    pair_id = f"{family['name']}_{pair_index:03d}"
    meta = metadata_mod.build_metadata(
        pair_id=pair_id, seed=seed, generator_version=GENERATOR_VERSION,
        reference_path=f"{family['split']}/{pair_id}_reference.png",
        search_path=f"{family['split']}/{pair_id}_search.png",
        gt_x=gt_x, gt_y=gt_y, membership=membership,
        family_name=family["name"], split=family["split"], crop_mode=family["crop_mode"],
        fallback_used=fallback_used, crop_tries=tries,
        degradation_params=params, mat_size_nm=params["mat_size_nm"],
        strip_width_nm=params["strip_width_nm"], preset_word_pitches=pitches,
    )
    return reference_img, search_img, meta


def _write_manifest(split_dir: str, records: list[dict]) -> None:
    with open(os.path.join(split_dir, "ground_truth.json"), "w") as f:
        json.dump(records, f, indent=2)
    flat = []
    for r in records:
        row = copy.deepcopy(r)
        row["mat_ids"] = ";".join(str(m) for m in row["mat_ids"])
        row["presets"] = ";".join(row["presets"])
        row["degradation_params"] = json.dumps(row["degradation_params"])
        flat.append(row)
    pd.DataFrame(flat).to_csv(os.path.join(split_dir, "ground_truth.csv"), index=False)


def generate_dataset(output_root: str, seed: int = 777001, families: Optional[list[dict]] = None,
                      only_splits: Optional[list[str]] = None, verbose: bool = True) -> dict[str, list[dict]]:
    """Generate every pair in `families` (default: the full FAMILIES table),
    optionally restricted to a subset of splits, writing PNGs + manifests
    under output_root/<split>/.
    """
    families = families if families is not None else FAMILIES
    records_by_split: dict[str, list[dict]] = {}
    for fam in families:
        if only_splits is not None and fam["split"] not in only_splits:
            continue
        split_dir = os.path.join(output_root, fam["split"])
        os.makedirs(split_dir, exist_ok=True)
        for i in range(fam["n"]):
            ref_img, search_img, meta = generate_pair(i, seed, fam)
            cv2.imwrite(os.path.join(output_root, meta["reference_path"]), ref_img)
            cv2.imwrite(os.path.join(output_root, meta["search_path"]), search_img)
            records_by_split.setdefault(fam["split"], []).append(meta)
            if verbose:
                print(f"  wrote {meta['pair_id']} (split={fam['split']}, fallback={meta['crop_mode_fallback_used']})")
    for split, records in records_by_split.items():
        _write_manifest(os.path.join(output_root, split), records)
        if verbose:
            print(f"[{split}] {len(records)} pairs -> {output_root}/{split}/ground_truth.{{json,csv}}")
    return records_by_split


def generate_acquisition_variant_set(scene_index: int, seed: int, base_family: dict
                                      ) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """One Reference + one Search image per named acquisition variant, all
    sharing the same macro canvas and crop location (only the Search-side
    acquisition parameters differ between variants). This is the
    "1 Reference, N re-acquisitions" concept from the reference generator
    (an "acquisition variant" - see FAMILIES docstring for why this is a
    different concept from a "structural family") - see
    reports/V2_ARCHITECTURE_PLAN.md section 2.
    """
    rng = _pair_rng(seed, base_family["split"], base_family["name"], scene_index)
    params = _resolve_params(base_family["overrides"], rng)
    layout = _build_layout(params, rng)
    x0, y0, fallback_used, tries = pick_crop_origin(base_family["crop_mode"], layout, rng, REFERENCE_SIZE_PX)
    gt_x_pre_warp, gt_y_pre_warp = _gt_from_origin(x0, y0)
    gt_x, gt_y = degradation_models.transform_point(
        gt_x_pre_warp, gt_y_pre_warp, params.get("rotation_deg", 0.0), params.get("extra_scale", 1.0),
        REFERENCE_SIZE_PX,
    )

    crop = layout.canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]
    reference_img = degradation_models.image_reference(crop, rng, params)

    searches: dict[str, np.ndarray] = {}
    for variant_idx, (name, overrides) in enumerate(ACQUISITION_VARIANTS.items()):
        vp = dict(params)
        vp.update(overrides)
        vp["blur_sigma_search_fine_px"] = vp["blur_search_effective_px"] * SCALE_FACTOR
        vp["scale_factor"] = SCALE_FACTOR
        # Deterministic per-variant RNG stream, offset by index (never by
        # Python's randomized built-in hash()) so variant order changes
        # never change any other variant's pixels.
        v_rng = np.random.default_rng(
            [seed, _family_salt(base_family["split"], base_family["name"]), scene_index * 97 + 500_000 + variant_idx]
        )
        searches[name] = degradation_models.image_search(layout.canvas, v_rng, vp)

    membership = macro_layout.crop_membership(x0, y0, REFERENCE_SIZE_PX, layout)
    meta = {
        "scene_id": f"scene_{scene_index:03d}",
        "seed": seed,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "crop_mode": base_family["crop_mode"],
        "crop_mode_fallback_used": fallback_used,
        "crop_mode_tries": tries,
        "membership": membership,
        "variant_names": list(ACQUISITION_VARIANTS.keys()),
        "variant_overrides": ACQUISITION_VARIANTS,
    }
    return reference_img, searches, meta


def generate_acquisition_variant_demo(output_root: str, seed: int = 777001, n_scenes: int = 3) -> None:
    base_family = dict(name="acquisition_variant_demo", split="development", crop_mode="single_mat", overrides={})
    out_dir = os.path.join(output_root, "development", "acquisition_variants")
    os.makedirs(out_dir, exist_ok=True)
    scenes_meta = []
    for scene_index in range(n_scenes):
        reference_img, searches, meta = generate_acquisition_variant_set(scene_index, seed, base_family)
        cv2.imwrite(os.path.join(out_dir, f"{meta['scene_id']}_reference.png"), reference_img)
        for name, img in searches.items():
            cv2.imwrite(os.path.join(out_dir, f"{meta['scene_id']}_search_{name}.png"), img)
        scenes_meta.append(meta)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(scenes_meta, f, indent=2)
    print(f"[acquisition_variants] {n_scenes} scenes x {len(ACQUISITION_VARIANTS)} variants -> {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the DriftSense V2 dataset.")
    parser.add_argument("--out", default="data", help="Output root directory.")
    parser.add_argument("--seed", type=int, default=777001)
    parser.add_argument("--splits", default=None,
                         help="Comma-separated subset of splits to generate (default: all).")
    parser.add_argument("--with-acquisition-variants", action="store_true",
                         help="Also generate the bonus acquisition-variant demo set.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only_splits = args.splits.split(",") if args.splits else None
    generate_dataset(args.out, seed=args.seed, only_splits=only_splits)
    if args.with_acquisition_variants:
        generate_acquisition_variant_demo(args.out, seed=args.seed)
