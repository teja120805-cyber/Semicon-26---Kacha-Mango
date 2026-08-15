"""GT-safety verification.

Two different claims, both checked, because neither implies the other:

1. Static: the generator is structurally incapable of leaking ground truth
   forward (no rendering function even accepts a GT-shaped parameter, and
   no rendering module imports the localization pipeline).
2. Dynamic: the ground truth it computes is actually correct - not just
   internally consistent, but independently re-derivable by cropping the
   Search image at the claimed GT box and confirming it resembles a
   downsampled Reference crop. This pattern is modeled on (independently
   reimplemented from, not copied from) a genuinely good idea found in the
   reference generator's own test suite during Phase 0 research.

Run directly (`python -m generator.test_gt_safety`) or via pytest.
"""
from __future__ import annotations

import ast
import inspect

import cv2
import numpy as np

from . import dataset_generator, degradation_models, macro_layout, mat_generator, pattern_renderer

RENDERING_MODULES = [pattern_renderer, mat_generator, macro_layout, degradation_models]
FORBIDDEN_PARAM_SUBSTRINGS = ("gt_x", "gt_y", "ground_truth", "target_x", "target_y", "answer_x", "answer_y")


def test_no_gt_shaped_parameters():
    for mod in RENDERING_MODULES:
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arg_names = [a.arg for a in (node.args.args + node.args.kwonlyargs)]
                for name in arg_names:
                    lowered = name.lower()
                    for bad in FORBIDDEN_PARAM_SUBSTRINGS:
                        assert bad not in lowered, (
                            f"{mod.__name__}.{node.name} has a GT-shaped parameter '{name}'"
                        )


def test_no_pipeline_import():
    for mod in RENDERING_MODULES:
        source = inspect.getsource(mod)
        assert "import pipeline" not in source and "from pipeline" not in source, (
            f"{mod.__name__} must not import the localization pipeline"
        )


def test_ground_truth_matches_reference_patch():
    family = dict(name="gt_safety_check", split="development", crop_mode="random", overrides={})
    ref_img, search_img, meta = dataset_generator.generate_pair(0, seed=123456, family=family)

    box = dataset_generator.REFERENCE_SIZE_PX // dataset_generator.SCALE_FACTOR  # 100
    # Pad rather than clamp, so a GT box a few px from the image edge stays
    # centered on the true GT instead of being silently shifted (see
    # test_ground_truth_correct_under_rotation_scale_drift and
    # test_dataset_validation.py for the edge-clamping bug this avoids).
    pad = box
    search_padded = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0 = int(round(meta["gt_x"] - box / 2.0)) + pad
    y0 = int(round(meta["gt_y"] - box / 2.0)) + pad
    search_patch = search_padded[y0:y0 + box, x0:x0 + box].astype(np.float64)

    expected = cv2.resize(ref_img.astype(np.float32), (box, box), interpolation=cv2.INTER_AREA).astype(np.float64)

    assert search_patch.shape == expected.shape == (box, box)
    diff = float(np.mean(np.abs(search_patch - expected)))
    assert diff < 50.0, f"GT box does not resemble a downsampled Reference crop (mean abs diff={diff:.2f})"


def test_ground_truth_correct_under_rotation_scale_drift():
    """Regression test for a real bug found during this generator's own
    Phase 0 validation: applying rotation/scale drift to the Search image
    without moving ground truth through the same transform left GT wrong by
    tens of pixels for any crop away from the image center. Exercises a
    family that actually samples nonzero rotation/scale (unlike the other
    GT-safety tests here, which use rotation=0/scale=1 families and would
    not have caught this)."""
    family = dict(name="rotation_gt_check", split="development", crop_mode="random",
                  overrides={"_rotation_range": (-4.0, 4.0), "_scale_range": (0.93, 1.07)})
    ref_img, search_img, meta = dataset_generator.generate_pair(0, seed=555, family=family)
    assert meta["degradation_params"]["rotation_deg"] != 0.0 or meta["degradation_params"]["extra_scale"] != 1.0

    box = dataset_generator.REFERENCE_SIZE_PX // dataset_generator.SCALE_FACTOR
    pad = box
    search_padded = cv2.copyMakeBorder(search_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0 = int(round(meta["gt_x"] - box / 2.0)) + pad
    y0 = int(round(meta["gt_y"] - box / 2.0)) + pad
    search_patch = search_padded[y0:y0 + box, x0:x0 + box].astype(np.float64)
    expected = cv2.resize(ref_img.astype(np.float32), (box, box), interpolation=cv2.INTER_AREA)

    # A similarity transform (rotation + uniform scale) acts identically
    # everywhere in the plane, so the comparison patch must be rotated the
    # same way about its OWN center - see the long comment in
    # test_dataset_validation.py for why comparing against a plain
    # downsampled Reference would fail even at the correct location.
    rotation_deg = meta["degradation_params"]["rotation_deg"]
    extra_scale = meta["degradation_params"]["extra_scale"]
    local_matrix = cv2.getRotationMatrix2D((box / 2.0, box / 2.0), rotation_deg, extra_scale)
    expected = cv2.warpAffine(expected, local_matrix, (box, box), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float64)

    diff = float(np.mean(np.abs(search_patch - expected)))
    assert diff < 50.0, f"GT under rotation/scale drift does not match Reference (mean abs diff={diff:.2f})"


def test_seed_reproducibility():
    family = dict(name="repro_check", split="development", crop_mode="single_mat", overrides={})
    r1, s1, m1 = dataset_generator.generate_pair(0, seed=42, family=family)
    r2, s2, m2 = dataset_generator.generate_pair(0, seed=42, family=family)
    assert np.array_equal(r1, r2), "Reference image not reproducible from the same seed"
    assert np.array_equal(s1, s2), "Search image not reproducible from the same seed"
    assert m1["gt_x"] == m2["gt_x"] and m1["gt_y"] == m2["gt_y"], "GT not reproducible from the same seed"


def test_pair_index_independence():
    """Regenerating only pair #3 (without generating #0-2 first) must give
    the exact same result as generating #0..3 in sequence and taking #3."""
    family = dict(name="index_independence_check", split="development", crop_mode="random", overrides={})
    _, _, direct = dataset_generator.generate_pair(3, seed=999, family=family)
    for i in range(3):
        dataset_generator.generate_pair(i, seed=999, family=family)
    _, _, after_others = dataset_generator.generate_pair(3, seed=999, family=family)
    assert direct["gt_x"] == after_others["gt_x"] and direct["gt_y"] == after_others["gt_y"]


def test_feature_size_scale_changes_the_image_and_periodicity_score():
    """`feature_size_scale` (Phase 6 addition) must actually change the
    rendered pitch, not just be a decorative parameter - and the derived
    periodicity_score must track the as-rendered pitch, not the unscaled
    preset table (see dataset_generator._touched_word_pitches)."""
    base = dict(name="fs_scale_check", split="development", crop_mode="single_mat",
                overrides={"force_preset": "mat_nominal"})
    scaled = dict(name="fs_scale_check", split="development", crop_mode="single_mat",
                  overrides={"force_preset": "mat_nominal", "feature_size_scale": 1.6})
    ref1, _, meta1 = dataset_generator.generate_pair(0, seed=42, family=base)
    ref2, _, meta2 = dataset_generator.generate_pair(0, seed=42, family=scaled)

    diff = float(np.mean(np.abs(ref1.astype(np.float64) - ref2.astype(np.float64))))
    assert diff > 10.0, f"feature_size_scale did not visibly change the rendered mat (diff={diff:.2f})"
    assert meta2["periodicity_score"] < meta1["periodicity_score"], (
        "periodicity_score did not decrease for a larger (less periodic) scaled pitch"
    )
    assert (meta1["gt_x"], meta1["gt_y"]) == (meta2["gt_x"], meta2["gt_y"]), (
        "feature_size_scale must not perturb ground truth (it only affects rendering)"
    )


def test_all_six_dram_presets_produce_distinct_geometry():
    """Every preset must be a genuinely different structure, not a label
    over identical geometry - each pairwise comparison (15 pairs across 6
    presets) must differ visibly at the same crop location/seed."""
    presets = list(mat_generator.PRESET_NAMES)
    assert len(presets) == 6, f"expected 6 DRAM presets, found {len(presets)}"
    images = {}
    for preset in presets:
        family = dict(name=f"preset_check_{preset}", split="development", crop_mode="single_mat",
                      overrides={"force_preset": preset})
        ref_img, _, _ = dataset_generator.generate_pair(0, seed=7, family=family)
        images[preset] = ref_img.astype(np.float64)
    for i, a in enumerate(presets):
        for b in presets[i + 1:]:
            diff = float(np.mean(np.abs(images[a] - images[b])))
            assert diff > 5.0, f"presets '{a}' and '{b}' produced near-identical geometry (diff={diff:.2f})"


def test_ground_truth_is_always_finite():
    """No NaN/inf ground truth, across a sample of families and rotation/
    scale-drift conditions - a NaN GT would silently corrupt every metric
    downstream without ever raising an error on its own."""
    samples = [
        dict(name="finite_check_plain", split="development", crop_mode="random", overrides={}),
        dict(name="finite_check_drift", split="development", crop_mode="random",
             overrides={"_rotation_range": (-5.0, 5.0), "_scale_range": (0.92, 1.08)}),
        dict(name="finite_check_barrel", split="development", crop_mode="random",
             overrides={"barrel_k": 0.01}),
    ]
    for family in samples:
        for pair_index in range(3):
            _, _, meta = dataset_generator.generate_pair(pair_index, seed=2024, family=family)
            assert np.isfinite(meta["gt_x"]) and np.isfinite(meta["gt_y"]), (
                f"non-finite ground truth for family '{family['name']}' pair {pair_index}: "
                f"({meta['gt_x']}, {meta['gt_y']})"
            )


def test_every_optional_degradation_is_exercised():
    """Regression guard against a specific bug class: a degradation
    parameter that is wired end-to-end but never set to a non-default
    value by any shipped family, silently making it dead weight."""
    optional_keys = [
        "linewidth_bias_nm", "rotation_deg", "extra_scale", "vignette_strength",
        "gamma", "barrel_k", "speckle_sigma", "salt_pepper_amount", "charging_prob",
    ]
    exercised = set()
    for fam in dataset_generator.FAMILIES:
        for key, value in fam["overrides"].items():
            key = key.lstrip("_").replace("rotation_range", "rotation_deg").replace("scale_range", "extra_scale")
            exercised.add(key)
    missing = [k for k in optional_keys if k not in exercised]
    assert not missing, f"Degradation parameter(s) never exercised by any family: {missing}"


def run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS: {name}")
    print(f"\n{len(tests)} GT-safety tests passed.")


if __name__ == "__main__":
    run_all()
