"""Null-control verification for P3. Two independent checks.

Check A - implementation correctness. Weighted ZNCC at uniform weights is
standard ZNCC algebraically, so the dense and point forms must agree with
OpenCV's TM_CCOEFF_NORMED (which pipeline/matching.py uses) to float32
accumulation noise. This checks the NEW code is right; it does not by itself
make the null exact, which is why check B exists separately.

Check B - null control. The harness at alpha=0 and at tie_eps=0 must
reproduce pipeline.localize.localize bit-for-bit, per pair, on x, y,
confidence and ambiguity_ratio - not merely on error@5px, which would hide
sub-pixel drift.

Run:  python -m experiments.discriminability_weighted.verify_null
"""
from __future__ import annotations

import cv2
import numpy as np

from evaluation.evaluate import load_manifest
from pipeline import matching
from pipeline.localize import localize

from .harness import localize_weighted
from .weighted_zncc import uniform_weights, weighted_zncc_map, weighted_zncc_point


def check_a(data_root: str = "data", n_pairs: int = 4) -> dict:
    manifest = load_manifest(data_root, "development").iloc[:n_pairs]
    dense_max, point_max = 0.0, 0.0
    argmax_mismatches = 0
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        for scale, rot, psf in ((10.0, 0.0, 0.0), (9.4, 2.5, 1.6), (10.8, -3.75, 0.0)):
            tmpl = matching.build_template(ref, scale, rot, psf)
            w = uniform_weights(tmpl.shape)
            prod = matching.correlate(search, tmpl)
            mine = weighted_zncc_map(search, tmpl, w)
            dense_max = max(dense_max, float(np.abs(prod - mine).max()))
            if int(prod.argmax()) != int(mine.argmax()):
                argmax_mismatches += 1
            ys, xs = np.unravel_index(int(prod.argmax()), prod.shape)
            patch = search[ys:ys + tmpl.shape[0], xs:xs + tmpl.shape[1]]
            point_max = max(point_max, abs(weighted_zncc_point(tmpl, patch, w) - float(prod[ys, xs])))
    return {"dense_max_abs_diff": dense_max, "point_max_abs_diff": point_max,
            "argmax_mismatches": argmax_mismatches, "n_templates": len(manifest) * 3}


def check_b(data_root: str = "data", split: str = "development") -> dict:
    manifest = load_manifest(data_root, split)
    rows, mismatches = [], 0
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        base = localize(ref, search)
        # Both null settings, independently.
        for label, kwargs in (("alpha0", dict(alpha=0.0, tie_eps=0.05)),
                               ("tieeps0", dict(alpha=1.0, tie_eps=0.0))):
            cand = localize_weighted(ref, search, scheme="lattice_shift", **kwargs)
            identical = (cand.x == base.x and cand.y == base.y
                         and cand.confidence == base.confidence
                         and cand.ambiguity_ratio == base.ambiguity_ratio)
            if not identical:
                mismatches += 1
            rows.append({"pair_id": row["pair_id"], "null": label, "identical": identical,
                          "dx": cand.x - base.x, "dy": cand.y - base.y})
    return {"n_pairs": len(manifest), "n_checks": len(rows), "mismatches": mismatches,
            "rows": rows}


if __name__ == "__main__":
    a = check_a()
    print("CHECK A - weighted ZNCC == production ZNCC at uniform weights")
    print(f"  templates tested   : {a['n_templates']}")
    print(f"  dense max |diff|   : {a['dense_max_abs_diff']:.3e}")
    print(f"  point max |diff|   : {a['point_max_abs_diff']:.3e}")
    print(f"  argmax mismatches  : {a['argmax_mismatches']}")
    print()
    b = check_b()
    print("CHECK B - harness null settings reproduce production bit-for-bit")
    print(f"  pairs              : {b['n_pairs']}")
    print(f"  checks (2 per pair): {b['n_checks']}")
    print(f"  mismatches         : {b['mismatches']}")
    if b["mismatches"]:
        for r in b["rows"]:
            if not r["identical"]:
                print(f"    MISMATCH {r['pair_id']} [{r['null']}] dx={r['dx']} dy={r['dy']}")
    else:
        print("  PASS - every pair identical on x, y, confidence, ambiguity_ratio")
