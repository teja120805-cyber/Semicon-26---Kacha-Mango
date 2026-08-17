"""Post-integration verification for the AMBIGUITY_THRESHOLD recalibration.

The claim is that this change cannot move any prediction. That was argued
from a grep of every consumer; this checks it against the running pipeline,
per pair, which is the only version of the claim worth having.

Method: run BOTH thresholds in the SAME process, comparing in-memory floats.

An earlier version of this script compared against a CSV baseline written
before the edit and reported spurious differences of 1e-13 on pred_y and
1e-16 on ambiguity_ratio - float64 CSV round-trip artifacts, not behaviour
changes (pred_x and confidence were exactly equal, which a real algorithmic
change could not produce). Comparing in-memory removes serialization from
the question entirely rather than papering over it with a tolerance.

  MUST be identical : x, y, confidence, ambiguity_ratio, error_px, accuracy
  MUST change       : the `ambiguous` flag (else the edit did nothing)

    python -m experiments.psr_confidence.verify_integration
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd

import importlib

from evaluation.evaluate import load_manifest

# NOT `from pipeline import localize` - pipeline/__init__.py re-exports the
# localize FUNCTION under that name, which shadows the module. This test has
# to rebind a module-level constant, so it needs the module itself.
loc = importlib.import_module("pipeline.localize")

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_THRESHOLD = 0.92


def run_at(threshold: float, manifest) -> pd.DataFrame:
    original = loc.AMBIGUITY_THRESHOLD
    loc.AMBIGUITY_THRESHOLD = threshold
    try:
        rows = []
        for _, row in manifest.iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
            srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
            r = loc.localize(ref, srch)
            rows.append({"pair_id": row["pair_id"], "x": r.x, "y": r.y,
                          "confidence": r.confidence, "ambiguity_ratio": r.ambiguity_ratio,
                          "ambiguous": bool(r.ambiguous),
                          "error_px": float(np.hypot(r.x - row["gt_x"], r.y - row["gt_y"]))})
        return pd.DataFrame(rows)
    finally:
        loc.AMBIGUITY_THRESHOLD = original


def main() -> None:
    manifest = load_manifest("data", "development")
    new_threshold = loc.AMBIGUITY_THRESHOLD
    print(f"comparing AMBIGUITY_THRESHOLD {OLD_THRESHOLD} (previous) vs "
          f"{new_threshold} (integrated), same process, {len(manifest)} pairs\n")

    before = run_at(OLD_THRESHOLD, manifest)
    after = run_at(new_threshold, manifest)
    m = before.merge(after, on="pair_id", suffixes=("_b", "_a"))

    ok = True
    for col in ("x", "y", "confidence", "ambiguity_ratio", "error_px"):
        same = bool((m[f"{col}_b"] == m[f"{col}_a"]).all())
        print(f"  {col:16s} bit-identical={same}")
        ok &= same

    acc_b = float((m.error_px_b <= 5).mean())
    acc_a = float((m.error_px_a <= 5).mean())
    cat_b = float((m.error_px_b > 50).mean())
    cat_a = float((m.error_px_a > 50).mean())
    print(f"\n  accuracy@5px       {acc_b:.4f} -> {acc_a:.4f}   identical={acc_b == acc_a}")
    print(f"  catastrophic rate  {cat_b:.4f} -> {cat_a:.4f}   identical={cat_b == cat_a}")
    ok &= (acc_b == acc_a) and (cat_b == cat_a)

    wrong = (m.error_px_a > 5).values
    fb, fa = m.ambiguous_b.values, m.ambiguous_a.values
    pb = (fb & wrong).sum() / max(fb.sum(), 1)
    pa = (fa & wrong).sum() / max(fa.sum(), 1)
    rb = (fb & wrong).sum() / max(wrong.sum(), 1)
    ra = (fa & wrong).sum() / max(wrong.sum(), 1)
    print(f"\n  flagged ambiguous  {fb.sum()}/{len(m)} -> {fa.sum()}/{len(m)}")
    print(f"  flag precision     {pb:.3f} -> {pa:.3f}")
    print(f"  failure recall     {rb:.3f} -> {ra:.3f}")

    moved = int((fb != fa).sum())
    print(f"\n  pairs whose flag changed: {moved}")

    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    m.to_csv(os.path.join(out, "integration_verification.csv"), index=False)

    print()
    if ok and moved:
        print("PASS - every prediction bit-identical; only the reported flag moved.")
    elif ok:
        print("FAIL - predictions identical but the flag never moved; the edit did nothing.")
    else:
        print("FAIL - a prediction changed. This change must not do that. REVERT.")


if __name__ == "__main__":
    main()
