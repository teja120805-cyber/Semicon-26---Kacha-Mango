"""Widen the integration verification beyond `development`.

verify_integration.py established on 24 development pairs that the
AMBIGUITY_THRESHOLD recalibration moves no prediction. This runs the same
same-process A/B over every complete pair available locally, so the claim
rests on a larger sample than one split.

Only pairs whose Reference AND Search files are both present are used;
`challenge` and `cross_generator` were never mirrored into this container,
and the count actually tested is printed rather than implied.

    python -m experiments.psr_confidence.verify_wide
"""
from __future__ import annotations

import importlib
import json
import os

import cv2
import numpy as np
import pandas as pd

loc = importlib.import_module("pipeline.localize")

OLD_THRESHOLD = 0.92
ROOT = os.path.dirname(os.path.abspath(__file__))


def complete_pairs(split: str):
    with open(f"data/{split}/ground_truth.json") as f:
        recs = json.load(f)
    out = []
    for r in recs:
        rp = os.path.join("data", r["reference_path"])
        sp = os.path.join("data", r["search_path"])
        if os.path.exists(rp) and os.path.exists(sp):
            out.append({"pair_id": r["pair_id"], "split": split,
                        "structural_family": r["structural_family"],
                        "ref": rp, "search": sp, "gt_x": r["gt_x"], "gt_y": r["gt_y"]})
    return out


def run_at(threshold, pairs):
    original = loc.AMBIGUITY_THRESHOLD
    loc.AMBIGUITY_THRESHOLD = threshold
    try:
        rows = []
        for p in pairs:
            ref = cv2.imread(p["ref"], cv2.IMREAD_UNCHANGED)
            srch = cv2.imread(p["search"], cv2.IMREAD_UNCHANGED)
            r = loc.localize(ref, srch)
            rows.append({"pair_id": p["pair_id"], "split": p["split"],
                          "structural_family": p["structural_family"],
                          "x": r.x, "y": r.y, "confidence": r.confidence,
                          "ambiguity_ratio": r.ambiguity_ratio, "ambiguous": bool(r.ambiguous),
                          "error_px": float(np.hypot(r.x - p["gt_x"], r.y - p["gt_y"]))})
        return pd.DataFrame(rows)
    finally:
        loc.AMBIGUITY_THRESHOLD = original


def main() -> None:
    pairs = []
    for s in ("development", "validation", "held_out"):
        got = complete_pairs(s)
        pairs += got
        print(f"  {s:14s} {len(got)} complete pairs")
    print(f"  TOTAL          {len(pairs)} pairs "
          f"(challenge / cross_generator not mirrored locally)\n")

    new = loc.AMBIGUITY_THRESHOLD
    before = run_at(OLD_THRESHOLD, pairs)
    after = run_at(new, pairs)
    m = before.merge(after, on="pair_id", suffixes=("_b", "_a"))

    ok = True
    print(f"AMBIGUITY_THRESHOLD {OLD_THRESHOLD} vs {new}, same process\n")
    for col in ("x", "y", "confidence", "ambiguity_ratio", "error_px"):
        same = bool((m[f"{col}_b"] == m[f"{col}_a"]).all())
        print(f"  {col:16s} bit-identical={same}")
        ok &= same

    ab = float((m.error_px_b <= 5).mean())
    aa = float((m.error_px_a <= 5).mean())
    cb = float((m.error_px_b > 50).mean())
    ca = float((m.error_px_a > 50).mean())
    print(f"\n  pooled accuracy@5px   {ab:.4f} -> {aa:.4f}   identical={ab == aa}")
    print(f"  catastrophic rate     {cb:.4f} -> {ca:.4f}   identical={cb == ca}")
    ok &= (ab == aa) and (cb == ca)

    print("\n  per split")
    for s, g in m.groupby(before.set_index('pair_id').loc[m.pair_id, 'split'].values):
        print(f"    {s:14s} n={len(g):3d}  acc {float((g.error_px_b <= 5).mean()):.4f} -> "
              f"{float((g.error_px_a <= 5).mean()):.4f}")

    wrong = (m.error_px_a > 5).values
    fb, fa = m.ambiguous_b.values, m.ambiguous_a.values
    print(f"\n  flagged ambiguous     {fb.sum()}/{len(m)} -> {fa.sum()}/{len(m)}")
    print(f"  flag precision        {(fb & wrong).sum() / max(fb.sum(), 1):.3f} -> "
          f"{(fa & wrong).sum() / max(fa.sum(), 1):.3f}")
    print(f"  failure recall        {(fb & wrong).sum() / max(wrong.sum(), 1):.3f} -> "
          f"{(fa & wrong).sum() / max(wrong.sum(), 1):.3f}")

    m.to_csv(os.path.join(ROOT, "outputs", "integration_verification_wide.csv"), index=False)
    print("\n" + ("PASS - every prediction bit-identical across all splits tested."
                  if ok else "FAIL - a prediction changed. REVERT."))


if __name__ == "__main__":
    main()
