"""Compute the confidence-gating figures for the README from the regenerated
frozen-benchmark results.

The README's Key Results table is entirely frozen-benchmark numbers, so the
confidence figures quoted there must be measured on the same 156 pairs - NOT
carried over from `experiments/psr_confidence/`'s tuning and held-back
surfaces, whose flag precision differs because precision tracks the failure
base rate (see reports/GATE_EXCEPTIONS.md exception 4).

Run against `outputs/reports/per_pair_results.csv` AFTER
`scripts/evaluate_model.py` has been re-run with AMBIGUITY_THRESHOLD = 0.990.

    python -m experiments.psr_confidence.frozen_confidence_numbers \
        --csv outputs/reports/per_pair_results.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

TOL_PX = 5.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/reports/per_pair_results.csv")
    p.add_argument("--old-csv", default=None,
                   help="pre-recalibration copy, to confirm predictions are unchanged")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    correct = (df.error_px <= TOL_PX).values
    flagged = df.ambiguous.astype(bool).values
    wrong = ~correct
    n = len(df)

    print(f"frozen benchmark: n={n}, pooled accuracy@5px = {correct.mean():.4f}\n")

    answered = ~flagged
    print("confidence gating at the integrated AMBIGUITY_THRESHOLD")
    print(f"  flagged ambiguous     {flagged.sum()}/{n}  ({flagged.mean():.1%})")
    print(f"  answered (unflagged)  {answered.sum()}/{n}  ({answered.mean():.1%})")
    if answered.sum():
        print(f"  accuracy on answered  {correct[answered].mean():.4f}")
    if flagged.sum():
        print(f"  flag precision        {(flagged & wrong).sum() / flagged.sum():.4f}")
    if wrong.sum():
        print(f"  failure recall        {(flagged & wrong).sum() / wrong.sum():.4f}")

    print("\n  per split")
    for s, g in df.groupby("split"):
        c = (g.error_px <= TOL_PX).values
        f = g.ambiguous.astype(bool).values
        a = ~f
        acc_a = c[a].mean() if a.sum() else float("nan")
        print(f"    {s:16s} n={len(g):3d}  answered {a.sum():3d} ({a.mean():5.1%})  "
              f"acc on answered {acc_a:.3f}")

    if args.old_csv:
        old = pd.read_csv(args.old_csv)
        m = old.merge(df, on="pair_id", suffixes=("_b", "_a"))
        print(f"\nprediction-invariance check against {args.old_csv} (n={len(m)})")
        allsame = True
        for col in ("pred_x", "pred_y", "error_px", "confidence"):
            if f"{col}_b" not in m:
                continue
            # CSV round-trip can differ in the last ulp; compare at float64
            # epsilon rather than exact equality, and print the worst case so
            # the reader can see it is round-trip noise and not behaviour.
            d = (m[f"{col}_b"] - m[f"{col}_a"]).abs().max()
            scale = max(float(m[f"{col}_b"].abs().max()), 1.0)
            same = bool(d <= 1e-9 * scale)
            allsame &= same
            print(f"  {col:12s} max|diff|={d:.3e}  within float64 round-trip={same}")
        ab = (old.error_px <= TOL_PX).mean()
        aa = correct.mean()
        print(f"  accuracy@5px  {ab:.4f} -> {aa:.4f}  identical={ab == aa}")
        print("\n  " + ("PASS - predictions unchanged; only the ambiguous flag moved."
                        if allsame and ab == aa else
                        "FAIL - something other than the flag changed. Investigate."))


if __name__ == "__main__":
    main()
