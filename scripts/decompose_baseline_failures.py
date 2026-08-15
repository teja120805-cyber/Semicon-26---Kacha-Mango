"""Part 1 baseline forensics: decompose the frozen benchmark's actual
per-pair results into failure categories (candidate-generation, ranking,
genuine ambiguity, refinement), cross-referenced with rotation/scale/
boundary/noise metadata, using the same instrumented pipeline wrapper
built for experiments/accuracy_forensics/ - applied here to the real
data/ pairs instead of a synthetic sweep.

Read-only with respect to production: imports pipeline/candidate_generation,
ranking, refinement directly (same as pipeline/localize.py's own
orchestration) purely to expose diagnostics; never modifies them.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from experiments.accuracy_forensics.harness import instrumented_localize  # noqa: E402

SPLITS = ("development", "validation", "held_out", "challenge", "cross_generator")
OUT_DIR = os.path.join("outputs", "reports")


def _bucket_rotation(deg: float) -> str:
    d = abs(deg)
    if d < 1e-9:
        return "none"
    if d <= 1.5:
        return "low (<=1.5deg)"
    if d <= 3.5:
        return "medium (1.5-3.5deg)"
    return "high (>3.5deg)"


def _bucket_scale(s: float) -> str:
    d = abs(s - 1.0)
    if d < 1e-9:
        return "none"
    if d <= 0.03:
        return "low (<=3%)"
    if d <= 0.06:
        return "medium (3-6%)"
    return "high (>6%)"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for split in SPLITS:
        manifest = load_manifest("data", split)
        for _, row in manifest.iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
            search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
            diag = instrumented_localize(ref, search, row["gt_x"], row["gt_y"])
            rec = {**row.to_dict(), **diag}
            rec["rotation_bucket"] = _bucket_rotation(row["rotation_deg"])
            rec["scale_bucket"] = _bucket_scale(row["extra_scale"])
            rec["boundary"] = bool(row["crosses_mat_boundary"] or row["crosses_strip_boundary"])
            records.append(rec)
            print(f"  [{split}] {row['pair_id']:28s} err={diag['error_px']:8.2f}px "
                  f"failure={diag['failure_location']}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "baseline_failure_decomposition.csv"), index=False)

    print("\n=== Failure location, pooled (all splits, n=%d) ===" % len(df))
    print(df["failure_location"].value_counts().to_string())

    print("\n=== Failure location x rotation bucket ===")
    print(pd.crosstab(df["rotation_bucket"], df["failure_location"]).to_string())

    print("\n=== Failure location x scale bucket ===")
    print(pd.crosstab(df["scale_bucket"], df["failure_location"]).to_string())

    print("\n=== Failure location x boundary ===")
    print(pd.crosstab(df["boundary"], df["failure_location"]).to_string())

    print("\n=== Failure location x rotation+scale interaction ===")
    both_drift = (df["rotation_bucket"] != "none") & (df["scale_bucket"] != "none")
    rot_only = (df["rotation_bucket"] != "none") & (df["scale_bucket"] == "none")
    scale_only = (df["rotation_bucket"] == "none") & (df["scale_bucket"] != "none")
    neither = (df["rotation_bucket"] == "none") & (df["scale_bucket"] == "none")
    for name, mask in [("neither", neither), ("rotation_only", rot_only),
                        ("scale_only", scale_only), ("both", both_drift)]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        acc5 = (sub["error_px"] <= 5).mean()
        print(f"  {name:14s} n={len(sub):4d} acc@5px={acc5:.3f} "
              f"failure_mix={sub['failure_location'].value_counts().to_dict()}")

    summary = {
        "n": len(df),
        "failure_location_counts": df["failure_location"].value_counts().to_dict(),
        "by_rotation_bucket": {
            k: v["failure_location"].value_counts().to_dict() for k, v in df.groupby("rotation_bucket")
        },
        "by_scale_bucket": {
            k: v["failure_location"].value_counts().to_dict() for k, v in df.groupby("scale_bucket")
        },
        "by_boundary": {
            str(k): v["failure_location"].value_counts().to_dict() for k, v in df.groupby("boundary")
        },
    }
    with open(os.path.join(OUT_DIR, "baseline_failure_decomposition_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/baseline_failure_decomposition.csv and _summary.json")


if __name__ == "__main__":
    main()
