"""Runs the context-separation analysis (experiments/spatial_context/harness.py)
over the worst known failures on the frozen benchmark: for each, compares
the TRUE location against the WINNING DECOY location at increasing context
window sizes, measuring whether a "distinctiveness" score
(1 - periodicity_strength) separates true from decoy as context widens.

Also checks the reverse-safety question: does using a wider window change
anything for currently-CORRECT cases (a sanity check that the metric isn't
just noise), by running the same measurement on a matched sample of
successful pairs.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.spatial_context.harness import extract_crop, periodicity_strength  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
WINDOW_SIZES_PX = (100, 125, 150, 200, 300)  # 1.0x, 1.25x, 1.5x, 2x, 3x the 100px reference footprint
TOP_N_FAILURES = 45


def distinctiveness(crop) -> float | None:
    if crop is None:
        return None
    return 1.0 - periodicity_strength(crop)


def analyze_case(search_img: np.ndarray, true_x: float, true_y: float, pred_x: float, pred_y: float) -> dict:
    row = {}
    for w in WINDOW_SIZES_PX:
        true_crop = extract_crop(search_img, true_x, true_y, w)
        decoy_crop = extract_crop(search_img, pred_x, pred_y, w)
        d_true = distinctiveness(true_crop)
        d_decoy = distinctiveness(decoy_crop)
        row[f"distinctiveness_true_w{w}"] = d_true
        row[f"distinctiveness_decoy_w{w}"] = d_decoy
        row[f"separation_w{w}"] = (d_true - d_decoy) if (d_true is not None and d_decoy is not None) else None
    return row


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "reports", "baseline_failure_decomposition.csv"))

    failures = df[df["failure_location"] != "success"].sort_values("error_px", ascending=False).head(TOP_N_FAILURES)
    successes = df[df["failure_location"] == "success"].sample(n=min(TOP_N_FAILURES, (df["failure_location"] == "success").sum()), random_state=42)

    def run_group(group_df: pd.DataFrame, label: str) -> pd.DataFrame:
        rows = []
        for _, row in group_df.iterrows():
            search = cv2.imread(os.path.join(PROJECT_ROOT, row["search_path"]), cv2.IMREAD_UNCHANGED)
            case = analyze_case(search, row["gt_x"], row["gt_y"], row["pred_x"], row["pred_y"])
            case["pair_id"] = row["pair_id"]
            case["error_px"] = row["error_px"]
            case["failure_location"] = row["failure_location"]
            case["periodicity_score"] = row.get("periodicity_score")
            case["group"] = label
            rows.append(case)
        return pd.DataFrame(rows)

    failures_result = run_group(failures, "failure")
    successes_result = run_group(successes, "success")
    combined = pd.concat([failures_result, successes_result], ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "context_separation_per_case.csv"), index=False)

    print("=== Mean separation (distinctiveness[true] - distinctiveness[decoy]) by window size ===")
    summary = {"window_sizes_px": list(WINDOW_SIZES_PX)}
    for label, group in [("failure", failures_result), ("success", successes_result)]:
        print(f"\n-- {label} cases (n={len(group)}) --")
        group_summary = {}
        for w in WINDOW_SIZES_PX:
            col = f"separation_w{w}"
            valid = group[col].dropna()
            mean_sep = float(valid.mean()) if len(valid) else None
            frac_positive = float((valid > 0).mean()) if len(valid) else None
            print(f"  w={w:4d}px  mean_separation={mean_sep}  frac_positive={frac_positive}  n_valid={len(valid)}")
            group_summary[w] = {"mean_separation": mean_sep, "frac_positive": frac_positive, "n_valid": int(len(valid))}
        summary[label] = group_summary

    # Does separation trend upward with window size for the failure cases specifically?
    fail_means = [summary["failure"][w]["mean_separation"] for w in WINDOW_SIZES_PX]
    print(f"\nFailure-case mean separation trend across window sizes: {fail_means}")
    trending_up = all(
        (fail_means[i] is None or fail_means[i + 1] is None or fail_means[i + 1] >= fail_means[i] - 0.02)
        for i in range(len(fail_means) - 1)
    )
    summary["failure_case_separation_trend"] = fail_means
    summary["monotonic_or_flat_increase"] = trending_up

    with open(os.path.join(OUT_DIR, "context_separation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/context_separation_summary.json and _per_case.csv")


if __name__ == "__main__":
    main()
