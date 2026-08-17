#!/usr/bin/env python
"""Two-stage, honest evaluation of ORB keypoint-fusion candidate generation:

  1. Sweep (ratio_threshold, min_inliers) on `development` ONLY (24 pairs) -
     never on validation/held_out/challenge/cross_generator, matching this
     project's established dev-only-tuning discipline.
  2. Run the single chosen config over the full frozen benchmark (all 5
     splits, n=156) and write a per-pair CSV in the exact schema
     evaluation/evaluate.py produces, so evaluation/benchmark.py's
     run_integration_gate can compare it against the real production
     baseline CSV unmodified.

Also logs, per pair, whether the keypoint candidate was generated at all
and whether it ended up as the final winner - direct mechanism diagnostics,
not just aggregate accuracy, matching the discipline of
hough_subpatch_voting's exploratory diagnostic pass.

Never touches pipeline/, generator/, model/, or data/ - reads images from
the shared data/ directory read-only, writes only under this experiment's
own outputs/.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402

from harness import localize_keypoint_fusion  # noqa: E402
from keypoint_gen import generate_keypoint_candidate  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

SWEEP_RATIOS = [0.65, 0.75, 0.85]
SWEEP_MIN_INLIERS = [4, 6, 10]


def _diagnose(reference, search, ranked_x, ranked_y, *, ratio_threshold, min_inliers):
    """Was a keypoint candidate proposed, and did it end up as the winner
    (within 2px, i.e. same location after refinement rounding)?"""
    kp = generate_keypoint_candidate(reference.astype("float32"), search.astype("float32"),
                                      ratio_threshold=ratio_threshold, min_inliers=min_inliers)
    if kp is None:
        return {"kp_proposed": False, "kp_was_winner": False}
    dist = math.hypot(kp.x - ranked_x, kp.y - ranked_y)
    return {"kp_proposed": True, "kp_was_winner": dist < 2.0}


def _run_split(split: str, *, ratio_threshold: float, min_inliers: int,
                diagnostics: bool = False) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize_keypoint_fusion(ref, search, ratio_threshold=ratio_threshold, min_inliers=min_inliers)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        record = {
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s,
            "ranking_mode": "classical+keypoint_fusion",
        }
        if diagnostics:
            record.update(_diagnose(ref, search, result.x, result.y,
                                     ratio_threshold=ratio_threshold, min_inliers=min_inliers))
        rows.append(record)
    return pd.DataFrame(rows)


def sweep_on_development() -> dict:
    print("=== Stage 1: sweeping (ratio_threshold, min_inliers) on development only (n=24) ===")
    results = []
    for ratio_threshold in SWEEP_RATIOS:
        for min_inliers in SWEEP_MIN_INLIERS:
            df = _run_split("development", ratio_threshold=ratio_threshold, min_inliers=min_inliers,
                             diagnostics=True)
            acc5 = float((df["error_px"] <= 5.0).mean())
            mean_err = float(df["error_px"].mean())
            n_proposed = int(df["kp_proposed"].sum())
            n_won = int(df["kp_was_winner"].sum())
            results.append({
                "ratio_threshold": ratio_threshold, "min_inliers": min_inliers,
                "acc_at_5px": acc5, "mean_error_px": mean_err, "n": len(df),
                "kp_proposed_count": n_proposed, "kp_won_count": n_won,
            })
            print(f"  ratio={ratio_threshold:<5} min_inliers={min_inliers:<3} acc@5px={acc5:.3f} "
                  f"mean_err={mean_err:8.2f}px  kp_proposed={n_proposed}/24 kp_won={n_won}/24")

    results.sort(key=lambda r: (-r["acc_at_5px"], r["mean_error_px"]))
    best = results[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dev_sweep_results.json"), "w") as f:
        json.dump({"sweep": results, "chosen": best}, f, indent=2)
    print(f"\nChosen config: ratio_threshold={best['ratio_threshold']} min_inliers={best['min_inliers']} "
          f"(development acc@5px={best['acc_at_5px']:.3f})")
    return best


def run_frozen_benchmark(ratio_threshold: float, min_inliers: int) -> pd.DataFrame:
    print(f"\n=== Stage 2: running full frozen benchmark (all splits) with "
          f"ratio_threshold={ratio_threshold}, min_inliers={min_inliers} ===")
    splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
    all_dfs = []
    for split in splits:
        t0 = time.perf_counter()
        df = _run_split(split, ratio_threshold=ratio_threshold, min_inliers=min_inliers, diagnostics=True)
        all_dfs.append(df)
        n_proposed = int(df["kp_proposed"].sum())
        n_won = int(df["kp_was_winner"].sum())
        print(f"  {split}: n={len(df)}, acc@5px={(df['error_px'] <= 5).mean():.3f}, "
              f"kp_proposed={n_proposed}/{len(df)} kp_won={n_won}/{len(df)}, "
              f"time={time.perf_counter() - t0:.1f}s")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "per_pair_results_keypoint_fusion.csv"), index=False)
    return combined


def main() -> None:
    best = sweep_on_development()
    candidate_df = run_frozen_benchmark(best["ratio_threshold"], best["min_inliers"])

    # Drop diagnostic-only columns before feeding to metrics/gate code (schema match).
    gate_df = candidate_df.drop(columns=["kp_proposed", "kp_was_winner"])

    candidate_report = metrics.full_report(gate_df)
    with open(os.path.join(OUT_DIR, "keypoint_fusion_metrics.json"), "w") as f:
        json.dump(candidate_report, f, indent=2)
    print("\n=== Candidate overall metrics ===")
    print(json.dumps(candidate_report["overall"], indent=2))

    baseline_df = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(baseline_df, gate_df, seeds_agree=True)
    with open(os.path.join(OUT_DIR, "integration_gate_result.json"), "w") as f:
        json.dump(gate, f, indent=2)
    print("\n=== Integration gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))

    b_acc5 = float((baseline_df["error_px"] <= 5).mean())
    c_acc5 = float((gate_df["error_px"] <= 5).mean())
    print(f"\nbaseline pooled acc@5px (n={len(baseline_df)}): {b_acc5:.4f}")
    print(f"candidate pooled acc@5px (n={len(gate_df)}): {c_acc5:.4f}")

    total_proposed = int(candidate_df["kp_proposed"].sum())
    total_won = int(candidate_df["kp_was_winner"].sum())
    print(f"\nKeypoint candidate proposed on {total_proposed}/{len(candidate_df)} pairs, "
          f"became the final winner on {total_won}/{len(candidate_df)} pairs.")


if __name__ == "__main__":
    main()
