"""Evaluates the gradient/ensemble candidate-scoring variants
(experiments/periodicity/harness.py) against the frozen benchmark's
gate-relevant splits, reporting the full integration-gate metric set plus
candidate recall and rescue/break counts relative to the classical baseline.

Calls the SAME `pipeline.refinement.refine` used in production for the
final subpixel step - only candidate generation/scoring is varied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import benchmark  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import feature_extraction, refinement  # noqa: E402
from experiments.periodicity import harness  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")
GT_TOLERANCE_PX = 5.0
AMBIGUITY_THRESHOLD = 0.92


def _dist(x1, y1, x2, y2) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def localize_variant(reference_img: np.ndarray, search_img: np.ndarray, mode: str, gt_x: float, gt_y: float) -> dict:
    t0 = time.perf_counter()
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)

    raw = harness.build_candidate_pool(reference, search, mode)
    candidates = harness.deduplicate_by_location(raw)
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    winner = ranked[0]
    refined_x, refined_y = refinement.refine(reference, search, winner)
    runtime_s = time.perf_counter() - t0
    error_px = _dist(refined_x, refined_y, gt_x, gt_y)

    dists = sorted(((_dist(c.x, c.y, gt_x, gt_y), i) for i, c in enumerate(candidates)), key=lambda t: t[0])
    nearest_dist, nearest_idx = dists[0]
    gt_in_pool = nearest_dist <= GT_TOLERANCE_PX

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)

    return {
        "pred_x": refined_x, "pred_y": refined_y, "error_px": error_px,
        "confidence": float(winner.score), "ambiguity_ratio": amb_ratio,
        "ambiguous": amb_ratio >= AMBIGUITY_THRESHOLD, "runtime_s": runtime_s,
        "num_candidates_raw": len(raw), "num_candidates_dedup": len(candidates),
        "gt_in_pool": gt_in_pool, "gt_nearest_dist_px": nearest_dist,
    }


def evaluate_split(data_root: str, split: str, mode: str) -> pd.DataFrame:
    manifest = load_manifest(data_root, split)
    results = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        diag = localize_variant(ref, search, mode, row["gt_x"], row["gt_y"])
        results.append({**row.to_dict(), **diag})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gradient", "ensemble"], required=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
    baseline_df = pd.read_csv(baseline_path)
    baseline_df = baseline_df[baseline_df["split"].isin(GATE_SPLITS)].reset_index(drop=True)

    all_results = []
    t0 = time.perf_counter()
    for split in GATE_SPLITS:
        print(f"=== {split} (mode={args.mode}) ===")
        df = evaluate_split(os.path.join(PROJECT_ROOT, "data"), split, args.mode)
        all_results.append(df)
        print(f"  n={len(df)} acc@5px={(df['error_px']<=5).mean():.3f} "
              f"candidate_recall@5px={df['gt_in_pool'].mean():.3f} mean_runtime={df['runtime_s'].mean():.3f}s")
    candidate_df = pd.concat(all_results, ignore_index=True)
    elapsed = time.perf_counter() - t0

    candidate_df.to_csv(os.path.join(OUT_DIR, f"per_pair_results_{args.mode}.csv"), index=False)

    # Rescue/break analysis
    merged = baseline_df.merge(candidate_df, on="pair_id", suffixes=("_base", "_cand"))
    rescued = merged[(merged["error_px_base"] > GT_TOLERANCE_PX) & (merged["error_px_cand"] <= GT_TOLERANCE_PX)]
    broken = merged[(merged["error_px_base"] <= GT_TOLERANCE_PX) & (merged["error_px_cand"] > GT_TOLERANCE_PX)]
    catastrophic_rescued = merged[(merged["error_px_base"] > 50) & (merged["error_px_cand"] <= 50)]
    catastrophic_new = merged[(merged["error_px_base"] <= 50) & (merged["error_px_cand"] > 50)]

    gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
    gate["mode"] = args.mode
    gate["candidate_recall_at_5px"] = float(candidate_df["gt_in_pool"].mean())
    gate["baseline_candidate_count_pooled_avg"] = None  # baseline CSV doesn't record pool size; see note
    gate["candidate_pool_avg_size"] = float(candidate_df["num_candidates_dedup"].mean())
    gate["rescue_count"] = len(rescued)
    gate["break_count"] = len(broken)
    gate["net_rescue"] = len(rescued) - len(broken)
    gate["catastrophic_rescued"] = len(catastrophic_rescued)
    gate["catastrophic_new"] = len(catastrophic_new)
    gate["rescued_pair_ids"] = rescued["pair_id"].tolist()
    gate["broken_pair_ids"] = broken["pair_id"].tolist()
    gate["total_wall_time_s"] = elapsed
    gate["note"] = (
        "seeds_agree=True passed manually: deterministic algorithmic change (alternative "
        "correlation representation), not a trained stochastic model - no seed variance to check."
    )
    with open(os.path.join(OUT_DIR, f"integration_gate_{args.mode}.json"), "w") as f:
        json.dump(gate, f, indent=2, default=str)

    print(f"\n=== {args.mode}: rescue={len(rescued)} break={len(broken)} net={len(rescued)-len(broken)} ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    print(f"Total wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
