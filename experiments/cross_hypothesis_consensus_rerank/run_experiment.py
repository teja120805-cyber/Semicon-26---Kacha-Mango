#!/usr/bin/env python
"""Two-stage, honest evaluation of the cross-hypothesis consensus re-ranker:

  1. Sweep (alpha, radius_px) on `development` ONLY (24 pairs) - never on
     validation/held_out/challenge/cross_generator, so the config choice
     can't be tuned against the same data the integration gate later
     scores (same discipline as model/train.py's early-stopping-on-
     development-only rule, and ranking.py's own dev-only epsilon tuning
     comment).
  2. Run the single chosen config over the full frozen benchmark (all 5
     splits, n=156) and write a per-pair CSV in the exact schema
     evaluation/evaluate.py produces, so evaluation/benchmark.py's
     run_integration_gate can compare it against the real production
     baseline CSV unmodified.

Never touches pipeline/, generator/, model/, or data/ - reads images from
the shared data/ directory read-only, writes only under this experiment's
own outputs/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402

from harness import localize_consensus  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

SWEEP_ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
SWEEP_RADII = [10.0, 15.0, 25.0]


def _run_split(split: str, *, alpha: float, radius_px: float, verbose: bool = False) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize_consensus(ref, search, alpha=alpha, radius_px=radius_px)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        rows.append({
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s, "ranking_mode": "consensus",
        })
        if verbose:
            print(f"  [{split}] {row['pair_id']:28s} err={error_px:7.2f}px")
    return pd.DataFrame(rows)


def sweep_on_development() -> dict:
    print("=== Stage 1: sweeping (alpha, radius_px) on development only (n=24) ===")
    dev_manifest = load_manifest(DATA_ROOT, "development")
    dev_images = []
    for _, row in dev_manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        dev_images.append((row, ref, search))

    results = []
    for alpha in SWEEP_ALPHAS:
        for radius_px in SWEEP_RADII:
            if alpha == 0.0 and radius_px != SWEEP_RADII[0]:
                continue  # alpha=0 collapses to rank_classical regardless of radius - only run once
            errors = []
            for row, ref, search in dev_images:
                result = localize_consensus(ref, search, alpha=alpha, radius_px=radius_px)
                errors.append(float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"])))
            errors = np.array(errors)
            acc5 = float(np.mean(errors <= 5.0))
            mean_err = float(np.mean(errors))
            results.append({"alpha": alpha, "radius_px": radius_px, "acc_at_5px": acc5,
                             "mean_error_px": mean_err, "n": len(errors)})
            print(f"  alpha={alpha:<5} radius_px={radius_px:<5} acc@5px={acc5:.3f} mean_err={mean_err:8.2f}px")

    results.sort(key=lambda r: (-r["acc_at_5px"], r["mean_error_px"]))
    best = results[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dev_sweep_results.json"), "w") as f:
        json.dump({"sweep": results, "chosen": best}, f, indent=2)
    print(f"\nChosen config: alpha={best['alpha']} radius_px={best['radius_px']} "
          f"(development acc@5px={best['acc_at_5px']:.3f})")
    return best


def run_frozen_benchmark(alpha: float, radius_px: float) -> pd.DataFrame:
    print(f"\n=== Stage 2: running full frozen benchmark (all splits) with alpha={alpha}, radius_px={radius_px} ===")
    splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
    all_dfs = []
    for split in splits:
        t0 = time.perf_counter()
        df = _run_split(split, alpha=alpha, radius_px=radius_px)
        all_dfs.append(df)
        print(f"  {split}: n={len(df)}, acc@5px={(df['error_px'] <= 5).mean():.3f}, "
              f"time={time.perf_counter() - t0:.1f}s")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "per_pair_results_consensus.csv"), index=False)
    return combined


def main() -> None:
    best = sweep_on_development()
    candidate_df = run_frozen_benchmark(best["alpha"], best["radius_px"])

    candidate_report = metrics.full_report(candidate_df)
    with open(os.path.join(OUT_DIR, "consensus_metrics.json"), "w") as f:
        json.dump(candidate_report, f, indent=2)
    print("\n=== Candidate overall metrics ===")
    print(json.dumps(candidate_report["overall"], indent=2))

    baseline_df = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
    with open(os.path.join(OUT_DIR, "integration_gate_result.json"), "w") as f:
        json.dump(gate, f, indent=2)
    print("\n=== Integration gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))

    baseline_report = metrics.full_report(baseline_df)
    print("\n=== Baseline vs candidate (pooled, all splits combined incl. development) ===")
    b_acc5 = float((baseline_df["error_px"] <= 5).mean())
    c_acc5 = float((candidate_df["error_px"] <= 5).mean())
    print(f"baseline pooled acc@5px (n={len(baseline_df)}): {b_acc5:.4f}")
    print(f"candidate pooled acc@5px (n={len(candidate_df)}): {c_acc5:.4f}")


if __name__ == "__main__":
    main()
