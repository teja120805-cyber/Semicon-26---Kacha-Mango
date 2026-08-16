#!/usr/bin/env python
"""Two-stage, honest evaluation of coarse-to-fine pyramid candidate fusion:

  1. Sweep (downsample_factor, top_k_coarse) on `development` ONLY
     (24 pairs) - never on validation/held_out/challenge/cross_generator,
     matching this project's established dev-only-tuning discipline.
     window_margin_px held fixed at 15 (generous relative to the ~5-7.5px
     measured periodic pitch and the grid's own scale/rotation step sizes).
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

from harness import localize_pyramid  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

SWEEP_DOWNSAMPLE = [3.0, 4.0, 6.0]
SWEEP_TOPK = [3, 5]
WINDOW_MARGIN_PX = 15


def _run_split(split: str, *, downsample_factor: float, top_k_coarse: int) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize_pyramid(ref, search, downsample_factor=downsample_factor,
                                   top_k_coarse=top_k_coarse, window_margin_px=WINDOW_MARGIN_PX)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        rows.append({
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s,
            "ranking_mode": "classical+pyramid_periodicity",
        })
    return pd.DataFrame(rows)


def sweep_on_development() -> dict:
    print("=== Stage 1: sweeping (downsample_factor, top_k_coarse) on development only (n=24) ===")
    results = []
    for downsample_factor in SWEEP_DOWNSAMPLE:
        for top_k_coarse in SWEEP_TOPK:
            df = _run_split("development", downsample_factor=downsample_factor, top_k_coarse=top_k_coarse)
            acc5 = float((df["error_px"] <= 5.0).mean())
            mean_err = float(df["error_px"].mean())
            results.append({
                "downsample_factor": downsample_factor, "top_k_coarse": top_k_coarse,
                "acc_at_5px": acc5, "mean_error_px": mean_err, "n": len(df),
            })
            print(f"  downsample={downsample_factor:<5} top_k_coarse={top_k_coarse:<3} "
                  f"acc@5px={acc5:.3f} mean_err={mean_err:8.2f}px")

    results.sort(key=lambda r: (-r["acc_at_5px"], r["mean_error_px"]))
    best = results[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dev_sweep_results.json"), "w") as f:
        json.dump({"sweep": results, "chosen": best}, f, indent=2)
    print(f"\nChosen config: downsample_factor={best['downsample_factor']} "
          f"top_k_coarse={best['top_k_coarse']} (development acc@5px={best['acc_at_5px']:.3f})")
    return best


def run_frozen_benchmark(downsample_factor: float, top_k_coarse: int) -> pd.DataFrame:
    print(f"\n=== Stage 2: running full frozen benchmark (all splits) with "
          f"downsample_factor={downsample_factor}, top_k_coarse={top_k_coarse} ===")
    splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
    all_dfs = []
    for split in splits:
        t0 = time.perf_counter()
        df = _run_split(split, downsample_factor=downsample_factor, top_k_coarse=top_k_coarse)
        all_dfs.append(df)
        print(f"  {split}: n={len(df)}, acc@5px={(df['error_px'] <= 5).mean():.3f}, "
              f"time={time.perf_counter() - t0:.1f}s")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "per_pair_results_pyramid.csv"), index=False)
    return combined


def main() -> None:
    best = sweep_on_development()
    candidate_df = run_frozen_benchmark(best["downsample_factor"], best["top_k_coarse"])

    candidate_report = metrics.full_report(candidate_df)
    with open(os.path.join(OUT_DIR, "pyramid_metrics.json"), "w") as f:
        json.dump(candidate_report, f, indent=2)
    print("\n=== Candidate overall metrics ===")
    print(json.dumps(candidate_report["overall"], indent=2))

    baseline_df = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
    with open(os.path.join(OUT_DIR, "integration_gate_result.json"), "w") as f:
        json.dump(gate, f, indent=2)
    print("\n=== Integration gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))

    b_acc5 = float((baseline_df["error_px"] <= 5).mean())
    c_acc5 = float((candidate_df["error_px"] <= 5).mean())
    print(f"\nbaseline pooled acc@5px (n={len(baseline_df)}): {b_acc5:.4f}")
    print(f"candidate pooled acc@5px (n={len(candidate_df)}): {c_acc5:.4f}")


if __name__ == "__main__":
    main()
