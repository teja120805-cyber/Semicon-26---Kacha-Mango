#!/usr/bin/env python
"""Two-stage, honest evaluation of PSF-matched template construction.

  1. Sweep sigma_extra on `development` ONLY (24 pairs).
  2. Run the single chosen sigma over the full frozen benchmark (all 5
     splits, n=156), write a per-pair CSV in the exact schema
     evaluation/evaluate.py produces, and hand it to the unmodified
     evaluation/benchmark.py integration gate against the real production
     baseline.

sigma_extra = 0.0 is included in the sweep and is provably identical to the
production pipeline, so the sweep contains its own null control.

Never touches pipeline/, generator/, model/, or data/.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402

from harness import localize_psf  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

SWEEP_SIGMA = [0.0, 0.4, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6]


def _run_split(split: str, *, sigma: float) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        res = localize_psf(ref, search, sigma_extra=sigma)
        rows.append({**row.to_dict(),
                     "pred_x": res.x, "pred_y": res.y,
                     "error_px": float(np.hypot(res.x - row["gt_x"], res.y - row["gt_y"])),
                     "confidence": res.confidence, "ambiguity_ratio": res.ambiguity_ratio,
                     "ambiguous": res.ambiguous, "runtime_s": res.runtime_s,
                     "ranking_mode": "psf_matched_template"})
    return pd.DataFrame(rows)


def sweep_on_development() -> dict:
    print("=== Stage 1: sweeping sigma_extra on development ONLY (n=24) ===")
    results = []
    for sigma in SWEEP_SIGMA:
        df = _run_split("development", sigma=sigma)
        acc5 = float((df["error_px"] <= 5.0).mean())
        results.append({"sigma_extra": sigma, "acc_at_5px": acc5,
                        "mean_error_px": float(df["error_px"].mean()),
                        "median_error_px": float(df["error_px"].median()), "n": len(df)})
        print(f"  sigma={sigma:<5} acc@5px={acc5:.3f}  "
              f"mean_err={results[-1]['mean_error_px']:8.2f}px  "
              f"median_err={results[-1]['median_error_px']:6.2f}px")
    results.sort(key=lambda r: (-r["acc_at_5px"], r["mean_error_px"]))
    best = results[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dev_sweep_results.json"), "w") as f:
        json.dump({"sweep": sorted(results, key=lambda r: r["sigma_extra"]), "chosen": best}, f, indent=2)
    print(f"\nChosen: sigma_extra={best['sigma_extra']} (development acc@5px={best['acc_at_5px']:.3f})")
    return best


def run_frozen_benchmark(sigma: float) -> pd.DataFrame:
    print(f"\n=== Stage 2: full frozen benchmark, sigma_extra={sigma} ===")
    dfs = []
    for split in ["development", "validation", "held_out", "challenge", "cross_generator"]:
        t0 = time.perf_counter()
        df = _run_split(split, sigma=sigma)
        dfs.append(df)
        print(f"  {split}: n={len(df)} acc@5px={(df['error_px'] <= 5).mean():.3f} "
              f"({time.perf_counter() - t0:.0f}s)")
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "per_pair_results_psf.csv"), index=False)
    return combined


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    best = sweep_on_development()
    candidate_df = run_frozen_benchmark(best["sigma_extra"])

    report = metrics.full_report(candidate_df)
    with open(os.path.join(OUT_DIR, "psf_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    baseline_df = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
    with open(os.path.join(OUT_DIR, "integration_gate_result.json"), "w") as f:
        json.dump(gate, f, indent=2)

    b_acc = float((baseline_df["error_px"] <= 5).mean())
    c_acc = float((candidate_df["error_px"] <= 5).mean())
    print("\n=== Integration gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    print(f"\nbaseline pooled acc@5px: {b_acc:.4f}")
    print(f"candidate pooled acc@5px: {c_acc:.4f}   ({c_acc - b_acc:+.4f})")

    print("\n=== per-split ===")
    for split in candidate_df["split"].unique():
        bb = baseline_df[baseline_df.split == split]
        cc = candidate_df[candidate_df.split == split]
        print(f"  {split:<16} {(bb.error_px <= 5).mean():.3f} -> {(cc.error_px <= 5).mean():.3f}")

    b = baseline_df.set_index("pair_id")["error_px"]
    c = candidate_df.set_index("pair_id")["error_px"]
    common = b.index.intersection(c.index)
    rescued = int(((b[common] > 5) & (c[common] <= 5)).sum())
    broken = int(((b[common] <= 5) & (c[common] > 5)).sum())
    print(f"\nRescued: {rescued}  |  Broken: {broken}  |  net {rescued - broken:+d}")
    print(f"runtime multiplier: {candidate_df.runtime_s.sum() / baseline_df.runtime_s.sum():.2f}x")


if __name__ == "__main__":
    main()
