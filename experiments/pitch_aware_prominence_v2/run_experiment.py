#!/usr/bin/env python
"""Two-stage, honest evaluation of pitch-aware prominence v2 (penalty-only
formula):

  1. Sweep (gamma, top_k) on `development` ONLY (24 pairs) - never on
     validation/held_out/challenge/cross_generator.
  2. Run the single chosen config over the full frozen benchmark (all 5
     splits, n=156) and write a per-pair CSV in the exact schema
     evaluation/evaluate.py produces, so evaluation/benchmark.py's
     run_integration_gate can compare it against the real production
     baseline CSV unmodified.

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402

from harness import localize_pitch_aware_v2  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

# Gamma can go much higher than v1's sweep now that positive bonuses are
# clipped out - a large gamma only ever strengthens the penalty on
# candidates with a genuine nearby periodic competitor, it can no longer
# runaway-reward a spurious outlier the way v1's symmetric formula could.
SWEEP_GAMMA = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 40.0]
SWEEP_TOPK = [4, 8]


def _run_split(split: str, *, gamma: float, top_k: int) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize_pitch_aware_v2(ref, search, gamma=gamma, top_k=top_k)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        rows.append({
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s,
            "ranking_mode": "pitch_aware_prominence_v2",
        })
    return pd.DataFrame(rows)


def sweep_on_development() -> dict:
    print("=== Stage 1: sweeping (gamma, top_k) on development only (n=24) ===")
    results = []
    for gamma in SWEEP_GAMMA:
        for top_k in SWEEP_TOPK:
            if gamma == 0.0 and top_k != SWEEP_TOPK[0]:
                continue
            df = _run_split("development", gamma=gamma, top_k=top_k)
            acc5 = float((df["error_px"] <= 5.0).mean())
            mean_err = float(df["error_px"].mean())
            results.append({"gamma": gamma, "top_k": top_k, "acc_at_5px": acc5,
                             "mean_error_px": mean_err, "n": len(df)})
            print(f"  gamma={gamma:<6} top_k={top_k:<3} acc@5px={acc5:.3f} mean_err={mean_err:8.2f}px")

    results.sort(key=lambda r: (-r["acc_at_5px"], r["mean_error_px"]))
    best = results[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dev_sweep_results.json"), "w") as f:
        json.dump({"sweep": results, "chosen": best}, f, indent=2)
    print(f"\nChosen config: gamma={best['gamma']} top_k={best['top_k']} "
          f"(development acc@5px={best['acc_at_5px']:.3f})")
    return best


def run_frozen_benchmark(gamma: float, top_k: int) -> pd.DataFrame:
    print(f"\n=== Stage 2: running full frozen benchmark (all splits) with gamma={gamma}, top_k={top_k} ===")
    splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
    all_dfs = []
    for split in splits:
        t0 = time.perf_counter()
        df = _run_split(split, gamma=gamma, top_k=top_k)
        all_dfs.append(df)
        print(f"  {split}: n={len(df)}, acc@5px={(df['error_px'] <= 5).mean():.3f}, "
              f"time={time.perf_counter() - t0:.1f}s")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "per_pair_results_pitch_aware_v2.csv"), index=False)
    return combined


def main() -> None:
    best = sweep_on_development()
    candidate_df = run_frozen_benchmark(best["gamma"], best["top_k"])

    candidate_report = metrics.full_report(candidate_df)
    with open(os.path.join(OUT_DIR, "pitch_aware_v2_metrics.json"), "w") as f:
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

    b = baseline_df.set_index("pair_id")["error_px"]
    c = candidate_df.set_index("pair_id")["error_px"]
    common = b.index.intersection(c.index)
    rescued = int(((b[common] > 5) & (c[common] <= 5)).sum())
    broken = int(((b[common] <= 5) & (c[common] > 5)).sum())
    changed = int(((b[common] - c[common]).abs() > 0.01).sum())
    print(f"\nRescued across 5px line: {rescued}  |  Broken across 5px line: {broken}  |  Total pairs changed: {changed}")


if __name__ == "__main__":
    main()
