#!/usr/bin/env python
"""Two-stage evaluation of sub-patch geometric-consistency re-ranking:
sweep (top_k, beta) on development only (n=24), then run the full frozen
benchmark with the chosen config and compare via the unmodified integration
gate. Same dev-only tuning discipline as
experiments/cross_hypothesis_consensus_rerank/run_experiment.py.

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

from harness import localize_subpatch  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

SWEEP_TOP_K = [3, 5, 8]
SWEEP_BETA = [0.0, 0.1, 0.2, 0.4]


def _run_split(split: str, *, top_k: int, beta: float) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize_subpatch(ref, search, top_k=top_k, beta=beta)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        rows.append({
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s, "ranking_mode": "subpatch",
        })
    return pd.DataFrame(rows)


def sweep_on_development() -> dict:
    print("=== Stage 1: sweeping (top_k, beta) on development only (n=24) ===")
    dev_manifest = load_manifest(DATA_ROOT, "development")
    dev_images = []
    for _, row in dev_manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        dev_images.append((row, ref, search))

    results = []
    for top_k in SWEEP_TOP_K:
        for beta in SWEEP_BETA:
            if beta == 0.0 and top_k != SWEEP_TOP_K[0]:
                continue  # beta=0 collapses to rank_classical regardless of top_k
            errors = []
            for row, ref, search in dev_images:
                result = localize_subpatch(ref, search, top_k=top_k, beta=beta)
                errors.append(float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"])))
            errors = np.array(errors)
            acc5 = float(np.mean(errors <= 5.0))
            mean_err = float(np.mean(errors))
            results.append({"top_k": top_k, "beta": beta, "acc_at_5px": acc5,
                             "mean_error_px": mean_err, "n": len(errors)})
            print(f"  top_k={top_k:<3} beta={beta:<5} acc@5px={acc5:.3f} mean_err={mean_err:8.2f}px")

    results.sort(key=lambda r: (-r["acc_at_5px"], r["mean_error_px"]))
    best = results[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dev_sweep_results.json"), "w") as f:
        json.dump({"sweep": results, "chosen": best}, f, indent=2)
    print(f"\nChosen config: top_k={best['top_k']} beta={best['beta']} "
          f"(development acc@5px={best['acc_at_5px']:.3f})")
    return best


def run_frozen_benchmark(top_k: int, beta: float) -> pd.DataFrame:
    print(f"\n=== Stage 2: running full frozen benchmark with top_k={top_k}, beta={beta} ===")
    splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
    all_dfs = []
    for split in splits:
        t0 = time.perf_counter()
        df = _run_split(split, top_k=top_k, beta=beta)
        all_dfs.append(df)
        print(f"  {split}: n={len(df)}, acc@5px={(df['error_px'] <= 5).mean():.3f}, "
              f"time={time.perf_counter() - t0:.1f}s")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "per_pair_results_subpatch.csv"), index=False)
    return combined


def main() -> None:
    best = sweep_on_development()
    candidate_df = run_frozen_benchmark(best["top_k"], best["beta"])

    candidate_report = metrics.full_report(candidate_df)
    with open(os.path.join(OUT_DIR, "subpatch_metrics.json"), "w") as f:
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
