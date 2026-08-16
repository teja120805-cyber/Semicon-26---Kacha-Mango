#!/usr/bin/env python
"""Runs the joint scale/rotation/position refinement over the full frozen
benchmark (all 5 splits, n=156) and compares against the production
baseline CSV via the unmodified integration gate. No hyperparameters to
tune (the refinement is deterministic parabolic interpolation against the
existing hypothesis grid), so there is no dev-only sweep stage here -
straight to the full benchmark, same as how refinement.py itself has no
tunable knobs in production.

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

from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402

from harness import localize_joint_refine  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")


def _run_split(split: str) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize_joint_refine(ref, search)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        rows.append({
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s,
            "ranking_mode": "classical+joint_refine",
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
    all_dfs = []
    for split in splits:
        t0 = time.perf_counter()
        df = _run_split(split)
        all_dfs.append(df)
        print(f"  {split}: n={len(df)}, acc@5px={(df['error_px'] <= 5).mean():.3f}, "
              f"time={time.perf_counter() - t0:.1f}s")
    candidate_df = pd.concat(all_dfs, ignore_index=True)
    candidate_df.to_csv(os.path.join(OUT_DIR, "per_pair_results_joint_refine.csv"), index=False)

    candidate_report = metrics.full_report(candidate_df)
    with open(os.path.join(OUT_DIR, "joint_refine_metrics.json"), "w") as f:
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
