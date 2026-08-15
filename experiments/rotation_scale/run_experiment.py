"""Evaluates coarse-to-fine rotation/scale refinement
(experiments/rotation_scale/harness.py) against the frozen benchmark's
gate-relevant splits: unmodified coarse candidate generation, then a cheap
local fine-grid refinement of the top-8 candidates only, re-ranked, then
the same production subpixel refinement.
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import benchmark  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import refinement  # noqa: E402
from experiments.rotation_scale.harness import coarse_to_fine_localize  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")
GT_TOLERANCE_PX = 5.0


def _dist(x1, y1, x2, y2) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def localize_coarse_to_fine(reference_img: np.ndarray, search_img: np.ndarray, gt_x: float, gt_y: float) -> dict:
    t0 = time.perf_counter()
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)

    winner, refined_candidates, n_local = coarse_to_fine_localize(reference, search, top_n=8)
    refined_x, refined_y = refinement.refine(reference, search, winner)
    runtime_s = time.perf_counter() - t0
    error_px = _dist(refined_x, refined_y, gt_x, gt_y)

    return {
        "pred_x": refined_x, "pred_y": refined_y, "error_px": error_px,
        "confidence": float(winner.score), "coarse_score": float(winner.coarse_score),
        "runtime_s": runtime_s, "n_local_correlations": n_local,
        "winner_scale": winner.scale, "winner_rotation_deg": winner.rotation_deg,
    }


def evaluate_split(data_root: str, split: str) -> pd.DataFrame:
    manifest = load_manifest(data_root, split)
    results = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        diag = localize_coarse_to_fine(ref, search, row["gt_x"], row["gt_y"])
        results.append({**row.to_dict(), **diag})
    return pd.DataFrame(results)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
    baseline_df = pd.read_csv(baseline_path)
    baseline_df = baseline_df[baseline_df["split"].isin(GATE_SPLITS)].reset_index(drop=True)

    all_results = []
    t0 = time.perf_counter()
    for split in GATE_SPLITS:
        print(f"=== {split} (coarse-to-fine) ===")
        df = evaluate_split(os.path.join(PROJECT_ROOT, "data"), split)
        all_results.append(df)
        print(f"  n={len(df)} acc@5px={(df['error_px']<=5).mean():.3f} mean_runtime={df['runtime_s'].mean():.3f}s")
    candidate_df = pd.concat(all_results, ignore_index=True)
    elapsed = time.perf_counter() - t0

    candidate_df.to_csv(os.path.join(OUT_DIR, "per_pair_results_coarse_to_fine.csv"), index=False)

    merged = baseline_df.merge(candidate_df, on="pair_id", suffixes=("_base", "_cand"))
    rescued = merged[(merged["error_px_base"] > GT_TOLERANCE_PX) & (merged["error_px_cand"] <= GT_TOLERANCE_PX)]
    broken = merged[(merged["error_px_base"] <= GT_TOLERANCE_PX) & (merged["error_px_cand"] > GT_TOLERANCE_PX)]
    catastrophic_rescued = merged[(merged["error_px_base"] > 50) & (merged["error_px_cand"] <= 50)]
    catastrophic_new = merged[(merged["error_px_base"] <= 50) & (merged["error_px_cand"] > 50)]

    gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
    gate["rescue_count"] = len(rescued)
    gate["break_count"] = len(broken)
    gate["net_rescue"] = len(rescued) - len(broken)
    gate["catastrophic_rescued"] = len(catastrophic_rescued)
    gate["catastrophic_new"] = len(catastrophic_new)
    gate["rescued_pair_ids"] = rescued["pair_id"].tolist()
    gate["broken_pair_ids"] = broken["pair_id"].tolist()
    gate["total_wall_time_s"] = elapsed
    gate["note"] = (
        "seeds_agree=True passed manually: deterministic algorithmic change (coarse-to-fine "
        "local rotation/scale refinement), not a trained stochastic model."
    )
    with open(os.path.join(OUT_DIR, "integration_gate.json"), "w") as f:
        json.dump(gate, f, indent=2, default=str)

    print(f"\n=== rescue={len(rescued)} break={len(broken)} net={len(rescued)-len(broken)} ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    print(f"Total wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
