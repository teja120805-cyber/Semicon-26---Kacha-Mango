#!/usr/bin/env python
"""Follow-up to the oracle diagnostic.

The oracle sweep showed every failure is a near-tie: the decoy beats the
true location by <0.05 ZNCC in 100% of cases, <0.02 in 79%. The obvious
next question is whether that thinness is VISIBLE WITHOUT GROUND TRUTH -
i.e. is there a pool-internal signal that flags "this pair is a coin-flip"?

If yes, it is a risk detector: a cheap gate that says which pairs deserve
an expensive disambiguation step, without touching the 74% that are fine.
If no, any disambiguation must be paid on every pair.

Uses ONLY the unmodified production candidate pool - no ground truth is
read except to label the outcome afterwards. Never modifies pipeline/.
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

from pipeline import candidate_generation  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

DISTINCT_PX = 10.0   # same radius production uses for deduplicate_by_location


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(BASELINE_CSV)
    rows = []
    t0 = time.perf_counter()
    for i, row in df.iterrows():
        ref = cv2.imread(os.path.join(DATA_ROOT, row["reference_path"]), cv2.IMREAD_UNCHANGED)
        search = cv2.imread(os.path.join(DATA_ROOT, row["search_path"]), cv2.IMREAD_UNCHANGED)
        ref = ref.astype(np.float32)
        search = search.astype(np.float32)

        pool = candidate_generation.build_candidate_pool(ref, search)
        cands = candidate_generation.deduplicate_by_location(pool)
        cands = sorted(cands, key=lambda c: c.score, reverse=True)
        top = cands[0]
        # Best candidate at a genuinely DIFFERENT location than the winner.
        runner = next((c for c in cands[1:]
                       if (c.x - top.x) ** 2 + (c.y - top.y) ** 2 > DISTINCT_PX ** 2), None)
        gap = float(top.score - runner.score) if runner is not None else float("nan")
        # Does any distinct-location candidate sit near ground truth?
        d_gt = [np.hypot(c.x - row["gt_x"], c.y - row["gt_y"]) for c in cands]
        best_near_gt = min(range(len(cands)), key=lambda k: d_gt[k])
        rows.append({
            "pair_id": row["pair_id"], "split": row["split"],
            "family": row["pair_id"].rsplit("_", 1)[0],
            "error_px": float(row["error_px"]), "correct": bool(row["error_px"] <= 5.0),
            "n_candidates": len(cands),
            "top_score": float(top.score),
            "runner_score": float(runner.score) if runner is not None else float("nan"),
            "gap": gap,
            "gt_rank": int(best_near_gt) if d_gt[best_near_gt] <= 5.0 else -1,
            "gt_dist_best_cand": float(d_gt[best_near_gt]),
            "gt_cand_score": float(cands[best_near_gt].score),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(df)}  ({time.perf_counter() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "margin_detector.csv"), index=False)

    ok, bad = out[out.correct], out[~out.correct]
    summary = {
        "n": len(out), "n_correct": len(ok), "n_wrong": len(bad),
        "gap_correct": {"median": float(ok.gap.median()), "mean": float(ok.gap.mean()),
                        "q25": float(ok.gap.quantile(.25)), "q10": float(ok.gap.quantile(.10))},
        "gap_wrong": {"median": float(bad.gap.median()), "mean": float(bad.gap.mean()),
                      "q75": float(bad.gap.quantile(.75)), "q90": float(bad.gap.quantile(.90))},
        "gt_in_pool_when_wrong": int((bad.gt_rank >= 0).sum()),
        "gt_in_pool_when_wrong_pct": float((bad.gt_rank >= 0).mean()),
        "sweep_thresholds": {},
    }
    for thr in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        flagged = out[out.gap < thr]
        summary["sweep_thresholds"][str(thr)] = {
            "flagged": int(len(flagged)),
            "flagged_pct": float(len(flagged) / len(out)),
            "wrong_caught": int((~flagged.correct).sum()),
            "recall_of_failures": float((~flagged.correct).sum() / len(bad)),
            "precision": float((~flagged.correct).sum() / len(flagged)) if len(flagged) else 0.0,
        }
    with open(os.path.join(OUT_DIR, "margin_detector_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nElapsed {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
