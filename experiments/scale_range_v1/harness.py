"""Instrumented eval loop for the scale_range_v1 experiment (compliance
item A2: help doc / pptx state a nominal 10:1 scale with robustness tests
spanning ~9:1-11:1; the shipped pipeline only searches/tests +-7-8%).

Calls the same production functions everywhere: candidate_generation.build_candidate_pool,
candidate_generation.deduplicate_by_location, ranking.rank_classical,
ranking.apply_center_tiebreak, refinement.refine - every one of these is
imported UNMODIFIED from pipeline/ (reports/V2_ARCHITECTURE_PLAN.md section
10: an experiment never edits pipeline/ in place). This mirrors
evaluation/evaluate.py::evaluate_split exactly, except that scale_hypotheses
is a parameter instead of always defaulting to
candidate_generation.DEFAULT_SCALE_HYPOTHESES - evaluate.py itself is never
touched, since pipeline.localize.localize() already exposes scale_hypotheses
as a keyword argument (it was already there for exactly this kind of
override - see pipeline/localize.py).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from pipeline.localize import localize


def evaluate_with_hypotheses(manifest: pd.DataFrame, scale_hypotheses: tuple, label: str,
                              verbose: bool = False) -> pd.DataFrame:
    records = []
    t0 = time.perf_counter()
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize(ref, search, ranking_mode="classical", scale_hypotheses=scale_hypotheses)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        records.append({
            **row.to_dict(),
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s, "ranking_mode": "classical",
        })
        if verbose:
            print(f"  [{label}] {row['pair_id']:28s} err={error_px:7.2f}px conf={result.confidence:.3f}")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(records)
    acc5 = (df["error_px"] <= 5).mean() if len(df) else float("nan")
    print(f"=== {label}: n={len(df)} acc@5px={acc5:.3f} wall_time={elapsed:.1f}s "
          f"n_hypotheses={len(scale_hypotheses)} ===")
    return df
