#!/usr/bin/env python
"""Step 2 of the center_tiebreak_v2 experiment: investigate the actual
score-gap distribution between the top-2/top-3 DEDUPLICATED candidates
across the FULL frozen benchmark (development + validation + held_out +
challenge + cross_generator, n=156) - not just `development` (n=24), which
is what reports/TIE_BREAK_IMPLEMENTATION.md's rejected first attempt used
("the smallest gap observed... across the development split was ~3.8e-4")
before generalizing a threshold from too small a sample.

Calls the unmodified production functions (candidate_generation.build_candidate_pool,
deduplicate_by_location, ranking.rank_classical, ranking.apply_center_tiebreak) - no
pipeline/ code changes. Because deduplicate_by_location already enforces >10px
separation between kept candidates, every top1/top2 gap measured here is a gap
between two GENUINELY DIFFERENT locations, not two detections of the same site -
that confound is structurally impossible post-dedup.

For every pair, records:
  - the raw score gap between rank 1-2 and rank 2-3 (absolute ZNCC-score
    difference AND as a fraction of the top score)
  - whether the pre-tiebreak winner is already correct (<=5px)
  - the same candidate_generation/candidate_ranking/genuine_ambiguity/refinement
    failure-location taxonomy this project's forensics already uses
  - a sweep, using the real (unmodified) apply_center_tiebreak, of how many
    pairs would qualify as "tied" (and whether they're currently correct or
    not) at each of several candidate threshold values

This is diagnostic only - it never decides anything by itself. The threshold
actually adopted, and its justification from this data, is documented in
REPORT.md; the gate evaluation (run_experiment.py) is run exactly once against
that single chosen value, not swept until the benchmark looks good.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import candidate_generation, ranking, refinement  # noqa: E402
from experiments.center_tiebreak_v2.harness import _classify_failure, _dist  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
ALL_SPLITS = ("development", "validation", "held_out", "challenge", "cross_generator")
GT_TOLERANCE_PX = 5.0

# Swept purely to characterize how many pairs each candidate value would
# touch - NOT to pick whichever looks best on the gate (that would be the
# exact "tuning until the benchmark looks good" this project's culture
# rejects). The value actually adopted is chosen from the percentile/
# separation analysis below, before run_experiment.py is ever run.
THRESHOLD_SWEEP = (1e-6, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 8e-2)


def _periodicity_bucket(score) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "unknown"
    return "high" if score >= 0.66 else ("medium" if score >= 0.33 else "low")


def investigate_pair(row: pd.Series) -> dict:
    ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
    search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
    reference = ref.astype(np.float32)
    search_f = search.astype(np.float32)

    raw = candidate_generation.build_candidate_pool(reference, search_f)
    candidates = candidate_generation.deduplicate_by_location(raw)
    ranked = ranking.rank_classical(candidates)

    n = len(ranked)
    top1 = ranked[0]
    top2 = ranked[1] if n > 1 else None
    top3 = ranked[2] if n > 2 else None

    abs_gap12 = float(top1.score - top2.score) if top2 is not None else float("nan")
    rel_gap12 = float(abs_gap12 / top1.score) if (top2 is not None and top1.score > 1e-9) else float("nan")
    abs_gap23 = float(top2.score - top3.score) if (top2 is not None and top3 is not None) else float("nan")
    rel_gap23 = (float(abs_gap23 / top2.score) if (top2 is not None and top3 is not None and top2.score > 1e-9)
                 else float("nan"))
    top1_top2_dist_px = _dist(top1.x, top1.y, top2.x, top2.y) if top2 is not None else float("nan")

    # Pre-tiebreak correctness (matches production's rank_classical winner,
    # refined - i.e. what ships today, epsilon=1e-6 being provably inert).
    refined_x, refined_y = refinement.refine(reference, search_f, top1)
    final_error_px = _dist(refined_x, refined_y, row["gt_x"], row["gt_y"])
    coarse_error_px = _dist(top1.x, top1.y, row["gt_x"], row["gt_y"])

    dists = sorted(((_dist(c.x, c.y, row["gt_x"], row["gt_y"]), i) for i, c in enumerate(candidates)),
                    key=lambda t: t[0])
    nearest_dist, nearest_idx = dists[0]
    gt_in_pool = nearest_dist <= GT_TOLERANCE_PX
    gt_candidate = candidates[nearest_idx] if gt_in_pool else None
    gt_candidate_score = None
    if gt_candidate is not None:
        for c in ranked:
            if c is gt_candidate:
                gt_candidate_score = float(c.score)
                break
    winner_is_gt_candidate = gt_in_pool and (gt_candidate is top1)
    failure_location = _classify_failure(
        final_error_px=final_error_px, gt_in_pool=gt_in_pool,
        winner_is_gt_candidate=winner_is_gt_candidate, coarse_error_px=coarse_error_px,
        winner_score=float(top1.score), gt_candidate_score=gt_candidate_score,
    )

    # Sweep: does the REAL (unmodified) apply_center_tiebreak fire at each
    # candidate threshold, and if so, does it change the winner?
    sweep = {}
    for t in THRESHOLD_SWEEP:
        reordered = ranking.apply_center_tiebreak(list(ranked), search_f.shape, tie_score_epsilon=t)
        sweep[f"fires_at_{t:g}"] = bool(reordered[0] is not top1)

    return {
        "pair_id": row["pair_id"], "split": row["split"], "structural_family": row["structural_family"],
        "n_candidates_dedup": n,
        "top1_score": float(top1.score), "top2_score": float(top2.score) if top2 is not None else float("nan"),
        "top3_score": float(top3.score) if top3 is not None else float("nan"),
        "abs_gap12": abs_gap12, "rel_gap12": rel_gap12, "abs_gap23": abs_gap23, "rel_gap23": rel_gap23,
        "top1_top2_dist_px": top1_top2_dist_px,
        "pre_tiebreak_error_px": final_error_px, "pre_tiebreak_correct": final_error_px <= GT_TOLERANCE_PX,
        "failure_location": failure_location,
        "periodicity_bucket": _periodicity_bucket(row.get("periodicity_score")),
        "boundary": bool(row.get("crosses_mat_boundary", False) or row.get("crosses_strip_boundary", False)),
        **sweep,
    }


def _percentiles(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {}
    pcts = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    return {f"p{p}": float(np.percentile(values, p)) for p in pcts} | {"n": int(len(values))}


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "n_pairs": len(df),
        "sanity_min_top1_top2_dist_px": float(df["top1_top2_dist_px"].min(skipna=True)),
        "abs_gap12_percentiles_all": _percentiles(df["abs_gap12"].to_numpy()),
        "rel_gap12_percentiles_all": _percentiles(df["rel_gap12"].to_numpy()),
        "abs_gap12_percentiles_by_correctness": {
            str(correct): _percentiles(sub["abs_gap12"].to_numpy())
            for correct, sub in df.groupby("pre_tiebreak_correct")
        },
        "abs_gap12_percentiles_by_failure_location": {
            str(loc): _percentiles(sub["abs_gap12"].to_numpy())
            for loc, sub in df.groupby("failure_location")
        },
        "abs_gap12_percentiles_by_periodicity": {
            str(b): _percentiles(sub["abs_gap12"].to_numpy())
            for b, sub in df.groupby("periodicity_bucket")
        },
    }
    # "Risk floor": the smallest top1/top2 gap among pairs that are ALREADY
    # correct today - any threshold at or above this touches at least one
    # currently-correct pair's ranked list (whether the tiebreak actually
    # FLIPS the winner also depends on which of the two is closer to center).
    correct = df[df["pre_tiebreak_correct"]]
    if len(correct):
        floor_row = correct.loc[correct["abs_gap12"].idxmin()]
        out["risk_floor"] = {
            "abs_gap12": float(floor_row["abs_gap12"]), "pair_id": floor_row["pair_id"],
            "split": floor_row["split"], "structural_family": floor_row["structural_family"],
        }
    # "Opportunity window": gap distribution specifically for genuine_ambiguity
    # failures (GT's own candidate present, narrowly outscored) - these are
    # the only failures a center-tiebreak could plausibly rescue (a
    # candidate_generation failure has no GT candidate in the pool at all;
    # this tiebreak reorders EXISTING candidates, it cannot invent a new one).
    amb = df[df["failure_location"] == "genuine_ambiguity"]
    out["genuine_ambiguity_count"] = len(amb)
    if len(amb):
        out["genuine_ambiguity_abs_gap12"] = sorted(float(v) for v in amb["abs_gap12"])
        out["genuine_ambiguity_pair_ids"] = amb["pair_id"].tolist()

    # Threshold sweep: how many pairs fire, split by whether they're
    # currently correct (risk) or a genuine_ambiguity failure (opportunity).
    sweep_table = []
    for t in THRESHOLD_SWEEP:
        col = f"fires_at_{t:g}"
        fires = df[df[col]]
        sweep_table.append({
            "threshold": t, "n_fires": int(len(fires)),
            "n_fires_currently_correct_at_risk": int(fires["pre_tiebreak_correct"].sum()),
            "n_fires_genuine_ambiguity_opportunity": int((fires["failure_location"] == "genuine_ambiguity").sum()),
            "n_fires_candidate_generation": int((fires["failure_location"] == "candidate_generation").sum()),
            "n_fires_candidate_ranking": int((fires["failure_location"] == "candidate_ranking").sum()),
            "fires_pair_ids": fires["pair_id"].tolist(),
        })
    out["threshold_sweep"] = sweep_table
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd.concat([load_manifest(os.path.join(PROJECT_ROOT, "data"), s) for s in ALL_SPLITS],
                          ignore_index=True)
    print(f"Investigating score gaps over {len(manifest)} pairs across splits: {ALL_SPLITS}")

    records = []
    for i, (_, row) in enumerate(manifest.iterrows()):
        rec = investigate_pair(row)
        records.append(rec)
        print(f"  [{i + 1}/{len(manifest)}] {rec['pair_id']:28s} top1={rec['top1_score']:.4f} "
              f"gap12={rec['abs_gap12']:.5f} correct={rec['pre_tiebreak_correct']} "
              f"failure={rec['failure_location']}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "score_gap_investigation.csv"), index=False)

    summary = summarize(df)
    with open(os.path.join(OUT_DIR, "score_gap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT_DIR}/score_gap_investigation.csv and score_gap_summary.json")


if __name__ == "__main__":
    main()
