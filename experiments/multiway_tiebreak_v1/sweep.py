#!/usr/bin/env python
"""Sweeps (tie_score_epsilon, min_group_size, max_spread_px) configurations
against the cached candidate pools (build_cache.py) to find a config that
rescues genuine multi-way periodicity ties without repeating either failure
mode of the rejected center_tiebreak_v2 experiment (isolated-pair harm,
catastrophic spatial outliers) - same "risk vs opportunity by threshold"
methodology as that experiment's REPORT.md section 2, extended with the two
new gating dimensions.

Baseline for every comparison is the SHIPPED production tie-break
(ranking.apply_center_tiebreak, epsilon=1e-6 - provably inert, 0 fires),
evaluated on this SAME cache the same way (min_group_size=1,
max_spread_px=inf collapses apply_multiway_tiebreak to the classical
production rule exactly, so a shared code path with the shipped mechanism
is used for the baseline too, not a hand-rolled copy).
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.multiway_tiebreak_v1.harness import evaluate_cached_pair  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
CACHE_PATH = os.path.join(OUT_DIR, "candidate_cache.json")

BASELINE_EPSILON = 1e-6  # shipped production value

# Sweep grid. Epsilon values chosen to span from the shipped inert floor up
# through the periodicity-pileup range reports/ACCURACY_FORENSICS.md
# documents (~0.002-0.007), plus a couple of larger values to see where it
# starts doing more harm than good. min_group_size=2 is included to
# reproduce (as a sanity check) roughly what the rejected v2 experiment
# tested; group>=3 is the actual new hypothesis.
EPSILONS = [1e-6, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02]
MIN_GROUP_SIZES = [2, 3, 4]
MAX_SPREADS = [200.0, 500.0, float("inf")]


def run_config(cache: list[dict], epsilon: float, min_group_size: int, max_spread_px: float) -> pd.DataFrame:
    rows = [
        evaluate_cached_pair(c, tie_score_epsilon=epsilon, min_group_size=min_group_size,
                              max_spread_px=max_spread_px, image_root=PROJECT_ROOT)
        for c in cache
    ]
    return pd.DataFrame(rows)


def summarize_vs_baseline(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict:
    merged = baseline_df.merge(candidate_df, on="pair_id", suffixes=("_base", "_cand"))
    rescued = merged[(merged["error_px_base"] > 5) & (merged["error_px_cand"] <= 5)]
    broken = merged[(merged["error_px_base"] <= 5) & (merged["error_px_cand"] > 5)]
    cat_rescued = merged[(merged["error_px_base"] > 50) & (merged["error_px_cand"] <= 50)]
    cat_new = merged[(merged["error_px_base"] <= 50) & (merged["error_px_cand"] > 50)]
    n_fired = int(candidate_df["tiebreak_fired"].sum())
    n_changed = int(candidate_df["tiebreak_changed_winner"].sum())
    return {
        "n_fired": n_fired, "n_changed_winner": n_changed,
        "rescue_count": len(rescued), "break_count": len(broken), "net_rescue": len(rescued) - len(broken),
        "catastrophic_rescued": len(cat_rescued), "catastrophic_new": len(cat_new),
        "rescued_pair_ids": rescued["pair_id"].tolist(), "broken_pair_ids": broken["pair_id"].tolist(),
        "catastrophic_new_pair_ids": cat_new["pair_id"].tolist(),
        "acc5_baseline": float((baseline_df["error_px"] <= 5).mean()),
        "acc5_candidate": float((candidate_df["error_px"] <= 5).mean()),
    }


def main() -> None:
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Loaded cache: {len(cache)} pairs")

    print("Computing baseline (shipped production rule, epsilon=1e-6, min_group_size=1)...")
    baseline_df = run_config(cache, BASELINE_EPSILON, min_group_size=1, max_spread_px=float("inf"))
    baseline_df.to_csv(os.path.join(OUT_DIR, "sweep_baseline.csv"), index=False)
    print(f"  baseline acc@5px = {(baseline_df['error_px'] <= 5).mean():.4f}  "
          f"fires={int(baseline_df['tiebreak_fired'].sum())}")

    results = []
    for eps in EPSILONS:
        for mgs in MIN_GROUP_SIZES:
            for spread in MAX_SPREADS:
                cand_df = run_config(cache, eps, mgs, spread)
                summ = summarize_vs_baseline(baseline_df, cand_df)
                summ.update({"epsilon": eps, "min_group_size": mgs, "max_spread_px": spread})
                results.append(summ)
                print(f"  eps={eps:<8g} min_grp={mgs} spread={spread:<6g}  "
                      f"fired={summ['n_fired']:3d} changed={summ['n_changed_winner']:3d} "
                      f"rescue={summ['rescue_count']:2d} break={summ['break_count']:2d} "
                      f"net={summ['net_rescue']:+3d} cat_new={summ['catastrophic_new']}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "sweep_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/sweep_results.csv ({len(results_df)} configs)")

    # Candidates that are "safe" by construction: zero new catastrophic
    # failures AND strictly positive net rescue.
    safe = results_df[(results_df["catastrophic_new"] == 0) & (results_df["net_rescue"] > 0)]
    safe = safe.sort_values(["net_rescue", "n_changed_winner"], ascending=[False, True])
    print(f"\n{len(safe)} configs with zero new catastrophic failures and net_rescue > 0:")
    print(safe[["epsilon", "min_group_size", "max_spread_px", "n_fired", "n_changed_winner",
                "rescue_count", "break_count", "net_rescue"]].to_string(index=False))


if __name__ == "__main__":
    main()
