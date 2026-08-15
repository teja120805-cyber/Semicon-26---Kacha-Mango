#!/usr/bin/env python
"""Final gate-quality confirmation for the multiway_tiebreak_v1 A6
candidate, selected from sweep.py's grid search (see outputs/sweep_results.csv):

    tie_score_epsilon = 0.005, min_group_size = 3, max_spread_px = 200.0

Chosen as the middle of a stable plateau (epsilon 0.003-0.007 all produce
the IDENTICAL outcome: net_rescue=+1, break_count=0, catastrophic_new=0 -
see sweep_results.csv), not tuned to a single lucky value - same selection
discipline as center_tiebreak_v2's own epsilon derivation.

Exactly like center_tiebreak_v2/run_experiment.py: full pipeline run from
scratch (harness.py::instrumented_localize_full) for BOTH baseline (shipped
production rule: epsilon=1e-6, which the new mechanism's min_group_size=1
collapses to exactly) and candidate, over (1) the frozen benchmark's 4 gate
splits (n=132) and (2) a genuinely fresh, independently-seeded dataset
(seed=502187 - distinct from production 777001, center_tiebreak_v2's
647301, and scale_range_v1's 913442) using completely UNMODIFIED production
FAMILIES (no scale-range widening here - A6 and A2 are independent
compliance items, this experiment isolates A6 alone).
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import benchmark  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from generator.dataset_generator import FAMILIES, generate_dataset  # noqa: E402
from experiments.multiway_tiebreak_v1.harness import instrumented_localize_full  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
FRESH_DATA_DIR = os.path.join(EXP_DIR, "fresh_data")
FRESH_SEED = 502187
FRESH_SPLITS = ("validation", "held_out", "challenge")

GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")
BASELINE_EPSILON = 1e-6  # shipped production value; min_group_size=1 makes this identical to ranking.apply_center_tiebreak

# Selected candidate config - see module docstring.
CAND_EPSILON = 0.005
CAND_MIN_GROUP_SIZE = 3
CAND_MAX_SPREAD_PX = 200.0


def ensure_fresh_dataset() -> None:
    if os.path.isdir(FRESH_DATA_DIR) and all(
        os.path.isfile(os.path.join(FRESH_DATA_DIR, s, "ground_truth.json")) for s in FRESH_SPLITS
    ):
        print(f"Fresh dataset already present at {FRESH_DATA_DIR}, seed {FRESH_SEED} - skipping regeneration.")
        return
    print(f"Generating fresh dataset (unmodified FAMILIES, seed={FRESH_SEED}) -> {FRESH_DATA_DIR}")
    families = [f for f in FAMILIES if f["split"] in FRESH_SPLITS]
    generate_dataset(FRESH_DATA_DIR, seed=FRESH_SEED, families=families, only_splits=list(FRESH_SPLITS), verbose=False)


def load_pairs(data_root: str, splits: list[str]) -> pd.DataFrame:
    return pd.concat([load_manifest(data_root, s) for s in splits], ignore_index=True)


def run_variant(manifest: pd.DataFrame, epsilon: float, min_group_size: int, max_spread_px: float,
                 label: str) -> pd.DataFrame:
    records = []
    t0 = time.perf_counter()
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        diag = instrumented_localize_full(ref, search, row["gt_x"], row["gt_y"], tie_score_epsilon=epsilon,
                                           min_group_size=min_group_size, max_spread_px=max_spread_px)
        records.append({**row.to_dict(), **diag})
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(records)
    acc5 = (df["error_px"] <= 5).mean()
    n_fired = int(df["tiebreak_fired"].sum())
    n_changed = int(df["tiebreak_changed_winner"].sum())
    print(f"  [{label}] n={len(df)} acc@5px={acc5:.3f} fired={n_fired} changed_winner={n_changed} "
          f"wall_time={elapsed:.1f}s")
    return df


def rescue_break(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict:
    merged = baseline_df.merge(candidate_df, on="pair_id", suffixes=("_base", "_cand"))
    rescued = merged[(merged["error_px_base"] > 5) & (merged["error_px_cand"] <= 5)]
    broken = merged[(merged["error_px_base"] <= 5) & (merged["error_px_cand"] > 5)]
    cat_rescued = merged[(merged["error_px_base"] > 50) & (merged["error_px_cand"] <= 50)]
    cat_new = merged[(merged["error_px_base"] <= 50) & (merged["error_px_cand"] > 50)]
    return {
        "rescue_count": len(rescued), "break_count": len(broken), "net_rescue": len(rescued) - len(broken),
        "catastrophic_rescued": len(cat_rescued), "catastrophic_new": len(cat_new),
        "rescued_pair_ids": rescued["pair_id"].tolist(), "broken_pair_ids": broken["pair_id"].tolist(),
        "catastrophic_new_pair_ids": cat_new["pair_id"].tolist(),
    }


def compare_gate_criteria(gate_a: dict, gate_b: dict, shared_keys: list[str]) -> bool:
    return all(gate_a["criteria"][k] == gate_b["criteria"][k] for k in shared_keys)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_fresh_dataset()

    # ---- Frozen benchmark ----
    frozen_manifest = load_pairs(os.path.join(PROJECT_ROOT, "data"), list(GATE_SPLITS))
    print(f"\n=== FROZEN benchmark (n={len(frozen_manifest)}) ===")
    frozen_baseline = run_variant(frozen_manifest, BASELINE_EPSILON, 1, float("inf"), "frozen/baseline")

    sandbox_baseline_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results_SANDBOX_BASELINE.csv")
    integrity = {"checked": False}
    if os.path.isfile(sandbox_baseline_path):
        shipped = pd.read_csv(sandbox_baseline_path)
        shipped = shipped[shipped["split"].isin(GATE_SPLITS)][["pair_id", "error_px"]].rename(
            columns={"error_px": "error_px_shipped"})
        cmp = frozen_baseline[["pair_id", "error_px"]].merge(shipped, on="pair_id", how="inner")
        max_diff = float((cmp["error_px"] - cmp["error_px_shipped"]).abs().max()) if len(cmp) else float("nan")
        integrity = {"checked": True, "n_compared": len(cmp), "max_abs_error_px_diff": max_diff,
                     "matches_sandbox_baseline": bool(max_diff < 1e-6)}
        print(f"  Integrity check vs sandbox baseline CSV: n={len(cmp)} max|diff|={max_diff:.2e} "
              f"({'MATCHES' if integrity['matches_sandbox_baseline'] else 'DIVERGES FROM'})")

    frozen_candidate = run_variant(frozen_manifest, CAND_EPSILON, CAND_MIN_GROUP_SIZE, CAND_MAX_SPREAD_PX,
                                    "frozen/candidate")
    frozen_baseline.to_csv(os.path.join(OUT_DIR, "final_per_pair_frozen_baseline.csv"), index=False)
    frozen_candidate.to_csv(os.path.join(OUT_DIR, "final_per_pair_frozen_candidate.csv"), index=False)

    frozen_rb = rescue_break(frozen_baseline, frozen_candidate)
    print(f"  Frozen rescue/break: {frozen_rb}")
    frozen_gate = benchmark.run_integration_gate(frozen_baseline, frozen_candidate, seeds_agree=None)
    print(json.dumps(frozen_gate["criteria"], indent=4))

    # ---- Fresh, independently-seeded dataset ----
    print(f"\n=== FRESH dataset (seed={FRESH_SEED}) ===")
    fresh_manifest = load_pairs(FRESH_DATA_DIR, list(FRESH_SPLITS))
    fresh_baseline = run_variant(fresh_manifest, BASELINE_EPSILON, 1, float("inf"), "fresh/baseline")
    fresh_candidate = run_variant(fresh_manifest, CAND_EPSILON, CAND_MIN_GROUP_SIZE, CAND_MAX_SPREAD_PX,
                                   "fresh/candidate")
    fresh_baseline.to_csv(os.path.join(OUT_DIR, "final_per_pair_fresh_baseline.csv"), index=False)
    fresh_candidate.to_csv(os.path.join(OUT_DIR, "final_per_pair_fresh_candidate.csv"), index=False)

    fresh_rb = rescue_break(fresh_baseline, fresh_candidate)
    print(f"  Fresh rescue/break: {fresh_rb}")
    fresh_gate = benchmark.run_integration_gate(fresh_baseline, fresh_candidate, seeds_agree=None)
    print(json.dumps(fresh_gate["criteria"], indent=4))

    shared_keys = ["3_improves_or_ties_cross_generator" if False else "4_no_catastrophic_increase",
                   "5_no_per_family_regression", "6_acceptable_runtime"]
    # cross_generator criterion doesn't apply to fresh (no fresh analogue - external/fixed data)
    shared_keys = ["4_no_catastrophic_increase", "5_no_per_family_regression", "6_acceptable_runtime"]
    # validation/held_out improve-criteria only meaningful where the mechanism could plausibly fire;
    # still include them since both splits are present in both datasets here.
    shared_keys += ["1_improves_validation", "2_improves_held_out"]
    seeds_agree = compare_gate_criteria(frozen_gate, fresh_gate, shared_keys)
    frozen_gate_final = benchmark.run_integration_gate(frozen_baseline, frozen_candidate, seeds_agree=seeds_agree)
    fresh_gate_final = benchmark.run_integration_gate(fresh_baseline, fresh_candidate, seeds_agree=seeds_agree)

    print(f"\n=== FINAL (seeds_agree = {seeds_agree}) ===")
    print(f"  Frozen gate passed: {frozen_gate_final['passed']}")
    print(f"  Fresh gate passed:  {fresh_gate_final['passed']}")

    summary = {
        "integrity_check": integrity,
        "config": {"tie_score_epsilon": CAND_EPSILON, "min_group_size": CAND_MIN_GROUP_SIZE,
                    "max_spread_px": CAND_MAX_SPREAD_PX},
        "frozen_rescue_break": frozen_rb, "fresh_rescue_break": fresh_rb,
        "frozen_gate": frozen_gate_final, "fresh_gate": fresh_gate_final,
        "seeds_agree": seeds_agree, "fresh_seed": FRESH_SEED,
    }
    with open(os.path.join(OUT_DIR, "final_gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/final_gate_summary.json")


if __name__ == "__main__":
    main()
