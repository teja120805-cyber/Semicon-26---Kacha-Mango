"""Benchmark harness: compares two per-pair result sets (the classical
baseline vs. a candidate learned-ranking model) and applies the mandatory
integration gate from reports/V2_ARCHITECTURE_PLAN.md section 8.

If the gate fails, this script's output says so plainly - it never
modifies pipeline/ or model/ itself. Integrating a passing candidate is
always a separate, deliberate code change afterward, never automatic.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import pandas as pd

GATE_TOLERANCE_PX = 5.0
CATASTROPHIC_PX = 50.0
MAX_RUNTIME_MULTIPLIER = 5.0


def _acc(df: pd.DataFrame, tol: float = GATE_TOLERANCE_PX) -> float:
    return float((df["error_px"] <= tol).mean()) if len(df) else float("nan")


def _catastrophic_rate(df: pd.DataFrame) -> float:
    return float((df["error_px"] > CATASTROPHIC_PX).mean()) if len(df) else float("nan")


def compare_split(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, split: str) -> dict:
    b = baseline_df[baseline_df["split"] == split]
    c = candidate_df[candidate_df["split"] == split]
    if len(b) == 0 or len(c) == 0:
        return {"split": split, "n_baseline": len(b), "n_candidate": len(c)}
    b_acc, c_acc = _acc(b), _acc(c)
    b_cat, c_cat = _catastrophic_rate(b), _catastrophic_rate(c)
    return {
        "split": split, "n_baseline": len(b), "n_candidate": len(c),
        "baseline_accuracy_5px": b_acc, "candidate_accuracy_5px": c_acc, "improved": c_acc > b_acc,
        "baseline_catastrophic_rate": b_cat, "candidate_catastrophic_rate": c_cat,
        "catastrophic_regression": c_cat > b_cat,
        "baseline_mean_runtime_s": float(b["runtime_s"].mean()),
        "candidate_mean_runtime_s": float(c["runtime_s"].mean()),
    }


def per_family_regression_check(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> list[dict]:
    """Criterion 5: no regression on any currently-correct case class,
    checked per structural family rather than only pooled (a pooled-only
    check can hide a family that got worse behind others that improved)."""
    rows = []
    families = sorted(set(baseline_df["structural_family"]) | set(candidate_df["structural_family"]))
    for family in families:
        b = baseline_df[baseline_df["structural_family"] == family]
        c = candidate_df[candidate_df["structural_family"] == family]
        if len(b) == 0 or len(c) == 0:
            continue
        b_acc, c_acc = _acc(b), _acc(c)
        rows.append({
            "structural_family": family, "n": len(b),
            "baseline_accuracy_5px": b_acc, "candidate_accuracy_5px": c_acc,
            "regressed": c_acc < b_acc - 1e-9,
        })
    return rows


def run_integration_gate(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame,
                          seeds_agree: Optional[bool] = None) -> dict:
    """Applies every criterion in reports/V2_ARCHITECTURE_PLAN.md section 8.

    `seeds_agree`: the result of a separate multi-seed training stability
    check (criterion 7). None ("not checked") counts as a FAIL for that
    criterion - an unverified claim of stability is not the same as a
    verified one, and the gate does not give the benefit of the doubt.
    """
    candidate_splits = set(baseline_df["split"]) & set(candidate_df["split"])
    splits = [s for s in ("validation", "held_out", "challenge", "cross_generator") if s in candidate_splits]
    per_split = {s: compare_split(baseline_df, candidate_df, s) for s in splits}
    family_rows = per_family_regression_check(baseline_df, candidate_df)

    def _get(split: str, key: str, default=None):
        return per_split.get(split, {}).get(key, default)

    runtime_ok = all(
        _get(s, "candidate_mean_runtime_s", 0) <= _get(s, "baseline_mean_runtime_s", 0) * MAX_RUNTIME_MULTIPLIER
        for s in splits
    )

    criteria = {
        "1_improves_validation": bool(_get("validation", "improved", False)),
        "2_improves_held_out": bool(_get("held_out", "improved", False)),
        "3_improves_or_ties_cross_generator": (
            _get("cross_generator", "candidate_accuracy_5px", -1.0) >=
            _get("cross_generator", "baseline_accuracy_5px", 0.0)
            if "cross_generator" in per_split else False
        ),
        "4_no_catastrophic_increase": all(not _get(s, "catastrophic_regression", True) for s in splits),
        "5_no_per_family_regression": len(family_rows) > 0 and not any(r["regressed"] for r in family_rows),
        "6_acceptable_runtime": runtime_ok,
        "7_stable_across_seeds": bool(seeds_agree),
    }
    return {"passed": all(criteria.values()), "criteria": criteria, "per_split": per_split, "per_family": family_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the V2 model integration gate.")
    parser.add_argument("--baseline", default="outputs/reports/per_pair_results.csv")
    parser.add_argument("--candidate", default="outputs/reports/per_pair_results_learned.csv")
    parser.add_argument("--out", default="outputs/reports/integration_gate.json")
    parser.add_argument("--seeds-agree", choices=["true", "false", "unknown"], default="unknown")
    args = parser.parse_args()

    seeds_agree = {"true": True, "false": False, "unknown": None}[args.seeds_agree]
    result = run_integration_gate(pd.read_csv(args.baseline), pd.read_csv(args.candidate), seeds_agree=seeds_agree)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"passed": result["passed"], "criteria": result["criteria"]}, indent=2))
