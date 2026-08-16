#!/usr/bin/env python
"""Evaluate per-pair template-selection rules offline, from the dual-pass CSV.

run_dual_pass.py already paid the expensive cost (both pools, every pair).
Every rule below is pure post-processing over that CSV, so comparing rules
costs no extra compute and - importantly - no extra frozen-benchmark runs.

Rules considered (all ground-truth-free):
  always_0      - the production pipeline
  always_1.6    - psf_matched_template's fixed sigma
  max_gap       - pick the arm with the larger absolute top-vs-runner-up gap
  max_relgap    - same, normalized by the top score (scale-invariant, so it
                  is not biased by blurred templates scoring higher overall)
  max_top       - pick the arm with the higher top score (a naive control -
                  expected to fail, since blur raises scores uniformly)
  oracle        - pick whichever arm is correct (upper bound, not a method)

The rule to propose is chosen on the PRODUCTION seed and then confirmed on
the SECOND seed. Reporting every rule's number on both seeds, rather than
only the winner's, is what keeps that from being another benchmark look.

Usage: python evaluate_rules.py [production|second|both]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
A, B = "s0.0", "s1.6"


def pick(df: pd.DataFrame, rule: str) -> pd.Series:
    """Boolean Series: True => use the sigma=1.6 arm for that pair."""
    if rule == "always_0":
        return pd.Series(False, index=df.index)
    if rule == "always_1.6":
        return pd.Series(True, index=df.index)
    if rule == "max_gap":
        return df[f"{B}_gap"].fillna(-np.inf) > df[f"{A}_gap"].fillna(-np.inf)
    if rule == "max_relgap":
        return df[f"{B}_relgap"].fillna(-np.inf) > df[f"{A}_relgap"].fillna(-np.inf)
    if rule == "max_top":
        return df[f"{B}_top"].fillna(-np.inf) > df[f"{A}_top"].fillna(-np.inf)
    raise ValueError(rule)


def err_under(df: pd.DataFrame, use_b: pd.Series) -> pd.Series:
    return np.where(use_b, df[f"{B}_err"], df[f"{A}_err"])


RULES = ["always_0", "always_1.6", "max_gap", "max_relgap", "max_top"]


def evaluate(df: pd.DataFrame, label: str) -> dict:
    print(f"\n{'=' * 72}\n{label}  (n={len(df)})\n{'=' * 72}")
    base_err = df[f"{A}_err"]
    base_acc = float((base_err <= 5).mean())
    out = {"n": int(len(df)), "baseline_acc": base_acc, "rules": {}}

    print(f"{'rule':<14} {'acc@5px':>9} {'delta':>9} {'resc':>5} {'brk':>5} {'%used1.6':>9}")
    for rule in RULES:
        use_b = pick(df, rule)
        e = pd.Series(err_under(df, use_b), index=df.index)
        acc = float((e <= 5).mean())
        resc = int(((base_err > 5) & (e <= 5)).sum())
        brk = int(((base_err <= 5) & (e > 5)).sum())
        out["rules"][rule] = {"acc": acc, "delta": acc - base_acc, "rescued": resc,
                              "broken": brk, "frac_using_1.6": float(use_b.mean())}
        print(f"{rule:<14} {acc:>9.4f} {acc - base_acc:>+9.4f} {resc:>5} {brk:>5} {use_b.mean():>8.1%}")

    orc = float(((df[f"{A}_err"] <= 5) | (df[f"{B}_err"] <= 5)).mean())
    out["oracle_upper_bound"] = orc
    print(f"{'oracle':<14} {orc:>9.4f} {orc - base_acc:>+9.4f}   <- upper bound, not a method")

    best = max((r for r in RULES if r not in ("always_0",)),
               key=lambda r: out["rules"][r]["acc"])
    print(f"\nbest non-trivial rule: {best}")

    print("\nper-family, best rule vs baseline:")
    use_b = pick(df, best)
    e = pd.Series(err_under(df, use_b), index=df.index)
    fam = pd.DataFrame({"family": df.family, "b": base_err <= 5, "c": e <= 5})
    per = fam.groupby("family").agg(n=("b", "size"), base=("b", "mean"), cand=("c", "mean"))
    per["delta"] = per.cand - per.base
    out["per_family_best_rule"] = {k: {"n": int(v.n), "base": float(v.base),
                                       "cand": float(v.cand), "delta": float(v.delta)}
                                   for k, v in per.iterrows()}
    for f, v in per.sort_values("delta").iterrows():
        mark = "  <-- REGRESSION" if v.delta < -1e-9 else ("  <-- gain" if v.delta > 1e-9 else "")
        print(f"  {f:<28} {v.base:.3f} -> {v.cand:.3f} ({v.delta:+.3f}){mark}")
    out["best_rule"] = best
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    todo = ["production", "second"] if which == "both" else [which]
    summary = {}
    for w in todo:
        path = os.path.join(OUT_DIR, f"dual_pass_{w}.csv")
        if not os.path.exists(path):
            print(f"(missing {path} - run run_dual_pass.py {w})")
            continue
        summary[w] = evaluate(pd.read_csv(path), f"seed: {w}")
    with open(os.path.join(OUT_DIR, "rule_evaluation.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
