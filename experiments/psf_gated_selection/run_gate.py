#!/usr/bin/env python
"""Materialize a selection rule as a per-pair CSV in evaluation/evaluate.py's
schema and run the UNMODIFIED integration gate against the production
baseline.

Rule choice, pre-committed on principle rather than on score: `max_relgap`.
On the production seed max_gap, max_relgap and max_top all tie at 0.7756, so
the benchmark cannot discriminate between them and something else must.
Blurring raises ZNCC scores systematically (the blurred arm's top score is
higher on most pairs), so an ABSOLUTE gap comparison is biased toward the
blurred arm; normalizing by the top score removes that bias. max_top is
rejected outright as unprincipled - it prefers whichever arm scores higher,
which blur guarantees. Both max_gap and max_relgap numbers are reported
everywhere so the choice can be audited.

Usage: python run_gate.py [rule]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import benchmark, metrics  # noqa: E402
from evaluate_rules import err_under, pick  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
RUNTIME_MULTIPLIER = 2.0   # two pool builds per pair, measured not assumed (see REPORT §5)


def main() -> None:
    rule = sys.argv[1] if len(sys.argv) > 1 else "max_relgap"
    d = pd.read_csv(os.path.join(OUT_DIR, "dual_pass_production.csv"))
    base = pd.read_csv(BASELINE_CSV)

    use_b = pick(d, rule)
    cand = base.copy().set_index("pair_id")
    sel = pd.DataFrame({
        "pair_id": d.pair_id,
        "pred_x": np.where(use_b, d["s1.6_x"], d["s0.0_x"]),
        "pred_y": np.where(use_b, d["s1.6_y"], d["s0.0_y"]),
        "error_px": err_under(d, use_b),
        "confidence": np.where(use_b, d["s1.6_top"], d["s0.0_top"]),
    }).set_index("pair_id")

    common = cand.index.intersection(sel.index)
    for col in ("pred_x", "pred_y", "error_px", "confidence"):
        cand.loc[common, col] = sel.loc[common, col]
    cand["runtime_s"] = cand["runtime_s"] * RUNTIME_MULTIPLIER
    cand["ranking_mode"] = f"psf_gated_{rule}"
    cand = cand.reset_index()
    cand.to_csv(os.path.join(OUT_DIR, f"per_pair_results_{rule}.csv"), index=False)

    gate = benchmark.run_integration_gate(base, cand, seeds_agree=True)
    json.dump(gate, open(os.path.join(OUT_DIR, f"integration_gate_{rule}.json"), "w"), indent=2)
    json.dump(metrics.full_report(cand), open(os.path.join(OUT_DIR, f"metrics_{rule}.json"), "w"), indent=2)

    print(f"=== rule: {rule} ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    ba, ca = float((base.error_px <= 5).mean()), float((cand.error_px <= 5).mean())
    print(f"\npooled {ba:.4f} -> {ca:.4f}  ({ca - ba:+.4f})")
    print("\nper split:")
    for s in base.split.unique():
        print(f"  {s:<16} {(base[base.split == s].error_px <= 5).mean():.3f} -> "
              f"{(cand[cand.split == s].error_px <= 5).mean():.3f}")
    print("\ncatastrophic (>50px):")
    print(f"  baseline {int((base.error_px > 50).sum())}  candidate {int((cand.error_px > 50).sum())}")
    if "details" in gate:
        print("\ngate details:")
        print(json.dumps(gate["details"], indent=2)[:1800])


if __name__ == "__main__":
    main()
