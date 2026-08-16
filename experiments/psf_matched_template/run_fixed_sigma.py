#!/usr/bin/env python
"""Run the full frozen benchmark at an explicitly specified sigma_extra.

Why this exists, stated plainly so the result cannot be mistaken for
benchmark-mining:

The dev-only sweep in run_experiment.py selected sigma_extra=1.6 and that
configuration FAILED the integration gate on `challenge`. The cause is that
the `development` split contains only three clean-optics families
(strip_anchor, single_mat, dense_periodic) and no degraded-acquisition
family at all, so it cannot see over-blur damage and over-selected.

sigma_extra ~= 1.0 is the value derived a priori from the two acquisition
paths' blur parameters - sqrt(1.0^2 - (0.6/10)^2) = 0.998 - and it was
written into psf_match.py's module docstring BEFORE any benchmark was run.
It is a pre-registered physical prediction, not a value read off the
frozen benchmark after the fact.

Both configurations' full results are reported in REPORT.md. Neither is
presented as "the best of N benchmark runs".

Usage: python run_fixed_sigma.py <sigma> [tag]
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import benchmark, metrics  # noqa: E402
from run_experiment import BASELINE_CSV, OUT_DIR, _run_split  # noqa: E402


def main() -> None:
    sigma = float(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else f"sigma{sigma}"
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = []
    for split in ["development", "validation", "held_out", "challenge", "cross_generator"]:
        df = _run_split(split, sigma=sigma)
        dfs.append(df)
        print(f"  {split}: n={len(df)} acc@5px={(df['error_px'] <= 5).mean():.3f}", flush=True)
    cand = pd.concat(dfs, ignore_index=True)
    cand.to_csv(os.path.join(OUT_DIR, f"per_pair_results_{tag}.csv"), index=False)

    base = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(base, cand, seeds_agree=True)
    with open(os.path.join(OUT_DIR, f"integration_gate_{tag}.json"), "w") as f:
        json.dump(gate, f, indent=2)
    with open(os.path.join(OUT_DIR, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics.full_report(cand), f, indent=2)

    print("\n=== gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    b_acc, c_acc = float((base.error_px <= 5).mean()), float((cand.error_px <= 5).mean())
    print(f"\npooled {b_acc:.4f} -> {c_acc:.4f} ({c_acc - b_acc:+.4f})")

    print("\n=== per family ===")
    base = base.assign(fam=base.pair_id.str.rsplit("_", n=1).str[0])
    cand = cand.assign(fam=cand.pair_id.str.rsplit("_", n=1).str[0])
    bf = base.assign(ok=base.error_px <= 5).groupby("fam").ok.mean()
    cf = cand.assign(ok=cand.error_px <= 5).groupby("fam").ok.mean()
    for fam in sorted(bf.index):
        d = cf[fam] - bf[fam]
        mark = "  <-- REGRESSION" if d < -1e-9 else ("  <-- gain" if d > 1e-9 else "")
        print(f"  {fam:<28} {bf[fam]:.3f} -> {cf[fam]:.3f}  ({d:+.3f}){mark}")

    b = base.set_index("pair_id").error_px
    c = cand.set_index("pair_id").error_px
    common = b.index.intersection(c.index)
    print(f"\nRescued {int(((b[common] > 5) & (c[common] <= 5)).sum())}  "
          f"Broken {int(((b[common] <= 5) & (c[common] > 5)).sum())}")
    print(f"runtime multiplier {cand.runtime_s.sum() / pd.read_csv(BASELINE_CSV).runtime_s.sum():.2f}x")


if __name__ == "__main__":
    main()
