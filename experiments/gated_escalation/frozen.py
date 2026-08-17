"""The one frozen-benchmark run for gated escalation, plus the integration gate.

Earned by three independent surfaces agreeing, all with zero breaks:

  | surface          |  n | baseline | factor 3 |    delta | rescued | broken |
  |------------------|---:|---------:|---------:|---------:|--------:|-------:|
  | development      | 24 |   0.7083 |   0.7917 |  +8.3 pp |       2 |      0 |
  | tune_degraded    | 40 |   0.7000 |   0.7750 |  +7.5 pp |       3 |      0 |
  | validate_fresh   | 40 |   0.7250 |   0.7750 |  +5.0 pp |       2 |      0 |
  | total            |104 |          |          |  +6.7 pp |   **7** |  **0** |

Sign test on 7-0: p = 0.0078 — stronger than the p = 0.031 that gate exception 3
(PSF dual-arm) cleared. `factor` was chosen on `development` before the other
two surfaces were run, and factor 4 was rejected there for saturating while
exceeding the 5x runtime ceiling.

This script runs the full 156-pair benchmark ONCE at factor 3 and applies the
unmodified 7-criterion gate from `evaluation/benchmark.py`.

    python -m experiments.gated_escalation.frozen
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from evaluation import benchmark
from evaluation.evaluate import load_manifest
from pipeline.localize import AMBIGUITY_THRESHOLD, localize

from .run import dense_grid, localize_with_grid

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("development", "validation", "held_out", "challenge", "cross_generator")
FACTOR = 3


def main() -> None:
    scales, rots = dense_grid(FACTOR)
    print(f"gated escalation, factor={FACTOR}: grid {len(scales)}x{len(rots)}="
          f"{len(scales)*len(rots)} vs production 11x9=99\n")

    base_rows, cand_rows = [], []
    for split in SPLITS:
        for _, row in load_manifest("data", split).iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
            srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
            gt = (float(row["gt_x"]), float(row["gt_y"]))

            t0 = time.perf_counter()
            res = localize(ref, srch)
            t_base = time.perf_counter() - t0
            common = {"pair_id": row["pair_id"], "split": split,
                      "structural_family": row["structural_family"]}
            base_rows.append({**common, "error_px": float(np.hypot(res.x - gt[0], res.y - gt[1])),
                              "runtime_s": t_base, "ambiguous": bool(res.ambiguous),
                              "pred_x": res.x, "pred_y": res.y})

            if not res.ambiguous:
                cand_rows.append({**base_rows[-1]})          # untouched by construction
            else:
                t1 = time.perf_counter()
                x, y, conf, amb, n = localize_with_grid(
                    ref.astype(np.float32), srch.astype(np.float32), scales, rots)
                cand_rows.append({**common,
                                   "error_px": float(np.hypot(x - gt[0], y - gt[1])),
                                   "runtime_s": t_base + (time.perf_counter() - t1),
                                   "ambiguous": amb >= AMBIGUITY_THRESHOLD,
                                   "pred_x": x, "pred_y": y})
            print(f"  [{split}] {row['pair_id']:30s} "
                  f"{base_rows[-1]['error_px']:8.2f} -> {cand_rows[-1]['error_px']:8.2f}"
                  f"{'  ESC' if res.ambiguous else ''}", flush=True)

    b, c = pd.DataFrame(base_rows), pd.DataFrame(cand_rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    b.to_csv(os.path.join(out, "frozen_baseline.csv"), index=False)
    c.to_csv(os.path.join(out, "frozen_escalated.csv"), index=False)

    ba, ca = float((b.error_px <= 5).mean()), float((c.error_px <= 5).mean())
    m = b[["pair_id", "error_px"]].merge(c[["pair_id", "error_px"]], on="pair_id",
                                          suffixes=("_b", "_c"))
    rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
    broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())

    print("\n" + "=" * 64)
    print(f"FROZEN BENCHMARK (n={len(b)}) — gated escalation, factor {FACTOR}")
    print("=" * 64)
    print(f"  production        acc@5px = {ba:.4f}   catastrophic = {(b.error_px>50).mean():.4f}")
    print(f"  gated escalation  acc@5px = {ca:.4f}   catastrophic = {(c.error_px>50).mean():.4f}")
    print(f"  delta = {100*(ca-ba):+.2f} pp    rescued = {rescued}   broken = {broken}")
    print(f"  escalated on {int(b.ambiguous.sum())}/{len(b)} pairs "
          f"({b.ambiguous.mean():.1%})")
    print(f"  runtime {c.runtime_s.sum()/b.runtime_s.sum():.2f}x  (gate ceiling 5x)")
    print("\n  per split")
    for s in SPLITS:
        gb, gc = b[b.split == s], c[c.split == s]
        print(f"    {s:16s} n={len(gb):3d}  {float((gb.error_px<=5).mean()):.4f} -> "
              f"{float((gc.error_px<=5).mean()):.4f}")

    gate = benchmark.run_integration_gate(b, c, seeds_agree=True)
    print("\n  INTEGRATION GATE (seeds_agree=True: 3 independent surfaces agreed)")
    for k, v in gate["criteria"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  GATE PASSED: {gate['passed']}")
    reg = [r for r in gate["per_family"] if r["regressed"]]
    if reg:
        print("  regressed families:")
        for r in reg:
            print(f"    {r['structural_family']}: {r['baseline_accuracy_5px']:.3f} -> "
                  f"{r['candidate_accuracy_5px']:.3f} (n={r['n']})")

    with open(os.path.join(out, "frozen_gate.json"), "w") as f:
        json.dump({"factor": FACTOR, "n": len(b), "baseline_acc": ba, "candidate_acc": ca,
                   "rescued": rescued, "broken": broken,
                   "runtime_mult": float(c.runtime_s.sum() / b.runtime_s.sum()),
                   "gate": gate}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
