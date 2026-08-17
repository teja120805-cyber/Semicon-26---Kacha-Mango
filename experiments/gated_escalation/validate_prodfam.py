"""Validate decisiveness-gated escalation on production-family surfaces.

The variant measured +2.56pp on the frozen benchmark (5 rescued / 1 broken) but
failed gate criterion 7 — it had no independent validation, having been designed
after inspecting frozen failures. This supplies that validation, on datasets
generated from the **production FAMILIES table** at fresh seeds, so the family
composition matches what the change will actually face.

Runs the same rule as `frozen_decisive.py`, unchanged, and applies the
unmodified integration gate per dataset.

`cross_generator` has no generated analogue, so these carry 136 pairs. Gate
criterion 3 reads False when a split is absent — a known artifact already
documented for gate exception 2, not a real finding.

    python -m experiments.gated_escalation.validate_prodfam --dataset prodfam_a
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from evaluation import benchmark
from evaluation.evaluate import load_manifest
from pipeline.localize import AMBIGUITY_THRESHOLD, localize

from .frozen_decisive import escalate
from .run import dense_grid

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("development", "validation", "held_out", "challenge")
FACTOR = 3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    root = os.path.join(ROOT, "data", args.dataset)
    scales, rots = dense_grid(FACTOR)

    base_rows, cand_rows, decisions = [], [], []
    for split in SPLITS:
        for _, row in load_manifest(root, split).iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
            srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
            gt = (float(row["gt_x"]), float(row["gt_y"]))
            t0 = time.perf_counter()
            res = localize(ref, srch)
            t_base = time.perf_counter() - t0
            common = {"pair_id": row["pair_id"], "split": split,
                      "structural_family": row["structural_family"]}
            b = {**common, "error_px": float(np.hypot(res.x - gt[0], res.y - gt[1])),
                 "runtime_s": t_base, "ambiguous": bool(res.ambiguous)}
            base_rows.append(b)
            if not res.ambiguous:
                cand_rows.append({**b})
                continue
            t1 = time.perf_counter()
            x, y, amb, esc_gap = escalate(ref.astype(np.float32), srch.astype(np.float32),
                                           scales, rots)
            dt = time.perf_counter() - t1
            accept = esc_gap > res.psf_decisiveness
            err_esc = float(np.hypot(x - gt[0], y - gt[1]))
            decisions.append({"pair_id": row["pair_id"], "accepted": accept,
                               "err_prod": b["error_px"], "err_esc": err_esc})
            cand_rows.append({**common, "runtime_s": t_base + dt,
                               "error_px": err_esc if accept else b["error_px"],
                               "ambiguous": (amb >= AMBIGUITY_THRESHOLD) if accept else b["ambiguous"]})
            print(f"  [{split}] {row['pair_id']:30s} {b['error_px']:8.2f} -> "
                  f"{cand_rows[-1]['error_px']:8.2f} {'ACCEPT' if accept else 'reject'}", flush=True)

    b, c = pd.DataFrame(base_rows), pd.DataFrame(cand_rows)
    d = pd.DataFrame(decisions)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    b.to_csv(os.path.join(out, f"{args.dataset}_baseline.csv"), index=False)
    c.to_csv(os.path.join(out, f"{args.dataset}_escalated.csv"), index=False)

    ba, ca = float((b.error_px <= 5).mean()), float((c.error_px <= 5).mean())
    m = b[["pair_id", "error_px"]].merge(c[["pair_id", "error_px"]], on="pair_id",
                                          suffixes=("_b", "_c"))
    rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
    broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())
    print("\n" + "=" * 62)
    print(f"{args.dataset}  n={len(b)}  (production FAMILIES table, fresh seed)")
    print("=" * 62)
    print(f"  production         acc@5px = {ba:.4f}  catastrophic = {(b.error_px>50).mean():.4f}")
    print(f"  decisiveness-gated acc@5px = {ca:.4f}  catastrophic = {(c.error_px>50).mean():.4f}")
    print(f"  delta = {100*(ca-ba):+.2f} pp   rescued = {rescued}  broken = {broken}")
    if len(d):
        print(f"  accepted {int(d.accepted.sum())}/{len(d)} escalations ({d.accepted.mean():.0%})")
    print(f"  runtime {c.runtime_s.sum()/b.runtime_s.sum():.2f}x")
    print("\n  per split")
    for s in SPLITS:
        gb, gc = b[b.split == s], c[c.split == s]
        if len(gb):
            print(f"    {s:14s} n={len(gb):3d}  {float((gb.error_px<=5).mean()):.4f} -> "
                  f"{float((gc.error_px<=5).mean()):.4f}")
    gate = benchmark.run_integration_gate(b, c, seeds_agree=None)
    print("\n  gate criteria")
    for k, v in gate["criteria"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    for r in gate["per_family"]:
        if r["regressed"]:
            print(f"    regressed: {r['structural_family']} "
                  f"{r['baseline_accuracy_5px']:.3f} -> {r['candidate_accuracy_5px']:.3f} (n={r['n']})")
    with open(os.path.join(out, f"{args.dataset}_gate.json"), "w") as f:
        json.dump({"dataset": args.dataset, "n": len(b), "baseline_acc": ba,
                   "candidate_acc": ca, "rescued": rescued, "broken": broken,
                   "cat_before": float((b.error_px > 50).mean()),
                   "cat_after": float((c.error_px > 50).mean()),
                   "gate": gate}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
