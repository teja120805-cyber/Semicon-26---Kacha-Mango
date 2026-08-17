"""Decisiveness-gated escalation — the second and final frozen run for this line.

`frozen.py` measured plain gated escalation: whenever a pair is flagged
`ambiguous`, the denser-grid answer is accepted unconditionally. Frozen result:
+1.92pp but **8 rescued / 5 broken**, four of the breaks turning sub-pixel
answers into new catastrophic errors, and the gate failed criteria 2, 4 and 5.

This variant accepts the escalated answer **only when its candidate pool is more
decisive than production's** — the same score gap to the best candidate at a
genuinely different location (`localize._decisiveness`) that production already
uses to choose between the two PSF arms (gate exception 3). When the escalated
pool is less decisive, the production answer stands **byte-for-byte**.

Two properties worth stating plainly:

  * **No free parameters.** There is no threshold, no tuned constant, no
    strength knob. It is a strict comparison between two numbers, so there is
    nothing to overfit and no configuration to select.
  * **Disclosure.** The idea was prompted by inspecting which pairs `frozen.py`
    broke. It is not parameter mining — there are no parameters — but it is not
    a blind pre-registration either, and this is **frozen run #2** for this line
    of work. Both facts belong in any report of the result.

A known confound, recorded before running: the two pools differ in size (99 vs
775 hypotheses before deduplication), and a denser pool may systematically
produce more near-neighbours, which could bias the decisiveness comparison in
one direction. The acceptance rate is reported so this is visible rather than
hidden.

    python -m experiments.gated_escalation.frozen_decisive
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
from pipeline import candidate_generation, feature_extraction, ranking, refinement
from pipeline.localize import (AMBIGUITY_THRESHOLD, PSF_MATCH_SIGMA, _decisiveness,
                                localize)

from .run import dense_grid

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("development", "validation", "held_out", "challenge", "cross_generator")
FACTOR = 3


def escalate(ref, srch, scales, rots):
    """Dense-grid localization, also returning the pool's decisiveness so the
    caller can decide whether to accept it."""
    best = None
    for sigma in (0.0, PSF_MATCH_SIGMA):
        pool = candidate_generation.deduplicate_by_location(
            candidate_generation.build_candidate_pool(
                ref, srch, scale_hypotheses=scales, rotation_hypotheses=rots, psf_sigma=sigma))
        gap = _decisiveness(pool)
        if best is None or gap > best[0]:
            best = (gap, sigma, pool)
    gap, sigma, pool = best
    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(pool), srch.shape)
    x, y = refinement.refine(ref, srch, ranked[0], sigma)
    amb = feature_extraction.ambiguity_ratio(sorted((c.score for c in pool), reverse=True))
    return x, y, amb, gap


def main() -> None:
    scales, rots = dense_grid(FACTOR)
    print(f"decisiveness-gated escalation, factor={FACTOR} "
          f"({len(scales)}x{len(rots)}={len(scales)*len(rots)} vs 99)\n")

    base_rows, cand_rows, decisions = [], [], []
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
            b = {**common, "error_px": float(np.hypot(res.x - gt[0], res.y - gt[1])),
                 "runtime_s": t_base, "ambiguous": bool(res.ambiguous),
                 "pred_x": res.x, "pred_y": res.y}
            base_rows.append(b)

            if not res.ambiguous:
                cand_rows.append({**b})
                continue

            t1 = time.perf_counter()
            x, y, amb, esc_gap = escalate(ref.astype(np.float32), srch.astype(np.float32),
                                           scales, rots)
            dt = time.perf_counter() - t1
            accept = esc_gap > res.psf_decisiveness
            decisions.append({"pair_id": row["pair_id"], "prod_gap": res.psf_decisiveness,
                               "esc_gap": esc_gap, "accepted": accept,
                               "err_prod": b["error_px"],
                               "err_esc": float(np.hypot(x - gt[0], y - gt[1]))})
            if accept:
                cand_rows.append({**common,
                                   "error_px": float(np.hypot(x - gt[0], y - gt[1])),
                                   "runtime_s": t_base + dt,
                                   "ambiguous": amb >= AMBIGUITY_THRESHOLD,
                                   "pred_x": x, "pred_y": y})
            else:
                cand_rows.append({**b, "runtime_s": t_base + dt})   # production stands
            print(f"  [{split}] {row['pair_id']:30s} {b['error_px']:8.2f} -> "
                  f"{cand_rows[-1]['error_px']:8.2f}  gap {res.psf_decisiveness:.4f}->"
                  f"{esc_gap:.4f} {'ACCEPT' if accept else 'reject'}", flush=True)

    b, c = pd.DataFrame(base_rows), pd.DataFrame(cand_rows)
    d = pd.DataFrame(decisions)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    b.to_csv(os.path.join(out, "decisive_baseline.csv"), index=False)
    c.to_csv(os.path.join(out, "decisive_escalated.csv"), index=False)
    d.to_csv(os.path.join(out, "decisive_decisions.csv"), index=False)

    ba, ca = float((b.error_px <= 5).mean()), float((c.error_px <= 5).mean())
    m = b[["pair_id", "error_px"]].merge(c[["pair_id", "error_px"]], on="pair_id",
                                          suffixes=("_b", "_c"))
    rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
    broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())

    print("\n" + "=" * 66)
    print(f"FROZEN BENCHMARK (n={len(b)}) — DECISIVENESS-GATED escalation, factor {FACTOR}")
    print("=" * 66)
    print(f"  production            acc@5px = {ba:.4f}  catastrophic = {(b.error_px>50).mean():.4f}")
    print(f"  decisiveness-gated    acc@5px = {ca:.4f}  catastrophic = {(c.error_px>50).mean():.4f}")
    print(f"  delta = {100*(ca-ba):+.2f} pp   rescued = {rescued}  broken = {broken}")
    print(f"  flagged {int(b.ambiguous.sum())}, escalation ACCEPTED on "
          f"{int(d.accepted.sum())}/{len(d)} ({d.accepted.mean():.0%})")
    print(f"  runtime {c.runtime_s.sum()/b.runtime_s.sum():.2f}x  (ceiling 5x)")
    print("\n  vs plain gated escalation (frozen.py): +1.92pp, 8 rescued / 5 broken, cat 0.1667")

    print("\n  per split")
    for s in SPLITS:
        gb, gc = b[b.split == s], c[c.split == s]
        print(f"    {s:16s} n={len(gb):3d}  {float((gb.error_px<=5).mean()):.4f} -> "
              f"{float((gc.error_px<=5).mean()):.4f}")

    if len(d):
        acc_d = d[d.accepted]
        rej_d = d[~d.accepted]
        print(f"\n  among ACCEPTED ({len(acc_d)}): rescued "
              f"{int(((acc_d.err_prod>5)&(acc_d.err_esc<=5)).sum())}, broken "
              f"{int(((acc_d.err_prod<=5)&(acc_d.err_esc>5)).sum())}")
        print(f"  among REJECTED ({len(rej_d)}): escalation would have rescued "
              f"{int(((rej_d.err_prod>5)&(rej_d.err_esc<=5)).sum())}, broken "
              f"{int(((rej_d.err_prod<=5)&(rej_d.err_esc>5)).sum())}"
              "   <- what the gate saved / cost")

    gate = benchmark.run_integration_gate(b, c, seeds_agree=None)
    print("\n  INTEGRATION GATE (seeds_agree=None — this variant has NOT been")
    print("  validated on an independent surface, so criterion 7 correctly fails)")
    for k, v in gate["criteria"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  GATE PASSED: {gate['passed']}")
    for r in gate["per_family"]:
        if r["regressed"]:
            print(f"    regressed: {r['structural_family']} "
                  f"{r['baseline_accuracy_5px']:.3f} -> {r['candidate_accuracy_5px']:.3f} (n={r['n']})")

    with open(os.path.join(out, "decisive_gate.json"), "w") as f:
        json.dump({"factor": FACTOR, "n": len(b), "baseline_acc": ba, "candidate_acc": ca,
                   "rescued": rescued, "broken": broken,
                   "accept_rate": float(d.accepted.mean()) if len(d) else None,
                   "runtime_mult": float(c.runtime_s.sum() / b.runtime_s.sum()),
                   "gate": gate}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
