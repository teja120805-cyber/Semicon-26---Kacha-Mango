"""Did `experiments/wider_candidate_pool/`'s "structural no-op" conclusion expire?

That experiment widened the candidate pool and measured bit-identical
predictions on all 132 gate-split pairs. Its stated mechanism:
`ranking.rank_classical` is a pure arg-max, and the global arg-max is always
among the per-hypothesis top-1 peaks, so retaining extra lower-scoring peaks
cannot change the winner. `reports/PROJECT_STATUS.md` records that as a
settled structural fact.

It is no longer true here: widening changes predictions on the degraded
tuning surface. The suspected cause is that the conclusion was measured
BEFORE the multiway centre tie-break was integrated (2026-08-15, gate
exception 2 in `reports/GATE_EXCEPTIONS.md`).
`ranking.apply_center_tiebreak`'s second tier fires when at least
`MULTIWAY_MIN_GROUP_SIZE` (3) candidates sit within
`MULTIWAY_TIE_SCORE_EPSILON` (0.005) of the top and their spread is under
`MULTIWAY_MAX_SPREAD_PX`. A wider pool retains more near-top candidates, so
it can reach that group size where the narrow pool could not — making pool
width matter through a path that did not exist when the no-op was measured.

The arg-max argument itself is still correct. What changed is that the
arg-max is no longer the last word.

This script tests that attribution directly rather than reasoning about it:
re-run the widening with the multiway tier disabled (min_group_size set
above any achievable group), and see whether the no-op returns.

    python -m experiments.wide_pool_rescoring.noop_expiry --surface tune_degraded
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from pipeline import ranking, refinement

from .run import SURFACES, build_cache, select_arm_and_pool

ROOT = os.path.dirname(os.path.abspath(__file__))
DISABLED = 10 ** 6  # a group size no pool can reach => multiway tier is off


def predict(entry, k: int, multiway_min_group: int):
    sigma, pool = select_arm_and_pool(entry, k)
    ranked = ranking.rank_classical(pool)
    ranked = ranking.apply_center_tiebreak(
        ranked, entry["search"].shape, multiway_min_group_size=multiway_min_group)
    x, y = refinement.refine(entry["ref"], entry["search"], ranked[0], sigma)
    gt = (float(entry["row"]["gt_x"]), float(entry["row"]["gt_y"]))
    return x, y, float(np.hypot(x - gt[0], y - gt[1])), len(pool)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    args = p.parse_args()

    cache = build_cache(SURFACES[args.surface], verbose=False)
    print(f"[{args.surface}] n={len(cache)}\n")

    prod = [predict(e, 2, ranking.MULTIWAY_MIN_GROUP_SIZE) for e in cache]
    prod_acc = float(np.mean([r[2] <= 5 for r in prod]))
    print(f"production (k=2, multiway ON) acc@5px = {prod_acc:.4f}\n")

    rows = []
    print(f"{'k':>4} {'multiway':>9} {'acc@5px':>9} {'changed':>8} {'rescued':>8} {'broken':>7} {'net':>5}")
    for multiway, label in ((ranking.MULTIWAY_MIN_GROUP_SIZE, "ON"), (DISABLED, "OFF")):
        # Baseline for the OFF arm is production-with-multiway-off, so the
        # comparison isolates widening rather than mixing in the tier change.
        base = prod if label == "ON" else [predict(e, 2, DISABLED) for e in cache]
        base_acc = float(np.mean([r[2] <= 5 for r in base]))
        for k in (2, 4, 8, 12):
            cur = [predict(e, k, multiway) for e in cache]
            changed = sum(1 for a, b in zip(base, cur) if (a[0], a[1]) != (b[0], b[1]))
            rescued = sum(1 for a, b in zip(base, cur) if a[2] > 5 >= b[2])
            broken = sum(1 for a, b in zip(base, cur) if a[2] <= 5 < b[2])
            acc = float(np.mean([r[2] <= 5 for r in cur]))
            rows.append({"k": k, "multiway": label, "acc_5px": acc, "base_acc": base_acc,
                          "changed": changed, "rescued": rescued, "broken": broken,
                          "net": rescued - broken,
                          "mean_pool": float(np.mean([r[3] for r in cur]))})
            print(f"{k:>4} {label:>9} {acc:>9.4f} {changed:>8} {rescued:>8} {broken:>7} "
                  f"{rescued - broken:>+5d}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{args.surface}_noop_expiry.csv"), index=False)

    off = df[(df.multiway == "OFF") & (df.k > 2)]
    on = df[(df.multiway == "ON") & (df.k > 2)]
    verdict = {
        "multiway_off_widening_is_noop": bool((off.changed == 0).all()),
        "multiway_on_widening_changes": int(on.changed.sum()),
        "multiway_on_net": int(on.net.sum()),
    }
    with open(os.path.join(out, f"{args.surface}_noop_expiry.json"), "w") as f:
        json.dump({"surface": args.surface, "rows": rows, "verdict": verdict}, f, indent=2)

    print("\n=== attribution ===")
    if verdict["multiway_off_widening_is_noop"]:
        print("With the multiway tier OFF, widening is a bit-exact no-op at every k.")
        print("With it ON, widening changes predictions. The no-op conclusion in")
        print("wider_candidate_pool/ was correct when measured and expired when the")
        print("multiway centre tie-break was integrated on 2026-08-15.")
    else:
        print("Widening changes predictions even with the multiway tier OFF -")
        print("the multiway tier is NOT the (only) cause. Investigate further before")
        print("claiming the no-op expired.")


if __name__ == "__main__":
    main()
