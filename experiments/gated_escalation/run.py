"""Confidence-gated escalation: spend more compute only where the pipeline
already knows it is unsure.

Every one of the thirteen prior attempts applied its change to **all** pairs,
which forces a global runtime budget and means any per-pair cost increase must
be paid 156 times. This inverts that: production runs unchanged, and a more
expensive second pass runs **only on pairs production flags `ambiguous`**.

Two things make this newly possible, both from this session:

  1. The ambiguity flag was recalibrated (0.92 -> 0.990, gate exception 4). At
     0.92 it fired on 82% of pairs, so gating on it saved almost nothing. It
     now fires on **35.3%**, which is what makes a 4x-cost second pass
     affordable: 1 + 3(0.353) = **2.06x total runtime**, against the gate's 5x
     ceiling.
  2. `experiments/reachability_verification/` identified exactly which pairs a
     second pass could possibly help.

**What to escalate to.** A denser scale/rotation hypothesis grid. This is not a
new idea — `experiments/finer_hypothesis_grid/` and `finer_grid_validation/`
doubled the grid density, measured net rescue +16/+6 on two independent sets,
and **were integrated**. It is the one structural change in this project with a
clear positive result. Density was never pushed further for one reason:
**global runtime**. Gating removes that constraint.

**Measured ceiling, before running.** Of the 35 frozen-benchmark failures, 30
are flagged and **19 of those are reachable** (true location already in the
pool). That caps this at +12.2pp. Nine of the 19 have the truth at **rank 2**,
median deficit 0.0165 ZNCC.

**Why this is not the rejected `center_tiebreak_v2`.** That experiment widened
the tie-break epsilon so a heuristic could pick the runner-up, and let through a
497px catastrophic regression. This changes no ranking rule: it recomputes the
candidate pool at finer granularity so the true location can win **on its own
score**. If it still loses, nothing is overridden.

**Risk, stated up front.** 25 currently-CORRECT pairs are also flagged, so
escalation runs on them too and could break them. That is the failure mode to
watch, and it is why the null control matters.

Null control: `--factor 1` escalates with the production grid, which must
reproduce production bit-for-bit.

    python -m experiments.gated_escalation.run --surface development
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, feature_extraction, ranking, refinement
from pipeline.localize import (AMBIGUITY_THRESHOLD, PSF_MATCH_SIGMA, _decisiveness,
                                localize)

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}


def dense_grid(factor: int):
    """Same span as production, `factor`x the density. factor=1 returns the
    production tuples unchanged (object-identical values), which is what makes
    the null control exact."""
    s = candidate_generation.DEFAULT_SCALE_HYPOTHESES
    r = candidate_generation.DEFAULT_ROTATION_HYPOTHESES
    if factor <= 1:
        return s, r
    def densify(vals):
        vals = list(vals)
        out = []
        for i in range(len(vals) - 1):
            for j in range(factor):
                out.append(vals[i] + (vals[i + 1] - vals[i]) * j / factor)
        out.append(vals[-1])
        return tuple(round(v, 6) for v in out)
    return densify(s), densify(r)


def localize_with_grid(ref, srch, scales, rots):
    best = None
    for sigma in (0.0, PSF_MATCH_SIGMA):
        pool = candidate_generation.deduplicate_by_location(
            candidate_generation.build_candidate_pool(
                ref, srch, scale_hypotheses=scales, rotation_hypotheses=rots, psf_sigma=sigma))
        gap = _decisiveness(pool)
        if best is None or gap > best[0]:
            best = (gap, sigma, pool)
    _, sigma, pool = best
    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(pool), srch.shape)
    x, y = refinement.refine(ref, srch, ranked[0], sigma)
    amb = feature_extraction.ambiguity_ratio(sorted((c.score for c in pool), reverse=True))
    return x, y, float(ranked[0].score), amb, len(pool)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="development", choices=sorted(SURFACES))
    p.add_argument("--factors", default="1,2,3,4")
    args = p.parse_args()

    manifest = load_manifest(SURFACES[args.surface], "development")
    pairs = [(r, cv2.imread(r["reference_path"], cv2.IMREAD_UNCHANGED),
              cv2.imread(r["search_path"], cv2.IMREAD_UNCHANGED)) for _, r in manifest.iterrows()]

    # Pass 1: production, unmodified. Determines the gate AND the baseline.
    base = []
    for row, ref, srch in pairs:
        t0 = time.perf_counter()
        res = localize(ref, srch)
        base.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                      "x": res.x, "y": res.y, "ambiguous": bool(res.ambiguous),
                      "runtime_s": time.perf_counter() - t0,
                      "error_px": float(np.hypot(res.x - row["gt_x"], res.y - row["gt_y"]))})
    bdf = pd.DataFrame(base)
    base_acc = float((bdf.error_px <= 5).mean())
    flagged = bdf.ambiguous.values
    print(f"[{args.surface}] n={len(bdf)}  production acc@5px={base_acc:.4f}  "
          f"flagged={flagged.sum()}/{len(bdf)} ({flagged.mean():.1%})\n")

    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    bdf.to_csv(os.path.join(out, f"{args.surface}_baseline.csv"), index=False)

    results = []
    for factor in [int(f) for f in args.factors.split(",")]:
        scales, rots = dense_grid(factor)
        rows = []
        for (row, ref, srch), b in zip(pairs, base):
            if not b["ambiguous"]:
                rows.append({**b})                      # untouched, by construction
                continue
            t0 = time.perf_counter()
            x, y, conf, amb, n = localize_with_grid(ref.astype(np.float32),
                                                     srch.astype(np.float32), scales, rots)
            rows.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                          "x": x, "y": y, "ambiguous": amb >= AMBIGUITY_THRESHOLD,
                          "runtime_s": b["runtime_s"] + (time.perf_counter() - t0),
                          "error_px": float(np.hypot(x - row["gt_x"], y - row["gt_y"]))})
        df = pd.DataFrame(rows)
        m = bdf[["pair_id", "error_px"]].merge(df[["pair_id", "error_px"]], on="pair_id",
                                                suffixes=("_b", "_c"))
        rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
        broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())
        acc = float((df.error_px <= 5).mean())
        identical = bool((bdf.x.values == df.x.values).all() and (bdf.y.values == df.y.values).all())
        rec = {"factor": factor, "n_scales": len(scales), "n_rotations": len(rots),
               "acc_5px": acc, "delta_pp": 100 * (acc - base_acc), "rescued": rescued,
               "broken": broken, "net": rescued - broken,
               "runtime_mult": float(df.runtime_s.sum() / bdf.runtime_s.sum()),
               "bit_identical_to_production": identical}
        results.append(rec)
        df.to_csv(os.path.join(out, f"{args.surface}_factor{factor}.csv"), index=False)
        tag = "  <- NULL CONTROL" if factor == 1 else ""
        print(f"factor={factor}  grid {len(scales)}x{len(rots)}={len(scales)*len(rots):<4} "
              f"acc={acc:.4f} ({rec['delta_pp']:+5.1f}pp) R={rescued} B={broken} "
              f"net={rec['net']:+d} runtime={rec['runtime_mult']:.2f}x "
              f"identical={identical}{tag}", flush=True)
        if factor == 1 and not identical:
            print("  NULL CONTROL FAILED — escalating with the production grid changed "
                  "predictions. Everything below is confounded; stop and investigate.")

    with open(os.path.join(out, f"{args.surface}_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "n": len(bdf), "baseline_acc": base_acc,
                   "flag_rate": float(flagged.mean()), "configs": results}, f, indent=2)
    print("\n=== ranked ===")
    print(pd.DataFrame(results).sort_values(["net", "acc_5px"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
