"""How often is the true location even IN the candidate pool?

This bounds every re-scoring idea, P3 included. The diagnostic found that
14 of 19 failures across both tuning surfaces are *unreachable* - ground
truth is not within 5px of any pooled candidate - so no scoring function,
however good, can fix them. Before tuning a re-scorer further it is worth
knowing whether that ceiling moves when candidate generation is widened.

This matters because of a specific, documented gap in the project's
evidence. `experiments/wider_candidate_pool/` widened the pool (6 peaks,
4px NMS) and measured a **structural no-op**: bit-identical predictions,
because `ranking.rank_classical` is a pure arg-max and the global arg-max
is always among the per-hypothesis top-1 peaks. `reports/PROJECT_STATUS.md`
draws the right conclusion - "fixing periodicity needs a re-ranking stage
that looks past the top classical score, not more candidates under the
current ranker."

The corollary was never tested: **a wider pool and a re-scoring stage are
useless separately but may not be together.** Widening raises recall while
changing nothing; a re-scorer can only pick from what recall provides. This
script measures the first half of that - the recall ceiling - which decides
whether the second half is worth building.

Efficiency: peaks are extracted once per hypothesis at the maximum k, then
every smaller k is derived by truncation (greedy NMS is prefix-stable, so
the top-k for k < K is exactly the first k of the top-K list - verified in
`_assert_prefix_stable`). Only the NMS radius needs a separate pass.

    python -m experiments.discriminability_weighted.pool_recall --surface tune_degraded
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching
from pipeline.candidate_generation import Candidate
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

ROOT = os.path.dirname(os.path.abspath(__file__))
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(ROOT, "data", "tune_degraded"),
    "validate_fresh": os.path.join(ROOT, "data", "validate_fresh"),
}

MAX_PEAKS = 12
RADII = (8, 4)
K_VALUES = (2, 3, 4, 6, 8, 12)
HIT_PX = 5.0


def _assert_prefix_stable() -> None:
    """top_k_peaks is greedy max-then-suppress, so its output for k is the
    prefix of its output for any K > k. Asserted on a synthetic map rather
    than assumed, because the truncation trick below depends on it."""
    rng = np.random.default_rng(0)
    m = rng.random((200, 200)).astype(np.float32)
    long = matching.top_k_peaks(m, 12, 8)
    short = matching.top_k_peaks(m, 5, 8)
    assert long[:5] == short, "top_k_peaks is not prefix-stable; truncation invalid"


def peaks_by_hypothesis(reference, search, psf_sigma, radius):
    """Raw MAX_PEAKS peaks for every (scale, rotation), kept as candidates."""
    out = []
    for scale in candidate_generation.DEFAULT_SCALE_HYPOTHESES:
        for rot in candidate_generation.DEFAULT_ROTATION_HYPOTHESES:
            tmpl = matching.build_template(reference, scale, rot, psf_sigma)
            smap = matching.correlate(search, tmpl)
            peaks = matching.top_k_peaks(smap, MAX_PEAKS, radius)
            out.append([
                Candidate(x=px + tmpl.shape[1] / 2.0, y=py + tmpl.shape[0] / 2.0, score=s,
                          scale=scale, rotation_deg=rot, template_size=tmpl.shape[0])
                for px, py, s in peaks
            ])
    return out


def pool_for(per_hyp, k):
    flat = [c for peaks in per_hyp for c in peaks[:k]]
    return candidate_generation.deduplicate_by_location(flat)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    args = p.parse_args()

    _assert_prefix_stable()
    manifest = load_manifest(SURFACES[args.surface], "development")

    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = np.array([float(row["gt_x"]), float(row["gt_y"])])

        rec = {"pair_id": row["pair_id"], "structural_family": row["structural_family"]}
        arm_pools = {}
        for radius in RADII:
            for sigma in (0.0, PSF_MATCH_SIGMA):
                per_hyp = peaks_by_hypothesis(ref, search, sigma, radius)
                arm_pools[(radius, sigma)] = per_hyp

        for radius in RADII:
            # Reproduce production's arm choice at the production width
            # (k=2), so "selected arm" means what production would select.
            gaps = {}
            for sigma in (0.0, PSF_MATCH_SIGMA):
                gaps[sigma] = _decisiveness(pool_for(arm_pools[(radius, sigma)], 2))
            sel_sigma = max((0.0, PSF_MATCH_SIGMA), key=lambda s: gaps[s])

            for k in K_VALUES:
                sel = pool_for(arm_pools[(radius, sel_sigma)], k)
                uni = candidate_generation.deduplicate_by_location(
                    [c for s in (0.0, PSF_MATCH_SIGMA) for c in pool_for(arm_pools[(radius, s)], k)])
                d_sel = min(np.hypot(c.x - gt[0], c.y - gt[1]) for c in sel)
                d_uni = min(np.hypot(c.x - gt[0], c.y - gt[1]) for c in uni)
                rec[f"r{radius}_k{k}_sel_hit"] = bool(d_sel <= HIT_PX)
                rec[f"r{radius}_k{k}_uni_hit"] = bool(d_uni <= HIT_PX)
                rec[f"r{radius}_k{k}_sel_n"] = len(sel)
                rec[f"r{radius}_k{k}_uni_n"] = len(uni)
                if k == 2 and radius == 8:
                    rec["prod_pool_n"] = len(sel)
                    rec["prod_hit"] = bool(d_sel <= HIT_PX)
                    rec["prod_dist_px"] = float(d_sel)
        rows.append(rec)
        print(f"  {row['pair_id']:32s} prod_hit={rec['prod_hit']} "
              f"r8k12_sel={rec['r8_k12_sel_hit']} r4k12_uni={rec['r4_k12_uni_hit']} "
              f"(prod pool {rec['prod_pool_n']} -> {rec['r4_k12_uni_n']})", flush=True)

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{args.surface}_pool_recall.csv"), index=False)

    print("\n=========== POOL RECALL (fraction of pairs with GT within 5px) ===========")
    print(f"surface: {args.surface}   n={len(df)}   production recall="
          f"{df.prod_hit.mean():.4f}")
    print(f"{'radius':>7} {'k':>4} {'selected arm':>14} {'both arms':>12} {'mean pool size':>16}")
    summary = []
    for radius in RADII:
        for k in K_VALUES:
            s = float(df[f"r{radius}_k{k}_sel_hit"].mean())
            u = float(df[f"r{radius}_k{k}_uni_hit"].mean())
            n = float(df[f"r{radius}_k{k}_uni_n"].mean())
            summary.append({"radius": radius, "k": k, "sel_recall": s, "uni_recall": u,
                            "mean_pool_size": n})
            print(f"{radius:>7} {k:>4} {s:>14.4f} {u:>12.4f} {n:>16.1f}")
    with open(os.path.join(out_dir, f"{args.surface}_pool_recall.json"), "w") as f:
        json.dump({"surface": args.surface, "n": len(df),
                   "production_recall": float(df.prod_hit.mean()), "grid": summary}, f, indent=2)

    miss = df[~df.prod_hit]
    if len(miss):
        best = miss[f"r4_k12_uni_hit"].mean()
        print(f"\nOf the {len(miss)} pairs production misses, the widest pool "
              f"(r=4, k=12, both arms) recovers {int(miss['r4_k12_uni_hit'].sum())} "
              f"({best:.1%}) into reach of a re-scorer.")
        print(miss[["pair_id", "prod_dist_px", "r8_k12_sel_hit", "r4_k12_uni_hit"]].to_string(index=False))


if __name__ == "__main__":
    main()
