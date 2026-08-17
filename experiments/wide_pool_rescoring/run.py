"""Wider candidate pool + discriminability re-scoring, tested together.

Motivation - a gap in the project's existing evidence, not a new hunch:

  * `experiments/wider_candidate_pool/` widened the pool and measured a
    **structural no-op**: bit-identical predictions, because
    `ranking.rank_classical` is a pure arg-max and the global arg-max is
    always among the per-hypothesis top-1 peaks. Correct, and it means
    widening ALONE can never help.
  * `experiments/discriminability_weighted/` (P3) built a re-scorer that
    looks past the top classical score, then measured that on the
    PRODUCTION pool it has almost nothing to work with: 14 of 19 failures
    across both tuning surfaces are unreachable, i.e. ground truth is not
    within 5px of any pooled candidate, so no scoring function can fix them.
  * `experiments/discriminability_weighted/pool_recall.py` then measured
    that widening lifts recall on the degraded tuning surface from 0.750 to
    0.900 - it recovers 7 of the 10 pairs production misses.

So each half is individually useless for a reason the other half supplies.
This tests them together, which nothing has. The honest prior is still
negative: a wider pool also admits far more decoys (mean pool size 76 ->
230), and P3's own diagnostic found the weighted margin favours the decoy
on 4 of 5 reachable failures. Recall is a ceiling, not an outcome.

Null control: `peaks_per_hypothesis=2, alpha=0, tie_eps=0` is production.

    python -m experiments.wide_pool_rescoring.run --surface tune_degraded
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching
from pipeline.candidate_generation import Candidate
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

from experiments.discriminability_weighted.harness import finish_from_pool

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}

MAX_PEAKS = 12
NMS_RADIUS = candidate_generation.SUPPRESSION_RADIUS_PX


def peaks_by_hypothesis(reference, search, psf_sigma):
    out = []
    for scale in candidate_generation.DEFAULT_SCALE_HYPOTHESES:
        for rot in candidate_generation.DEFAULT_ROTATION_HYPOTHESES:
            tmpl = matching.build_template(reference, scale, rot, psf_sigma)
            smap = matching.correlate(search, tmpl)
            peaks = matching.top_k_peaks(smap, MAX_PEAKS, NMS_RADIUS)
            out.append([
                Candidate(x=px + tmpl.shape[1] / 2.0, y=py + tmpl.shape[0] / 2.0, score=s,
                          scale=scale, rotation_deg=rot, template_size=tmpl.shape[0])
                for px, py, s in peaks
            ])
    return out


def build_cache(data_root: str, verbose: bool = True):
    """One pass per pair captures MAX_PEAKS peaks per hypothesis for both PSF
    arms. Every narrower k is a prefix of that (greedy NMS is prefix-stable,
    asserted in pool_recall._assert_prefix_stable), so the whole k sweep is
    free after this."""
    manifest = load_manifest(data_root, "development")
    cache = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        t0 = time.perf_counter()
        arms = {s: peaks_by_hypothesis(ref, search, s) for s in (0.0, PSF_MATCH_SIGMA)}
        cache.append({"row": row, "ref": ref, "search": search, "arms": arms,
                       "pool_time_s": time.perf_counter() - t0})
        if verbose:
            print(f"  pooled {row['pair_id']}", flush=True)
    return cache


def select_arm_and_pool(entry, k: int):
    """Production's arm choice is made at production width (k=2) and then the
    chosen arm is widened. Deciding decisiveness on the WIDE pool instead
    would silently change the integrated PSF rule, which is a separate
    change and must not ride along inside this one."""
    gaps = {}
    pools2 = {}
    for sigma, per_hyp in entry["arms"].items():
        pools2[sigma] = candidate_generation.deduplicate_by_location(
            [c for peaks in per_hyp for c in peaks[:2]])
        gaps[sigma] = _decisiveness(pools2[sigma])
    sel = max((0.0, PSF_MATCH_SIGMA), key=lambda s: gaps[s])
    if k == 2:
        return sel, pools2[sel]
    wide = candidate_generation.deduplicate_by_location(
        [c for peaks in entry["arms"][sel] for c in peaks[:k]])
    return sel, wide


def run_config(cache, k, **cfg) -> pd.DataFrame:
    rows = []
    for entry in cache:
        row = entry["row"]
        sigma, pool = select_arm_and_pool(entry, k)
        res = finish_from_pool(entry["ref"], entry["search"], sigma, pool,
                               pool_time_s=entry["pool_time_s"], **cfg)
        gt = (float(row["gt_x"]), float(row["gt_y"]))
        rows.append({
            "pair_id": row["pair_id"], "structural_family": row["structural_family"],
            "error_px": float(np.hypot(res.x - gt[0], res.y - gt[1])),
            "pool_size": len(pool), "psf_sigma": sigma,
            "tie_group_size": res.tie_group_size, "rescored": res.rescored,
            "winner_changed": res.winner_changed, "runtime_s": res.runtime_s,
            "recall_hit": bool(min(np.hypot(c.x - gt[0], c.y - gt[1]) for c in pool) <= 5.0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    p.add_argument("--ks", default="2,4,8,12")
    p.add_argument("--alphas", default="0.5,1.0")
    p.add_argument("--tie-eps", default="0.01,0.02,0.05")
    p.add_argument("--schemes", default="lattice_shift,confuser_variance")
    p.add_argument("--max-groups", default="4,8,16")
    args = p.parse_args()

    print(f"[{args.surface}] caching wide peak lists")
    cache = build_cache(SURFACES[args.surface])
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    base = run_config(cache, 2, alpha=0.0, scheme="lattice_shift", tie_eps=0.0)
    base_acc = float((base.error_px <= 5).mean())
    base.to_csv(os.path.join(out_dir, f"{args.surface}_baseline.csv"), index=False)
    print(f"\nproduction baseline acc@5px = {base_acc:.4f} (n={len(base)}), "
          f"recall = {base.recall_hit.mean():.4f}\n")

    # Arg-max on a wide pool must reproduce production exactly - the
    # structural no-op wider_candidate_pool measured. Verified, not assumed:
    # if this ever differs, the widening itself changed behaviour and every
    # number below would be confounded.
    for k in (4, 8, 12):
        nop = run_config(cache, k, alpha=0.0, scheme="lattice_shift", tie_eps=0.0)
        same = bool((nop.error_px.values == base.error_px.values).all())
        print(f"  no-op check k={k:<3} argmax identical to production: {same} "
              f"(recall {base.recall_hit.mean():.3f} -> {nop.recall_hit.mean():.3f})")
        if not same:
            print("    NOTE: widening alone changed predictions - investigate before trusting below")

    results = []
    ks = [int(x) for x in args.ks.split(",")]
    for k, scheme, alpha, tie_eps, mg in itertools.product(
            ks, args.schemes.split(","), [float(a) for a in args.alphas.split(",")],
            [float(t) for t in args.tie_eps.split(",")], [int(m) for m in args.max_groups.split(",")]):
        df = run_config(cache, k, alpha=alpha, scheme=scheme, tie_eps=tie_eps, max_group=mg)
        merged = base[["pair_id", "error_px"]].merge(
            df[["pair_id", "error_px"]], on="pair_id", suffixes=("_b", "_c"))
        rescued = int(((merged.error_px_b > 5) & (merged.error_px_c <= 5)).sum())
        broken = int(((merged.error_px_b <= 5) & (merged.error_px_c > 5)).sum())
        acc = float((df.error_px <= 5).mean())
        rec = {"k": k, "scheme": scheme, "alpha": alpha, "tie_eps": tie_eps, "max_group": mg,
               "acc_5px": acc, "delta_pp": 100 * (acc - base_acc), "rescued": rescued,
               "broken": broken, "net": rescued - broken, "recall": float(df.recall_hit.mean()),
               "mean_group": float(df.tie_group_size.mean()),
               "n_winner_changed": int(df.winner_changed.sum())}
        results.append(rec)
        print(f"k={k:<3} {scheme:18s} a={alpha:<4} t={tie_eps:<5} mg={mg:<3} "
              f"acc={acc:.4f} ({rec['delta_pp']:+5.1f}pp) R={rescued} B={broken} "
              f"net={rec['net']:+d} recall={rec['recall']:.3f} grp={rec['mean_group']:.1f}",
              flush=True)
        df.to_csv(os.path.join(out_dir, f"{args.surface}_k{k}_{scheme}_a{alpha}_t{tie_eps}_mg{mg}.csv"),
                  index=False)

    sdf = pd.DataFrame(results).sort_values(["net", "acc_5px"], ascending=False)
    sdf.to_csv(os.path.join(out_dir, f"{args.surface}_summary.csv"), index=False)
    with open(os.path.join(out_dir, f"{args.surface}_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "baseline_acc": base_acc, "n": len(base),
                   "configs": results}, f, indent=2)
    print("\n=== top 12 by net rescue ===")
    print(sdf.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
