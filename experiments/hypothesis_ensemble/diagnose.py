"""Stop taking the max: is a robust cross-hypothesis aggregate less biased?

`candidate_generation.deduplicate_by_location` keeps the **highest-scoring**
detection at each location and discards the rest. But each location is detected
by many of the 81 scale x rotation hypotheses, so there is a whole score
*distribution* per location that is thrown away before ranking ever happens.

**Why the max is the wrong estimator.** The winner is the arg-max over ~160
correlated noisy scores, which suffers winner's curse: it is biased toward
whichever location happened to draw a lucky hypothesis. A decoy that aligns
well with one particular (scale, rotation) wins on its single best score, while
the true location should score highly across *many* hypotheses. A robust
aggregate — median, or a trimmed mean — is a less biased estimator of "how well
does this location actually match".

**Why this is not an already-rejected experiment.**
`experiments/cross_hypothesis_consensus_rerank/` re-ranked by *how many*
hypotheses found a location — a **count**, measured as a clean no-op, because
every genuine location is found by many hypotheses. This uses the **magnitude
distribution**, which nothing has touched. `prominence_rerank` compared a peak
against its spatial annulus, not against its own hypothesis ensemble.

**The count confound, handled explicitly.** A cluster's mean is confounded with
its size, and size is the statistic already shown to be a no-op. So a
count-matched variant is also computed: the mean of each cluster's **top-5**
scores only, which is size-independent by construction. If the plain mean helps
but top-5 does not, the effect is really count and the direction is closed.

**Bar (pre-registered, same as DDIS/OTSDF/low-frequency):** >70% preference for
truth on the 22 reachable failures. Chance is 50%.

    python -m experiments.hypothesis_ensemble.diagnose
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, ranking
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

ROOT = os.path.dirname(os.path.abspath(__file__))
REACH = os.path.join(os.path.dirname(ROOT), "reachability_verification",
                      "outputs", "frozen_reachability.csv")
RADIUS = 10.0          # same clustering radius as deduplicate_by_location
TOPK = 5


def cluster(raw):
    """Group raw detections into location clusters, mirroring
    deduplicate_by_location's greedy highest-score-first assignment so the
    cluster representatives are exactly production's kept candidates."""
    ordered = sorted(raw, key=lambda c: c.score, reverse=True)
    reps, members = [], []
    r2 = RADIUS ** 2
    for c in ordered:
        placed = False
        for i, rep in enumerate(reps):
            if (c.x - rep.x) ** 2 + (c.y - rep.y) ** 2 <= r2:
                members[i].append(c)
                placed = True
                break
        if not placed:
            reps.append(c)
            members.append([c])
    return reps, members


def aggregates(scores):
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    n = len(s)
    trim = s[: max(1, int(round(0.8 * n)))]      # drop the weakest 20%
    return {"max": float(s[0]), "mean": float(s.mean()), "median": float(np.median(s)),
            "trimmed": float(trim.mean()), "top5": float(s[:TOPK].mean()), "count": float(n)}


def main() -> None:
    reach = pd.read_csv(REACH)
    targets = reach[(~reach.correct) & (reach.reachable)]
    manifests = {s: load_manifest("data", s).set_index("pair_id") for s in targets.split.unique()}
    print(f"{len(targets)} reachable failures; bar = >70% preference for truth\n")

    keys = ("max", "mean", "median", "trimmed", "top5", "count")
    rows = []
    for _, tr in targets.iterrows():
        row = manifests[tr.split].loc[tr.pair_id]
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = (float(row["gt_x"]), float(row["gt_y"]))

        # Production's arm choice, then keep that arm's RAW (pre-dedup) pool.
        best = None
        for sigma in (0.0, PSF_MATCH_SIGMA):
            raw = candidate_generation.build_candidate_pool(ref, srch, psf_sigma=sigma)
            gap = _decisiveness(candidate_generation.deduplicate_by_location(raw))
            if best is None or gap > best[0]:
                best = (gap, sigma, raw)
        _, sigma, raw = best
        reps, members = cluster(raw)

        # chosen = production's winner = the top cluster representative
        order = np.argsort([-r.score for r in reps])
        ci = int(order[0])
        # truth cluster = representative nearest ground truth
        ti = int(np.argmin([np.hypot(r.x - gt[0], r.y - gt[1]) for r in reps]))
        if ti == ci:
            continue                      # not a distinguishable pair
        ac, at = aggregates([c.score for c in members[ci]]), \
                 aggregates([c.score for c in members[ti]])
        rec = {"pair_id": tr.pair_id, "split": tr.split, "family": tr.structural_family,
               "n_chosen": int(ac["count"]), "n_truth": int(at["count"])}
        for k in keys:
            rec[f"{k}_margin"] = at[k] - ac[k]
        rows.append(rec)
        print(f"  {tr.pair_id:30s} n {int(ac['count']):3d}/{int(at['count']):3d}  "
              f"max={rec['max_margin']:+.4f} mean={rec['mean_margin']:+.4f} "
              f"med={rec['median_margin']:+.4f} top5={rec['top5_margin']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "hypothesis_ensemble.csv"), index=False)

    print("\n" + "=" * 62)
    print(f"CROSS-HYPOTHESIS AGGREGATES ON {len(df)} REACHABLE FAILURES")
    print("=" * 62)
    print(f"  {'aggregate':>12} {'median margin':>15} {'prefers truth':>16}")
    summary = {}
    for k in keys:
        v = df[f"{k}_margin"]
        n = int((v > 0).sum())
        summary[k] = {"median": float(v.median()), "prefers_truth": n, "n": len(df)}
        note = ""
        if k == "max":
            note = "   <- production's estimator (0% expected)"
        elif k == "count":
            note = "   <- already shown a no-op by cross_hypothesis_consensus_rerank"
        elif n / max(len(df), 1) > 0.70:
            note = "   <- PASSES BAR"
        print(f"  {k:>12} {v.median():>15.5f} {n:>9}/{len(df):<4} "
              f"({n/max(len(df),1):.0%}){note}")
    print("\n  Read `top5` as the decisive column: it is size-independent, so it separates")
    print("  a genuine magnitude effect from the cluster-count effect already known to be")
    print("  a no-op. If `mean` passes but `top5` does not, the effect is only count.")
    with open(os.path.join(out, "hypothesis_ensemble.json"), "w") as f:
        json.dump({"n": len(df), "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
