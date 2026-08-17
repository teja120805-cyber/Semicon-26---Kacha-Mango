"""Verify the campaign's headline claim on the authoritative benchmark.

`experiments/REACHABILITY_CAMPAIGN.md` — and, since 2026-08-17, the project
README's Known Limitations — assert that **~74% of failures are unreachable**:
ground truth is not within 5px of any pooled candidate, so no re-scoring,
re-ranking or tie-breaking stage can fix them.

That number rests on **19 failures** across two surfaces (`development` 4/7,
a fresh degraded 40-pair set 10/12). It is a load-bearing claim now published
in the README, and 19 events is a thin basis for it. The frozen 156-pair
benchmark contains ~35 failures — nearly twice the sample, on the
authoritative data — so the claim can be checked properly.

This is a **diagnostic, not a scoring run**: it selects nothing, tunes
nothing, and changes no configuration. It reports reachability and the
per-split decomposition, and either confirms the published figure or
corrects it. If it corrects it, the README and campaign report must be
amended — a claim that is wrong is worse than one that is absent.

    python -m experiments.reachability_verification.run
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
SPLITS = ("development", "validation", "held_out", "challenge", "cross_generator")
HIT_PX = 5.0


def main() -> None:
    rows = []
    for split in SPLITS:
        manifest = load_manifest("data", split)
        for _, row in manifest.iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt = (float(row["gt_x"]), float(row["gt_y"]))

            best = None
            for sigma in (0.0, PSF_MATCH_SIGMA):
                pool = candidate_generation.deduplicate_by_location(
                    candidate_generation.build_candidate_pool(ref, srch, psf_sigma=sigma))
                gap = _decisiveness(pool)
                if best is None or gap > best[0]:
                    best = (gap, sigma, pool)
            _, sigma, pool = best
            chosen = ranking.apply_center_tiebreak(ranking.rank_classical(pool), srch.shape)[0]

            err = float(np.hypot(chosen.x - gt[0], chosen.y - gt[1]))
            d_truth = min(np.hypot(c.x - gt[0], c.y - gt[1]) for c in pool)
            # Rank of the truth-nearest candidate, when reachable: 1 would mean
            # the pool already ranks it first (impossible for a failure), and a
            # small rank means a selector has a realistic chance.
            ordered = ranking.rank_classical(pool)
            truth_rank = None
            for i, c in enumerate(ordered, start=1):
                if np.hypot(c.x - gt[0], c.y - gt[1]) <= HIT_PX:
                    truth_rank = i
                    break
            rows.append({
                "pair_id": row["pair_id"], "split": split,
                "structural_family": row["structural_family"],
                "error_px": err, "correct": err <= HIT_PX,
                "pool_size": len(pool), "psf_sigma": sigma,
                "nearest_candidate_px": d_truth, "reachable": bool(d_truth <= HIT_PX),
                "truth_rank": truth_rank,
                "top_score": float(ordered[0].score),
                "truth_score": float(ordered[truth_rank - 1].score) if truth_rank else np.nan,
            })
            print(f"  [{split}] {row['pair_id']:30s} err={err:8.2f} "
                  f"reach={rows[-1]['reachable']} rank={truth_rank}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "frozen_reachability.csv"), index=False)

    fails = df[~df.correct]
    unreach = fails[~fails.reachable]
    print("\n" + "=" * 62)
    print(f"FROZEN BENCHMARK REACHABILITY   n={len(df)}  accuracy={df.correct.mean():.4f}")
    print("=" * 62)
    print(f"  failures                 : {len(fails)}")
    print(f"  unreachable (cand. gen.) : {len(unreach)}  ({len(unreach)/max(len(fails),1):.1%})")
    print(f"  reachable (selector)     : {len(fails) - len(unreach)}  "
          f"({1 - len(unreach)/max(len(fails),1):.1%})")
    print(f"\n  PUBLISHED CLAIM: 74% unreachable (from 19 failures)")
    print(f"  MEASURED HERE  : {len(unreach)/max(len(fails),1):.1%} (from {len(fails)} failures)")

    print("\n  per split")
    print(f"    {'split':16s} {'n':>4} {'fails':>6} {'unreach':>8} {'share':>7}")
    for s, g in df.groupby("split"):
        f = g[~g.correct]
        u = f[~f.reachable]
        share = f"{len(u)/len(f):.0%}" if len(f) else "-"
        print(f"    {s:16s} {len(g):>4} {len(f):>6} {len(u):>8} {share:>7}")

    reach_fails = fails[fails.reachable]
    if len(reach_fails):
        print(f"\n  where truth sits in the pool, on the {len(reach_fails)} reachable failures")
        print(f"    truth_rank: median {reach_fails.truth_rank.median():.0f}, "
              f"min {reach_fails.truth_rank.min():.0f}, max {reach_fails.truth_rank.max():.0f}")
        gap = reach_fails.top_score - reach_fails.truth_score
        print(f"    score gap to the winner: median {gap.median():.4f}, max {gap.max():.4f}")

    print(f"\n  selector efficiency = accuracy / recall = "
          f"{df.correct.mean():.4f} / {df.reachable.mean():.4f} = "
          f"{df.correct.mean()/df.reachable.mean():.4f}")

    with open(os.path.join(out, "frozen_reachability.json"), "w") as f:
        json.dump({"n": len(df), "accuracy": float(df.correct.mean()),
                   "recall": float(df.reachable.mean()),
                   "failures": len(fails), "unreachable": len(unreach),
                   "unreachable_share": float(len(unreach) / max(len(fails), 1)),
                   "selector_efficiency": float(df.correct.mean() / df.reachable.mean())}, f, indent=2)


if __name__ == "__main__":
    main()
