"""P5 — Deformable Diversity Similarity, the one survey proposal never tested.

`reports/RESEARCH_SURVEY_SCORING.md` §P5 proposes DDIS as a re-ranker over the
existing top-K. Neither `ACCURACY_90_CAMPAIGN.md` (nine experiments) nor
`REACHABILITY_CAMPAIGN.md` (six) ever ran it. P1, P2, P3, P4 and P6 have all
now been tested or explicitly ruled out; P5 is the gap.

**Why it is worth a thirteenth attempt after twelve rejections.** Every prior
attempt arbitrated by ZNCC or a monotone function of it — reweighting it,
prewhitening it, penalising its peaks, or replacing it with a learned score
trained on this data. DDIS is not a similarity at all in that sense: it asks
how many template patches find *distinct* nearest neighbours in the candidate
window. At a true match most patches match something unique; at a periodic
decoy they collapse onto a few repeating positions. **That collapse is the
periodic signature itself**, measured directly rather than inferred from a
correlation peak.

**The bar, set before running** (from `experiments/reachability_verification/`):
the frozen benchmark has **22 reachable failures** where the true location is
in the pool at median rank 3, losing by a median of **0.029 ZNCC**. A useful
signal must separate that. This measures exactly that separation, on those
exact pairs, and nothing else. No harness, no sweep, no end-to-end run until
this says it is worth building.

Reported per pair:
  ddis_truth / ddis_chosen  diversity score at each location
  ddis_margin               truth - chosen; POSITIVE means DDIS would flip it
  zncc_margin               the deficit DDIS has to overcome (negative)

    python -m experiments.ddis_diversity.diagnose
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching, ranking
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

ROOT = os.path.dirname(os.path.abspath(__file__))
REACH_CSV = os.path.join(os.path.dirname(ROOT), "reachability_verification",
                          "outputs", "frozen_reachability.csv")

PATCH = 8          # template patch side, in template pixels
STRIDE = 4         # patch sampling stride
SEARCH_R = 12      # how far a patch may look for its nearest neighbour


def ddis_score(template: np.ndarray, window: np.ndarray) -> float:
    """Deformable Diversity Similarity.

    Each template patch searches a +-SEARCH_R neighbourhood of its own
    position in the candidate window for its best match (lowest SSD). The
    score is the fraction of DISTINCT destinations those matches land on:
    1.0 means every patch found its own unique correspondence, low values
    mean many patches collapsed onto the same few positions — the signature
    of matching against a repeating lattice.

    Deliberately ignores match QUALITY and uses only the diversity of the
    correspondence field, so this cannot silently become another ZNCC
    variant; it carries information ZNCC does not.
    """
    t = template.astype(np.float32)
    w = window.astype(np.float32)
    h, wd = t.shape
    dests = []
    for y in range(0, h - PATCH + 1, STRIDE):
        for x in range(0, wd - PATCH + 1, STRIDE):
            patch = t[y:y + PATCH, x:x + PATCH]
            y0, y1 = max(0, y - SEARCH_R), min(h - PATCH + 1, y + SEARCH_R + 1)
            x0, x1 = max(0, x - SEARCH_R), min(wd - PATCH + 1, x + SEARCH_R + 1)
            region = w[y0:y1 + PATCH - 1, x0:x1 + PATCH - 1]
            if region.shape[0] < PATCH or region.shape[1] < PATCH:
                continue
            # SSD via matchTemplate on the local region only (cheap).
            res = cv2.matchTemplate(region, patch, cv2.TM_SQDIFF)
            idx = int(np.argmin(res))
            ry, rx = divmod(idx, res.shape[1])
            dests.append((y0 + ry, x0 + rx))
    if not dests:
        return float("nan")
    return len(set(dests)) / len(dests)


def main() -> None:
    reach = pd.read_csv(REACH_CSV)
    targets = reach[(~reach.correct) & (reach.reachable)]
    print(f"reachable failures on the frozen benchmark: {len(targets)}")
    print(f"the bar: median ZNCC deficit to overcome = "
          f"{(targets.top_score - targets.truth_score).median():.4f}\n")

    manifests = {s: load_manifest("data", s).set_index("pair_id")
                 for s in targets.split.unique()}

    rows = []
    for _, tr in targets.iterrows():
        row = manifests[tr.split].loc[tr.pair_id]
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
        ordered = ranking.rank_classical(pool)
        chosen = ordered[0]
        truth = min(pool, key=lambda c: np.hypot(c.x - gt[0], c.y - gt[1]))

        def window_for(c):
            t = matching.build_template(ref, c.scale, c.rotation_deg, sigma)
            th, tw = t.shape
            x0, y0 = int(round(c.x - tw / 2.0)), int(round(c.y - th / 2.0))
            if x0 < 0 or y0 < 0 or y0 + th > srch.shape[0] or x0 + tw > srch.shape[1]:
                return None, None
            return t, srch[y0:y0 + th, x0:x0 + tw]

        t_c, w_c = window_for(chosen)
        t_t, w_t = window_for(truth)
        if w_c is None or w_t is None:
            continue
        d_c, d_t = ddis_score(t_c, w_c), ddis_score(t_t, w_t)
        rows.append({
            "pair_id": tr.pair_id, "split": tr.split, "family": tr.structural_family,
            "zncc_chosen": float(chosen.score), "zncc_truth": float(truth.score),
            "zncc_margin": float(truth.score - chosen.score),
            "ddis_chosen": d_c, "ddis_truth": d_t, "ddis_margin": d_t - d_c,
        })
        print(f"  {tr.pair_id:30s} zncc_margin={rows[-1]['zncc_margin']:+.4f}  "
              f"ddis {d_c:.3f}->{d_t:.3f}  margin={rows[-1]['ddis_margin']:+.4f}  "
              f"{'FLIP' if rows[-1]['ddis_margin'] > 0 else ''}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "ddis_reachable_failures.csv"), index=False)

    flips = int((df.ddis_margin > 0).sum())
    print("\n" + "=" * 60)
    print(f"DDIS ON THE {len(df)} REACHABLE FAILURES")
    print("=" * 60)
    print(f"  ZNCC margin (the deficit): median {df.zncc_margin.median():+.4f}")
    print(f"  DDIS margin              : median {df.ddis_margin.median():+.4f}")
    print(f"  DDIS prefers TRUTH on    : {flips}/{len(df)}  ({flips/max(len(df),1):.0%})")
    print(f"\n  chance would be ~50%. A useful signal needs to be clearly above it,")
    print(f"  AND must not destroy the pairs that are currently correct — which this")
    print(f"  diagnostic does not test. Treat >70% here as the bar for building a harness.")
    with open(os.path.join(out, "ddis_summary.json"), "w") as f:
        json.dump({"n_reachable_failures": len(df), "ddis_prefers_truth": flips,
                   "share": flips / max(len(df), 1),
                   "median_zncc_margin": float(df.zncc_margin.median()),
                   "median_ddis_margin": float(df.ddis_margin.median())}, f, indent=2)


if __name__ == "__main__":
    main()
