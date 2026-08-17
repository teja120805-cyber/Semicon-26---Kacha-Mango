"""Anchor on the aperiodic sub-region and use it to RESTRICT the search.

Every change tried in this campaign either re-scores the pool or enlarges
it. `experiments/wide_pool_rescoring/` showed enlarging is strictly harmful
(0 rescues, breaks scaling with recall) and
`experiments/discriminability_weighted/` showed re-scoring does nothing.
The untried direction is the opposite one: **remove decoys** rather than
add candidates or re-rank them.

Literature basis, from `reports/RESEARCH_SURVEY_SCORING.md`'s own sources -
both cited there but neither turned into one of the P1-P6 proposals:

  * KLA US9830421B2 explicitly REJECTS device features that are "repeating"
    or "lacking uniqueness" and targets array corners and boundaries. Our
    own forensics say the same thing: 89.8% accuracy on boundary-crossing
    crops against 54.4% on crops crossing neither.
  * Cognex US6324299B1 "sub-models" - match on a sub-region rather than the
    whole model.
  * US20090103799A1 - classify the pattern by FFT, then CLAMP the search
    range accordingly.

Why a sub-template is genuinely different from P3's weighting, which failed:
P3 kept the full 100x100 window and reweighted inside it, so the periodic
90% still dominated the ZNCC **denominator** even at weight ~0. Restricting
the support removes that content entirely - it contributes nothing, not even
to normalization.

Mechanism:
  1. Locate the most aperiodic sub-window of the Reference, reusing P3's
     lattice-shift dissimilarity map (which the P3 report established IS a
     working discriminability estimator - Gini 0.57, 44% more mass on
     top-decile gradient pixels - even though weighting by it did nothing).
  2. Correlate that sub-template over the Search image to get a spatial
     PRIOR: a handful of places the aperiodic structure could be.
  3. Keep only full-template candidates within `radius_px` of one of those
     places. Decoys sitting in featureless array interior are eliminated.
  4. If the Reference has no meaningfully aperiodic region, or the prior is
     not decisive, fall back to the unrestricted pool - never fail closed.

Null control: `radius_px = inf` keeps every candidate, reproducing
production bit-for-bit.

Stage 1 is a DIAGNOSTIC that decides whether stage 2 is worth running: if
the prior does not cover the true location more often than it covers the
winning decoy, the idea is dead and no end-to-end run is warranted.

    python -m experiments.aperiodic_anchor.run --surface tune_degraded
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
from pipeline import candidate_generation, feature_extraction, matching, ranking, refinement
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness, localize

from experiments.discriminability_weighted.weights import (estimate_lattice_vectors,
                                                            _shift_dissimilarity)

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}

SUB_FRACTION = 0.45      # sub-window side, as a fraction of the template side
PRIOR_PEAKS = 6
PRIOR_NMS_PX = 25


def aperiodic_subwindow(template: np.ndarray, fraction: float = SUB_FRACTION):
    """Top-left of the sub-window carrying the most lattice-shift
    dissimilarity, plus the concentration ratio that says whether the
    Reference has a meaningfully aperiodic region at all.

    ratio = (mean dissimilarity inside the window) / (mean over the whole
    template). 1.0 means the "aperiodic" region is no more aperiodic than
    anywhere else, i.e. there is nothing to anchor on.
    """
    vecs = estimate_lattice_vectors(template)
    if not vecs:
        return None, 1.0
    dis = _shift_dissimilarity(template, vecs)
    dis = cv2.GaussianBlur(dis, (0, 0), 2.0, borderType=cv2.BORDER_REPLICATE)
    side = max(8, int(round(template.shape[0] * fraction)))
    if side >= template.shape[0]:
        return None, 1.0
    integral = cv2.integral(dis.astype(np.float64))
    h, w = dis.shape
    sums = (integral[side:h + 1, side:w + 1] - integral[0:h - side + 1, side:w + 1]
            - integral[side:h + 1, 0:w - side + 1] + integral[0:h - side + 1, 0:w - side + 1])
    idx = int(np.argmax(sums))
    sy, sx = divmod(idx, sums.shape[1])
    best_mean = float(sums.max()) / (side * side)
    overall = float(dis.mean())
    ratio = best_mean / overall if overall > 1e-12 else 1.0
    return (sx, sy, side), ratio


def prior_peaks(reference, search, psf_sigma, scale, rotation):
    """Candidate locations of the Reference CENTRE implied by matching only
    the aperiodic sub-window. The offset from sub-window to template centre
    is added back, so these are directly comparable to full-template
    candidate coordinates."""
    tmpl = matching.build_template(reference, scale, rotation, psf_sigma)
    box, ratio = aperiodic_subwindow(tmpl)
    if box is None:
        return [], ratio
    sx, sy, side = box
    sub = tmpl[sy:sy + side, sx:sx + side]
    if sub.std() < 1e-6:
        return [], ratio
    smap = matching.correlate(search, sub)
    peaks = matching.top_k_peaks(smap, PRIOR_PEAKS, PRIOR_NMS_PX)
    th, tw = tmpl.shape
    out = []
    for px, py, s in peaks:
        # sub-window top-left in Search coords -> template centre in Search coords
        out.append((px - sx + tw / 2.0, py - sy + th / 2.0, s))
    return out, ratio


def diagnose(imgs, min_ratio: float) -> pd.DataFrame:
    rows = []
    for row, ref, srch in imgs:
        gt = (float(row["gt_x"]), float(row["gt_y"]))
        best = None
        for sigma in (0.0, PSF_MATCH_SIGMA):
            pool = candidate_generation.deduplicate_by_location(
                candidate_generation.build_candidate_pool(ref, srch, psf_sigma=sigma))
            g = _decisiveness(pool)
            if best is None or g > best[0]:
                best = (g, sigma, pool)
        _, psf, pool = best
        chosen = ranking.rank_classical(pool)[0]
        err = float(np.hypot(chosen.x - gt[0], chosen.y - gt[1]))
        peaks, ratio = prior_peaks(ref, srch, psf, chosen.scale, chosen.rotation_deg)
        if not peaks:
            rows.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                          "err": err, "correct": err <= 5, "ratio": ratio,
                          "d_truth": np.nan, "d_chosen": np.nan, "usable": False})
            continue
        d_truth = min(np.hypot(p[0] - gt[0], p[1] - gt[1]) for p in peaks)
        d_chosen = min(np.hypot(p[0] - chosen.x, p[1] - chosen.y) for p in peaks)
        rows.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                      "err": err, "correct": err <= 5, "ratio": ratio,
                      "d_truth": d_truth, "d_chosen": d_chosen, "usable": ratio >= min_ratio})
        print(f"  {row['pair_id']:32s} err={err:8.2f} ratio={ratio:4.2f} "
              f"d_truth={d_truth:7.1f} d_chosen={d_chosen:7.1f}", flush=True)
    return pd.DataFrame(rows)


def localize_anchored(ref, srch, radius_px, min_ratio):
    t0 = time.perf_counter()
    best = None
    for sigma in (0.0, PSF_MATCH_SIGMA):
        pool = candidate_generation.deduplicate_by_location(
            candidate_generation.build_candidate_pool(ref, srch, psf_sigma=sigma))
        g = _decisiveness(pool)
        if best is None or g > best[0]:
            best = (g, sigma, pool)
    _, psf, pool = best
    ranked = ranking.rank_classical(pool)
    restricted = False
    if np.isfinite(radius_px):
        top = ranked[0]
        peaks, ratio = prior_peaks(ref, srch, psf, top.scale, top.rotation_deg)
        if peaks and ratio >= min_ratio:
            keep = [c for c in ranked
                    if min(np.hypot(p[0] - c.x, p[1] - c.y) for p in peaks) <= radius_px]
            if keep:                      # never fail closed
                ranked, restricted = keep, True
    ranked = ranking.apply_center_tiebreak(ranked, srch.shape)
    x, y = refinement.refine(ref, srch, ranked[0], psf)
    scores = sorted((c.score for c in pool), reverse=True)
    return x, y, float(ranked[0].score), feature_extraction.ambiguity_ratio(scores), \
        restricted, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    p.add_argument("--min-ratio", type=float, default=1.15)
    p.add_argument("--radii", default="40,80,150")
    p.add_argument("--skip-diagnostic", action="store_true")
    args = p.parse_args()

    manifest = load_manifest(SURFACES[args.surface], "development")
    imgs = [(r, cv2.imread(r["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32),
             cv2.imread(r["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32))
            for _, r in manifest.iterrows()]
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)

    if not args.skip_diagnostic:
        print("=== STAGE 1: does the aperiodic prior cover truth more than the decoy? ===")
        d = diagnose(imgs, args.min_ratio)
        d.to_csv(os.path.join(out, f"{args.surface}_prior_diagnostic.csv"), index=False)
        f = d[~d.correct & d.usable & d.d_truth.notna()]
        c = d[d.correct & d.usable & d.d_truth.notna()]
        print(f"\nusable (aperiodic ratio >= {args.min_ratio}): {int(d.usable.sum())}/{len(d)}")
        if len(f):
            print(f"FAILURES  (n={len(f)}): median d_truth={f.d_truth.median():.1f}px  "
                  f"median d_chosen={f.d_chosen.median():.1f}px  "
                  f"prior nearer truth on {int((f.d_truth < f.d_chosen).sum())}/{len(f)}")
        if len(c):
            print(f"CORRECT   (n={len(c)}): median d_truth={c.d_truth.median():.1f}px  "
                  f"within 80px on {int((c.d_truth <= 80).sum())}/{len(c)}")
        print("\nGO if the prior sits nearer truth than the decoy on most failures AND covers")
        print("nearly every correct pair (else restriction will break what already works).\n")

    print("=== STAGE 2: end-to-end ===")
    bad = []
    for row, ref, srch in imgs[:5]:
        base = localize(ref.astype(np.uint8), srch.astype(np.uint8))
        x, y, conf, _a, _r, _t = localize_anchored(ref, srch, float("inf"), args.min_ratio)
        if not (x == base.x and y == base.y and conf == base.confidence):
            bad.append(row["pair_id"])
    print(f"null control (radius=inf): checked 5, mismatches={len(bad)} {bad}")
    if bad:
        raise SystemExit("null control failed")

    def run(radius):
        rows = []
        for row, ref, srch in imgs:
            x, y, conf, amb, restricted, rt = localize_anchored(ref, srch, radius, args.min_ratio)
            rows.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                          "error_px": float(np.hypot(x - row["gt_x"], y - row["gt_y"])),
                          "restricted": restricted, "runtime_s": rt})
        return pd.DataFrame(rows)

    base_df = run(float("inf"))
    base_acc = float((base_df.error_px <= 5).mean())
    print(f"baseline acc@5px = {base_acc:.4f}  n={len(base_df)}\n")

    results = []
    for radius in [float(v) for v in args.radii.split(",")]:
        df = run(radius)
        m = base_df[["pair_id", "error_px"]].merge(df[["pair_id", "error_px"]], on="pair_id",
                                                    suffixes=("_b", "_c"))
        rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
        broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())
        acc = float((df.error_px <= 5).mean())
        rec = {"radius_px": radius, "acc_5px": acc, "delta_pp": 100 * (acc - base_acc),
               "rescued": rescued, "broken": broken, "net": rescued - broken,
               "n_restricted": int(df.restricted.sum()), "mean_runtime_s": float(df.runtime_s.mean())}
        results.append(rec)
        df.to_csv(os.path.join(out, f"{args.surface}_r{radius}.csv"), index=False)
        print(f"radius={radius:<6} acc={acc:.4f} ({rec['delta_pp']:+5.1f}pp) R={rescued} "
              f"B={broken} net={rec['net']:+d} restricted={rec['n_restricted']}/{len(df)} "
              f"rt={rec['mean_runtime_s']:.2f}s", flush=True)

    with open(os.path.join(out, f"{args.surface}_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "baseline_acc": base_acc, "n": len(base_df),
                   "min_ratio": args.min_ratio, "configs": results}, f, indent=2)
    print("\n=== ranked ===")
    print(pd.DataFrame(results).sort_values(["net", "acc_5px"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
