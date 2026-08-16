#!/usr/bin/env python
"""Dual-pass PSF selection: run the pool with BOTH the production template
and the PSF-matched (sigma=1.6) template, and keep whichever produced the
more decisive match.

Why not a threshold gate on the spectral estimator. That was the plan in
psf_second_seed/REPORT.md §4, and checking it before running showed it
cannot work: the families that GAIN under sigma=1.6 span estimated sigma
0.36-1.06 and the families that LOSE span 0.34-1.03. The ranges overlap
almost completely, so no threshold separates them. That recommendation was
wrong and is corrected here.

What this uses instead. oracle_ceiling_diagnostic measured a pool-internal
statistic that genuinely predicts correctness without ground truth: the gap
between the top candidate's score and the best score at a location >10px
away (correct pairs median 0.0188, wrong pairs 0.0026; 95% failure recall
at a 0.01 threshold). If that statistic says which ANSWERS are trustworthy,
it can also say which TEMPLATE to trust on a given pair.

This script does the expensive part once - both pools for every pair,
recording each arm's winner, scores, gap and resulting error - and writes a
CSV. Selection rules are then evaluated offline in pure post-processing
(evaluate_rules.py), so comparing candidate rules costs no extra compute
and no extra benchmark runs.

Runs on either seed. Never modifies pipeline/, generator/, or model/.

Usage: python run_dual_pass.py [production|second]
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "psf_matched_template"))

from pipeline import candidate_generation, matching, ranking  # noqa: E402
from pipeline.refinement import _parabolic_offset  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402

from psf_match import build_template_psf_matched  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
SIGMAS = (0.0, 1.6)
DISTINCT_PX = 10.0
SPLITS = ["development", "validation", "held_out", "challenge", "cross_generator"]


def arm(reference, search, sigma):
    """One pool build + rank + refine at a given sigma. Everything except
    the template is the unmodified production path."""
    cands = []
    for scale in candidate_generation.DEFAULT_SCALE_HYPOTHESES:
        for rot in candidate_generation.DEFAULT_ROTATION_HYPOTHESES:
            t = build_template_psf_matched(reference, scale, rot, sigma)
            smap = matching.correlate(search, t)
            for px, py, sc in matching.top_k_peaks(smap, candidate_generation.PEAKS_PER_HYPOTHESIS,
                                                   candidate_generation.SUPPRESSION_RADIUS_PX):
                cands.append(candidate_generation.Candidate(
                    x=px + t.shape[1] / 2.0, y=py + t.shape[0] / 2.0, score=sc,
                    scale=scale, rotation_deg=rot, template_size=t.shape[0]))
    cands = candidate_generation.deduplicate_by_location(cands)
    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(cands), search.shape)
    win = ranked[0]

    ordered = sorted(cands, key=lambda c: c.score, reverse=True)
    top = ordered[0]
    runner = next((c for c in ordered[1:]
                   if (c.x - top.x) ** 2 + (c.y - top.y) ** 2 > DISTINCT_PX ** 2), None)
    gap = float(top.score - runner.score) if runner is not None else float("nan")

    t = build_template_psf_matched(reference, win.scale, win.rotation_deg, sigma)
    smap = matching.correlate(search, t)
    h, w = smap.shape
    px = int(np.clip(round(win.x - t.shape[1] / 2.0), 1, w - 2))
    py = int(np.clip(round(win.y - t.shape[0] / 2.0), 1, h - 2))
    dx = _parabolic_offset(smap[py, px - 1], smap[py, px], smap[py, px + 1])
    dy = _parabolic_offset(smap[py - 1, px], smap[py, px], smap[py + 1, px])
    return {"x": px + dx + t.shape[1] / 2.0, "y": py + dy + t.shape[0] / 2.0,
            "top_score": float(top.score), "gap": gap,
            "rel_gap": gap / float(top.score) if top.score else float("nan"),
            "n_cand": len(cands)}


def iter_pairs(which: str):
    if which == "production":
        for split in SPLITS:
            man = load_manifest(os.path.join(PROJECT_ROOT, "data"), split)
            for _, r in man.iterrows():
                yield {"pair_id": r["pair_id"], "split": split,
                       "family": r["pair_id"].rsplit("_", 1)[0],
                       "reference_path": r["reference_path"], "search_path": r["search_path"],
                       "gt_x": float(r["gt_x"]), "gt_y": float(r["gt_y"])}
    else:
        man = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments", "psf_second_seed",
                                       "outputs", "second_seed_baseline.csv"))
        for _, r in man.iterrows():
            yield {"pair_id": r["pair_id"], "split": r["split"], "family": r["family"],
                   "reference_path": r["reference_path"], "search_path": r["search_path"],
                   "gt_x": float(r["gt_x"]), "gt_y": float(r["gt_y"])}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "production"
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for n, p in enumerate(iter_pairs(which), 1):
        ref = cv2.imread(p["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        se = cv2.imread(p["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        rec = {k: p[k] for k in ("pair_id", "split", "family", "gt_x", "gt_y")}
        for s in SIGMAS:
            a = arm(ref, se, s)
            tag = f"s{s}"
            rec[f"{tag}_x"], rec[f"{tag}_y"] = a["x"], a["y"]
            rec[f"{tag}_err"] = float(np.hypot(a["x"] - p["gt_x"], a["y"] - p["gt_y"]))
            rec[f"{tag}_top"], rec[f"{tag}_gap"] = a["top_score"], a["gap"]
            rec[f"{tag}_relgap"], rec[f"{tag}_ncand"] = a["rel_gap"], a["n_cand"]
        rows.append(rec)
        if n % 20 == 0:
            print(f"  {n} pairs [{time.perf_counter() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, f"dual_pass_{which}.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  (n={len(df)})")
    print(f"  always sigma=0.0 : acc@5px {(df['s0.0_err'] <= 5).mean():.4f}")
    print(f"  always sigma=1.6 : acc@5px {(df['s1.6_err'] <= 5).mean():.4f}")
    print(f"  oracle selection : acc@5px "
          f"{((df['s0.0_err'] <= 5) | (df['s1.6_err'] <= 5)).mean():.4f}   <- headroom")


if __name__ == "__main__":
    main()
