#!/usr/bin/env python
"""Builds a reusable per-pair cache of ranked candidate pools for the full
frozen benchmark (all 156 pairs, including `development` - investigation
benefits from every pair even though `development` isn't a gate split,
same convention as center_tiebreak_v2/investigate_gaps.py).

This is the one expensive step (full multi-scale x multi-rotation
candidate_generation.build_candidate_pool for every pair - the same cost
as a full evaluate_all run). Every unmodified pipeline function is called
exactly as production does (candidate_generation.build_candidate_pool,
candidate_generation.deduplicate_by_location, ranking.rank_classical) -
nothing here is a reimplementation. Caching the ranked pool (not just the
winner) is what lets experiments/multiway_tiebreak_v1/sweep.py try many
tie-break configurations cheaply afterward without re-running candidate
generation for each one - only pipeline.refinement.refine() (cheap: one
extra correlation for the winning hypothesis, not the whole grid) needs to
re-run per config, since the tie-break can change which candidate wins.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import candidate_generation, ranking  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
ALL_SPLITS = ("development", "validation", "held_out", "challenge", "cross_generator")
TOP_N_CACHED = 30  # generous - any plausible tie-break epsilon only ever touches a handful of top candidates


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd_concat_manifest()
    print(f"Building candidate-pool cache for {len(manifest)} pairs...")

    records = []
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(manifest.iterrows()):
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)

        raw_candidates = candidate_generation.build_candidate_pool(ref, search)
        candidates = candidate_generation.deduplicate_by_location(raw_candidates)
        ranked = ranking.rank_classical(candidates)

        top = ranked[:TOP_N_CACHED]
        records.append({
            "pair_id": row["pair_id"], "split": row["split"], "structural_family": row["structural_family"],
            "reference_path": row["reference_path"], "search_path": row["search_path"],
            "gt_x": row["gt_x"], "gt_y": row["gt_y"],
            "search_height": int(search.shape[0]), "search_width": int(search.shape[1]),
            "n_candidates_total": len(candidates),
            "ranked_top": [
                {"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                 "rotation_deg": c.rotation_deg, "template_size": c.template_size}
                for c in top
            ],
        })
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(manifest)} pairs cached ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - t0
    print(f"Done: {len(records)} pairs cached in {elapsed:.1f}s")
    with open(os.path.join(OUT_DIR, "candidate_cache.json"), "w") as f:
        json.dump(records, f)
    print(f"Wrote {OUT_DIR}/candidate_cache.json")


def pd_concat_manifest():
    import pandas as pd
    return pd.concat([load_manifest(os.path.join(PROJECT_ROOT, "data"), s) for s in ALL_SPLITS], ignore_index=True)


if __name__ == "__main__":
    main()
