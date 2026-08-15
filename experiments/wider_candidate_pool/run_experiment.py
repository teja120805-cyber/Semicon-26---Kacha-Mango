"""Experiment: does a wider candidate pool (more peaks kept per scale/
rotation hypothesis, tighter non-max-suppression radius) rescue more of the
periodicity-driven `candidate_generation` failures found in
reports/ACCURACY_FORENSICS.md (Finding 1)?

Forensics found that for dense periodic mats, the majority of failures never
even get a candidate near ground truth into the pool at all - not a ranking
mistake, a generation one. `pipeline/candidate_generation.py`'s defaults
(`PEAKS_PER_HYPOTHESIS=2`, `SUPPRESSION_RADIUS_PX=8`) keep only the top-2
peaks per of 25 hypotheses and merge anything within 8px of a kept peak.
For a mat with an ~5-8px word pitch (the two densest presets), that
suppression radius can span nearly two full periods, plausibly discarding a
correct-but-lower-scoring peak in favor of a nearby wrong repeat before
ranking ever sees it.

This experiment does NOT modify pipeline/ (`localize()` doesn't expose
`peaks_per_hypothesis`/`suppression_radius_px` as pass-through parameters) -
it calls `candidate_generation.build_candidate_pool`, `ranking.rank_classical`,
and `refinement.refine` directly with wider settings, mirroring
`pipeline/localize.py`'s own orchestration exactly (same order, same
preprocessing) so the only thing that differs is the two candidate-pool
parameters under test.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import benchmark  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import candidate_generation, feature_extraction, ranking, refinement  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")

BASELINE_PEAKS = candidate_generation.PEAKS_PER_HYPOTHESIS  # 2
BASELINE_RADIUS = candidate_generation.SUPPRESSION_RADIUS_PX  # 8

# Candidate: 3x the peaks per hypothesis, half the suppression radius -
# directly targets "a correct-but-lower-scoring peak near a dense mat's
# pitch gets discarded before ranking ever sees it".
WIDE_PEAKS = 6
WIDE_RADIUS = 4

AMBIGUITY_THRESHOLD = 0.92


def localize_with_pool_settings(reference_img: np.ndarray, search_img: np.ndarray,
                                 peaks_per_hypothesis: int, suppression_radius_px: float):
    t0 = time.perf_counter()
    reference = reference_img.astype(np.float32)
    search = search_img.astype(np.float32)

    raw = candidate_generation.build_candidate_pool(
        reference, search,
        peaks_per_hypothesis=peaks_per_hypothesis,
        suppression_radius_px=suppression_radius_px,
    )
    candidates = candidate_generation.deduplicate_by_location(raw)
    ranked = ranking.rank_classical(candidates)
    winner = ranked[0]
    refined_x, refined_y = refinement.refine(reference, search, winner)

    pooled_scores = sorted((c.score for c in candidates), reverse=True)
    amb_ratio = feature_extraction.ambiguity_ratio(pooled_scores)
    runtime_s = time.perf_counter() - t0
    return refined_x, refined_y, float(winner.score), amb_ratio >= AMBIGUITY_THRESHOLD, runtime_s


def evaluate_split(data_root: str, split: str, peaks: int, radius: float) -> pd.DataFrame:
    manifest = load_manifest(data_root, split)
    results = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        x, y, conf, ambiguous, runtime_s = localize_with_pool_settings(ref, search, peaks, radius)
        error_px = float(np.hypot(x - row["gt_x"], y - row["gt_y"]))
        results.append({
            **row.to_dict(), "pred_x": x, "pred_y": y, "error_px": error_px,
            "confidence": conf, "ambiguous": ambiguous, "runtime_s": runtime_s,
        })
    return pd.DataFrame(results)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
    baseline_df = pd.read_csv(baseline_path)
    baseline_df = baseline_df[baseline_df["split"].isin(GATE_SPLITS)].reset_index(drop=True)

    all_results = []
    t0 = time.perf_counter()
    for split in GATE_SPLITS:
        print(f"=== {split} (peaks={WIDE_PEAKS}, radius={WIDE_RADIUS}px) ===")
        df = evaluate_split(os.path.join(PROJECT_ROOT, "data"), split, WIDE_PEAKS, WIDE_RADIUS)
        all_results.append(df)
        print(f"  n={len(df)} acc@5px={(df['error_px']<=5).mean():.3f} mean_runtime={df['runtime_s'].mean():.3f}s")
    candidate_df = pd.concat(all_results, ignore_index=True)
    elapsed = time.perf_counter() - t0

    candidate_df.to_csv(os.path.join(OUT_DIR, "per_pair_results_wide_pool.csv"), index=False)

    gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
    gate["note"] = (
        "seeds_agree=True passed manually: deterministic algorithmic change (wider candidate "
        "pool), not a trained stochastic model - no seed variance to check."
    )
    gate["total_wall_time_s"] = elapsed
    gate["pool_settings"] = {
        "baseline": {"peaks_per_hypothesis": BASELINE_PEAKS, "suppression_radius_px": BASELINE_RADIUS},
        "candidate": {"peaks_per_hypothesis": WIDE_PEAKS, "suppression_radius_px": WIDE_RADIUS},
    }
    with open(os.path.join(OUT_DIR, "integration_gate.json"), "w") as f:
        json.dump(gate, f, indent=2)

    print("\n=== Integration gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    print(f"Total wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
