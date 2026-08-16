#!/usr/bin/env python
"""Second-seed generalization check for pitch_aware_prominence, following
this report's own recommendation (and the project's `multiway_tiebreak_v1`
precedent of validating a near-miss on an independently-seeded dataset
before recommending integration).

Generates a fresh dataset with the SAME default FAMILIES table (i.e. the
same structural coverage as production) but a different seed - by
construction (generator/dataset_generator.py's per-pair
default_rng([seed, family_salt(split, family), pair_index]) scheme) this
has zero RNG overlap with production (seed 777001), the forensics sweep
(830001), or any other experiment's seed used in this campaign. Only
covers development/validation/held_out/challenge (`cross_generator` is not
part of the FAMILIES table, and this experiment does not have a second
generator to draw from) - a real, honestly-labeled limitation, noted in
the report.

Never touches pipeline/, generator/, model/, or data/ - writes only under
this experiment's own validation_data/ and outputs/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)

from generator.dataset_generator import generate_dataset  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import metrics  # noqa: E402
from pipeline.localize import localize  # noqa: E402

from harness import localize_pitch_aware  # noqa: E402

SEED = 618234
OUT_ROOT = os.path.join(os.path.dirname(__file__), "validation_data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SPLITS = ["development", "validation", "held_out", "challenge"]

BEST_GAMMA = 1.0
BEST_TOPK = 4


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"=== Generating fresh dataset, seed={SEED} (independent of production 777001) ===")
    t0 = time.perf_counter()
    generate_dataset(OUT_ROOT, seed=SEED, only_splits=SPLITS, verbose=False)
    print(f"  generated in {time.perf_counter() - t0:.1f}s")

    baseline_rows = []
    candidate_rows = []
    for split in SPLITS:
        manifest = load_manifest(OUT_ROOT, split)
        t0 = time.perf_counter()
        for _, row in manifest.iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
            search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)

            base_result = localize(ref, search)  # unmodified production pipeline, classical mode
            base_err = float(np.hypot(base_result.x - row["gt_x"], base_result.y - row["gt_y"]))
            baseline_rows.append({**row.to_dict(), "pred_x": base_result.x, "pred_y": base_result.y,
                                   "error_px": base_err})

            cand_result = localize_pitch_aware(ref, search, gamma=BEST_GAMMA, top_k=BEST_TOPK)
            cand_err = float(np.hypot(cand_result.x - row["gt_x"], cand_result.y - row["gt_y"]))
            candidate_rows.append({**row.to_dict(), "pred_x": cand_result.x, "pred_y": cand_result.y,
                                    "error_px": cand_err})
        print(f"  {split}: n={len(manifest)}, time={time.perf_counter() - t0:.1f}s")

    baseline_df = pd.DataFrame(baseline_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    baseline_df.to_csv(os.path.join(OUT_DIR, "second_seed_baseline.csv"), index=False)
    candidate_df.to_csv(os.path.join(OUT_DIR, "second_seed_pitch_aware.csv"), index=False)

    print("\n=== Per-split comparison (fresh seed 618234) ===")
    for split in SPLITS:
        b = baseline_df[baseline_df["split"] == split]["error_px"]
        c = candidate_df[candidate_df["split"] == split]["error_px"]
        print(f"  {split:16s} n={len(b):3d}  baseline_acc@5px={float((b<=5).mean()):.3f}  "
              f"candidate_acc@5px={float((c<=5).mean()):.3f}")

    b_all = baseline_df["error_px"]
    c_all = candidate_df["error_px"]
    print(f"\nPooled (n={len(b_all)}): baseline_acc@5px={float((b_all<=5).mean()):.4f}  "
          f"candidate_acc@5px={float((c_all<=5).mean()):.4f}")

    b_idx = baseline_df.set_index("pair_id")["error_px"]
    c_idx = candidate_df.set_index("pair_id")["error_px"]
    common = b_idx.index.intersection(c_idx.index)
    rescued = common[(b_idx[common] > 5) & (c_idx[common] <= 5)]
    broken = common[(b_idx[common] <= 5) & (c_idx[common] > 5)]
    print(f"\nRescued across 5px line: {len(rescued)}  |  Broken across 5px line: {len(broken)}")
    for pid in rescued:
        print(f"  RESCUED {pid}: {b_idx[pid]:.2f}px -> {c_idx[pid]:.2f}px")
    for pid in broken:
        print(f"  BROKEN  {pid}: {b_idx[pid]:.2f}px -> {c_idx[pid]:.2f}px")

    summary = {
        "seed": SEED, "splits": SPLITS,
        "n": len(common),
        "baseline_acc5": float((b_idx[common] <= 5).mean()),
        "candidate_acc5": float((c_idx[common] <= 5).mean()),
        "rescued": list(rescued), "broken": list(broken),
        "per_split": {
            split: {
                "n": int((baseline_df["split"] == split).sum()),
                "baseline_acc5": float((baseline_df[baseline_df["split"] == split]["error_px"] <= 5).mean()),
                "candidate_acc5": float((candidate_df[candidate_df["split"] == split]["error_px"] <= 5).mean()),
            } for split in SPLITS
        },
    }
    with open(os.path.join(OUT_DIR, "second_seed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
