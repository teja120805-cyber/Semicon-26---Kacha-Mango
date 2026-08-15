"""Phases 3-5: evaluates production classical (25-hypothesis) vs. the
finer_hypothesis_grid candidate (81-hypothesis) on the targeted validation
set(s), using the exact same instrumented pipeline wrapper
(experiments/finer_hypothesis_grid/harness.py::instrumented_localize) so
both variants get identical images/ground truth and diagnostics. Computes
the full required metric set, per-family breakdown, and a failure-level
classification (rescue/break/unchanged + mechanism) for every pair where
the two variants disagree.

Never modifies production pipeline/, generator/, or data/ - only imports
pipeline.candidate_generation/ranking/refinement unmodified, exactly as
finer_hypothesis_grid already does.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from experiments.finer_hypothesis_grid.harness import instrumented_localize  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")

BASELINE_SCALE = (9.2, 9.6, 10.0, 10.4, 10.8)
BASELINE_ROTATION = (-5.0, -2.5, 0.0, 2.5, 5.0)
FINE_SCALE = (9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8)
FINE_ROTATION = (-5.0, -3.75, -2.5, -1.25, 0.0, 1.25, 2.5, 3.75, 5.0)

ACCURACY_TOLERANCES_PX = (1, 2, 3, 5, 10)
RECALL_TOLERANCES_PX = (1, 2, 5, 10, 20)


def full_metrics(df: pd.DataFrame) -> dict:
    err = df["error_px"].to_numpy()
    out = {"n": int(len(df))}
    for tol in ACCURACY_TOLERANCES_PX:
        out[f"accuracy_at_{tol}px"] = float(np.mean(err <= tol))
    out["median_error_px"] = float(np.median(err))
    out["mean_error_px"] = float(np.mean(err))
    out["p90_error_px"] = float(np.percentile(err, 90))
    out["p95_error_px"] = float(np.percentile(err, 95))
    out["max_error_px"] = float(np.max(err))
    out["failure_rate_gt_10px"] = float(np.mean(err > 10))
    out["failure_rate_gt_50px"] = float(np.mean(err > 50))
    out["mean_runtime_s"] = float(df["runtime_s"].mean())
    for tol in RECALL_TOLERANCES_PX:
        out[f"candidate_recall_at_{tol}px"] = float(np.mean(df["gt_nearest_candidate_dist_px"] <= tol))
    return out


def run_variant(manifest: pd.DataFrame, scale_hyp, rotation_hyp) -> pd.DataFrame:
    records = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        diag = instrumented_localize(ref, search, row["gt_x"], row["gt_y"],
                                      scale_hypotheses=scale_hyp, rotation_hypotheses=rotation_hyp)
        rec = {**row.to_dict(), **diag}
        records.append(rec)
    return pd.DataFrame(records)


def classify_change(base_row: pd.Series, fine_row: pd.Series) -> dict:
    """For a pair where the two variants disagree: rescue/break/unchanged,
    plus a best-effort mechanism label for rescues and breaks."""
    base_ok = base_row["error_px"] <= 5
    fine_ok = fine_row["error_px"] <= 5
    if base_ok == fine_ok:
        return {"change": "unchanged", "mechanism": None}

    change = "rescue" if fine_ok else "break"
    if change == "rescue":
        # Why was baseline wrong, and what changed?
        if not base_row["gt_in_pool"]:
            if fine_row["gt_in_pool"]:
                mechanism = "candidate_generation_recovered"
            else:
                mechanism = "other_gt_still_not_in_pool_but_final_error_ok"  # refinement/tie-break edge case
        elif not base_row["winner_is_gt_candidate"]:
            # GT was in pool but didn't win under baseline grid - did a finer
            # hypothesis make its score competitive (grid-misalignment fix)?
            mechanism = "hypothesis_grid_misalignment_corrected"
        else:
            mechanism = "refinement_or_other"
    else:
        # break: fine grid made a previously-correct pair wrong.
        if not fine_row["gt_in_pool"]:
            mechanism = "gt_dropped_from_pool_under_finer_grid"
        elif not fine_row["winner_is_gt_candidate"]:
            mechanism = "new_hypothesis_created_a_competing_wrong_peak"
        else:
            mechanism = "refinement_or_other"

    return {"change": change, "mechanism": mechanism}


def evaluate_one_dataset(name: str, data_root: str) -> dict:
    manifest = load_manifest(data_root, "validation")
    print(f"\n{'='*10} {name}: n={len(manifest)} {'='*10}")

    t0 = time.perf_counter()
    baseline_df = run_variant(manifest, BASELINE_SCALE, BASELINE_ROTATION)
    baseline_wall = time.perf_counter() - t0
    print(f"  baseline: acc@5px={(baseline_df['error_px']<=5).mean():.3f} "
          f"runtime/pair={baseline_wall/len(manifest):.3f}s")

    t0 = time.perf_counter()
    fine_df = run_variant(manifest, FINE_SCALE, FINE_ROTATION)
    fine_wall = time.perf_counter() - t0
    print(f"  fine grid: acc@5px={(fine_df['error_px']<=5).mean():.3f} "
          f"runtime/pair={fine_wall/len(manifest):.3f}s")

    baseline_df.to_csv(os.path.join(OUT_DIR, f"per_pair_baseline_{name}.csv"), index=False)
    fine_df.to_csv(os.path.join(OUT_DIR, f"per_pair_fine_{name}.csv"), index=False)

    # Phase 4: failure-level classification for every differing pair.
    changes = []
    for i in range(len(manifest)):
        base_row, fine_row = baseline_df.iloc[i], fine_df.iloc[i]
        c = classify_change(base_row, fine_row)
        c["pair_id"] = base_row["pair_id"]
        c["structural_family"] = base_row["structural_family"]
        c["base_error_px"] = float(base_row["error_px"])
        c["fine_error_px"] = float(fine_row["error_px"])
        changes.append(c)
    changes_df = pd.DataFrame(changes)
    changes_df.to_csv(os.path.join(OUT_DIR, f"failure_level_changes_{name}.csv"), index=False)

    rescues = changes_df[changes_df["change"] == "rescue"]
    breaks = changes_df[changes_df["change"] == "break"]
    print(f"  rescue={len(rescues)} break={len(breaks)} net={len(rescues)-len(breaks)}")
    print(f"  rescue mechanisms: {rescues['mechanism'].value_counts().to_dict()}")
    print(f"  break mechanisms: {breaks['mechanism'].value_counts().to_dict()}")

    # Per-family breakdown
    per_family = {}
    for fam in sorted(manifest["structural_family"].unique()):
        b = baseline_df[baseline_df["structural_family"] == fam]
        f = fine_df[fine_df["structural_family"] == fam]
        per_family[fam] = {
            "n": len(b),
            "baseline_acc5px": float((b["error_px"] <= 5).mean()),
            "fine_acc5px": float((f["error_px"] <= 5).mean()),
        }

    return {
        "dataset": name, "n": len(manifest),
        "baseline_metrics": full_metrics(baseline_df),
        "fine_metrics": full_metrics(fine_df),
        "baseline_runtime_per_pair_s": baseline_wall / len(manifest),
        "fine_runtime_per_pair_s": fine_wall / len(manifest),
        "runtime_ratio": (fine_wall / len(manifest)) / (baseline_wall / len(manifest)),
        "rescue_count": len(rescues), "break_count": len(breaks), "net_rescue": len(rescues) - len(breaks),
        "rescue_mechanisms": rescues["mechanism"].value_counts().to_dict(),
        "break_mechanisms": breaks["mechanism"].value_counts().to_dict(),
        "per_family": per_family,
    }


def clean_runtime_ratio(data_root: str, n_sample: int = 12) -> float:
    manifest = load_manifest(data_root, "validation").iloc[:n_sample]
    base_times, fine_times = [], []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        t0 = time.perf_counter()
        instrumented_localize(ref, search, row["gt_x"], row["gt_y"],
                               scale_hypotheses=BASELINE_SCALE, rotation_hypotheses=BASELINE_ROTATION)
        base_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        instrumented_localize(ref, search, row["gt_x"], row["gt_y"],
                               scale_hypotheses=FINE_SCALE, rotation_hypotheses=FINE_ROTATION)
        fine_times.append(time.perf_counter() - t0)
    return float(np.mean(fine_times) / np.mean(base_times))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    datasets = [("dataset_A", os.path.join(EXP_DIR, "data")), ("dataset_B", os.path.join(EXP_DIR, "data_b"))]
    for name, root in datasets:
        if os.path.isfile(os.path.join(root, "validation", "ground_truth.json")):
            results[name] = evaluate_one_dataset(name, root)
        else:
            print(f"Skipping {name}: {root} not found yet.")

    if "dataset_A" in results:
        print("\n=== Clean interleaved runtime ratio (dataset A sample) ===")
        results["clean_runtime_ratio_dataset_A"] = clean_runtime_ratio(os.path.join(EXP_DIR, "data"))
        print(f"  {results['clean_runtime_ratio_dataset_A']:.2f}x")

    with open(os.path.join(OUT_DIR, "validation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/validation_results.json")


if __name__ == "__main__":
    main()
