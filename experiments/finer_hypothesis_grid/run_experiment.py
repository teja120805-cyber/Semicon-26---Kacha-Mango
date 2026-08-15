"""Experiment A: rigorous re-validation of the finer scale/rotation
hypothesis grid (81 vs. 25 hypotheses).

Prior run (see REPORT.md history) found net rescue +5 on the frozen
benchmark's gate splits, failing only criterion 1 (validation tied at 90%
rather than improving - a ceiling effect, not a regression) once a
contention-contaminated runtime measurement was corrected.

This script answers the follow-up questions explicitly required before any
integration decision:
  1. Does the same benefit replicate on a genuinely fresh, independently-
     seeded dataset (not just the frozen benchmark)?
  2. Does it specifically help high-periodicity / no-boundary / rotation-
     scale / candidate-generation-failure cases, or is the net rescue
     concentrated somewhere unexpected?
  3. Is candidate recall itself improved, not just final accuracy?

Uses experiments/finer_hypothesis_grid/harness.py's instrumented pipeline
wrapper (mirrors pipeline/localize.py's own orchestration; never modifies
pipeline/) so baseline (25 hyp) and fine (81 hyp) get identical diagnostics
on both datasets.
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

from evaluation import metrics as eval_metrics  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from generator.dataset_generator import FAMILIES, generate_dataset  # noqa: E402
from experiments.finer_hypothesis_grid.harness import instrumented_localize  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
FRESH_DATA_DIR = os.path.join(EXP_DIR, "fresh_data")
FRESH_SEED = 424242  # deliberately different from the production seed (777001)
FRESH_SPLITS = ("validation", "held_out", "challenge")  # development is train-only; cross_generator is external/fixed

BASELINE_SCALE = (9.2, 9.6, 10.0, 10.4, 10.8)
BASELINE_ROTATION = (-5.0, -2.5, 0.0, 2.5, 5.0)
FINE_SCALE = (9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8)
FINE_ROTATION = (-5.0, -3.75, -2.5, -1.25, 0.0, 1.25, 2.5, 3.75, 5.0)

GT_TOLERANCE_PX = 5.0


def _bucket_rotation(deg: float) -> str:
    d = abs(deg)
    return "none" if d < 1e-9 else ("low" if d <= 1.5 else ("medium" if d <= 3.5 else "high"))


def _bucket_scale(s: float) -> str:
    d = abs(s - 1.0)
    return "none" if d < 1e-9 else ("low" if d <= 0.03 else ("medium" if d <= 0.06 else "high"))


def _bucket_periodicity(score) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "unknown"
    return "high" if score >= 0.66 else ("medium" if score >= 0.33 else "low")


def ensure_fresh_dataset() -> None:
    if os.path.isdir(FRESH_DATA_DIR) and all(
        os.path.isfile(os.path.join(FRESH_DATA_DIR, s, "ground_truth.json")) for s in FRESH_SPLITS
    ):
        print(f"Fresh dataset already present at {FRESH_DATA_DIR}, seed {FRESH_SEED} - skipping regeneration.")
        return
    print(f"Generating fresh independent dataset (seed={FRESH_SEED}) -> {FRESH_DATA_DIR}")
    families = [f for f in FAMILIES if f["split"] in FRESH_SPLITS]
    generate_dataset(FRESH_DATA_DIR, seed=FRESH_SEED, families=families, only_splits=list(FRESH_SPLITS), verbose=False)


def load_pairs(data_root: str, splits: list[str]) -> pd.DataFrame:
    frames = [load_manifest(data_root, s) for s in splits]
    return pd.concat(frames, ignore_index=True)


def run_variant(manifest: pd.DataFrame, scale_hyp, rotation_hyp) -> pd.DataFrame:
    records = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        diag = instrumented_localize(ref, search, row["gt_x"], row["gt_y"],
                                      scale_hypotheses=scale_hyp, rotation_hypotheses=rotation_hyp)
        rec = {**row.to_dict(), **diag}
        rec["rotation_bucket"] = _bucket_rotation(row.get("rotation_deg", 0.0))
        rec["scale_bucket"] = _bucket_scale(row.get("extra_scale", 1.0))
        rec["periodicity_bucket"] = _bucket_periodicity(row.get("periodicity_score"))
        rec["boundary"] = bool(row.get("crosses_mat_boundary", False) or row.get("crosses_strip_boundary", False))
        records.append(rec)
    return pd.DataFrame(records)


def rescue_break(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict:
    merged = baseline_df.merge(candidate_df, on="pair_id", suffixes=("_base", "_cand"))
    rescued = merged[(merged["error_px_base"] > GT_TOLERANCE_PX) & (merged["error_px_cand"] <= GT_TOLERANCE_PX)]
    broken = merged[(merged["error_px_base"] <= GT_TOLERANCE_PX) & (merged["error_px_cand"] > GT_TOLERANCE_PX)]
    cat_rescued = merged[(merged["error_px_base"] > 50) & (merged["error_px_cand"] <= 50)]
    cat_new = merged[(merged["error_px_base"] <= 50) & (merged["error_px_cand"] > 50)]
    return {
        "rescue_count": len(rescued), "break_count": len(broken), "net_rescue": len(rescued) - len(broken),
        "catastrophic_rescued": len(cat_rescued), "catastrophic_new": len(cat_new),
        "rescued_pair_ids": rescued["pair_id"].tolist(), "broken_pair_ids": broken["pair_id"].tolist(),
    }


def breakdown_by(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for val, sub in df.groupby(col):
        out[str(val)] = {
            "n": len(sub), "accuracy_at_5px": float((sub["error_px"] <= 5).mean()),
            "candidate_recall_at_5px": float(sub["gt_in_pool"].mean()),
            "failure_location_counts": sub["failure_location"].value_counts().to_dict(),
        }
    return out


def evaluate_dataset(name: str, manifest: pd.DataFrame) -> dict:
    print(f"\n=== {name}: baseline (25 hyp) ===")
    t0 = time.perf_counter()
    baseline_df = run_variant(manifest, BASELINE_SCALE, BASELINE_ROTATION)
    baseline_time = time.perf_counter() - t0
    print(f"  n={len(baseline_df)} acc@5px={(baseline_df['error_px']<=5).mean():.3f} "
          f"candidate_recall@5px={baseline_df['gt_in_pool'].mean():.3f} runtime/pair={baseline_time/len(baseline_df):.3f}s")

    print(f"=== {name}: fine grid (81 hyp) ===")
    t0 = time.perf_counter()
    fine_df = run_variant(manifest, FINE_SCALE, FINE_ROTATION)
    fine_time = time.perf_counter() - t0
    print(f"  n={len(fine_df)} acc@5px={(fine_df['error_px']<=5).mean():.3f} "
          f"candidate_recall@5px={fine_df['gt_in_pool'].mean():.3f} runtime/pair={fine_time/len(fine_df):.3f}s")

    rb = rescue_break(baseline_df, fine_df)
    print(f"  rescue={rb['rescue_count']} break={rb['break_count']} net={rb['net_rescue']}")

    return {
        "dataset": name, "n": len(manifest),
        "baseline": {
            "overall": eval_metrics.summarize(baseline_df),
            "candidate_recall_at_5px": float(baseline_df["gt_in_pool"].mean()),
            "by_periodicity": breakdown_by(baseline_df, "periodicity_bucket"),
            "by_boundary": breakdown_by(baseline_df, "boundary"),
            "by_rotation": breakdown_by(baseline_df, "rotation_bucket"),
            "by_scale": breakdown_by(baseline_df, "scale_bucket"),
            "failure_location_counts": baseline_df["failure_location"].value_counts().to_dict(),
            "mean_runtime_s": baseline_time / len(baseline_df),
        },
        "fine_grid": {
            "overall": eval_metrics.summarize(fine_df),
            "candidate_recall_at_5px": float(fine_df["gt_in_pool"].mean()),
            "by_periodicity": breakdown_by(fine_df, "periodicity_bucket"),
            "by_boundary": breakdown_by(fine_df, "boundary"),
            "by_rotation": breakdown_by(fine_df, "rotation_bucket"),
            "by_scale": breakdown_by(fine_df, "scale_bucket"),
            "failure_location_counts": fine_df["failure_location"].value_counts().to_dict(),
            "mean_runtime_s": fine_time / len(fine_df),
        },
        "rescue_break": rb,
        "baseline_df": baseline_df, "fine_df": fine_df,
    }


def clean_runtime_ratio(manifest: pd.DataFrame, n_sample: int = 12) -> float:
    """Interleaved timing (baseline/fine alternating on the same pairs) to
    get a contention-fair runtime ratio, independent of whatever else may
    be running concurrently."""
    sample = manifest.iloc[:n_sample]
    base_times, fine_times = [], []
    for _, row in sample.iterrows():
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
    ensure_fresh_dataset()

    frozen_manifest = load_pairs(os.path.join(PROJECT_ROOT, "data"),
                                  ["validation", "held_out", "challenge", "cross_generator"])
    fresh_manifest = load_pairs(FRESH_DATA_DIR, list(FRESH_SPLITS))

    results = {}
    for name, manifest in [("frozen_benchmark", frozen_manifest), ("fresh_independent_seed424242", fresh_manifest)]:
        results[name] = evaluate_dataset(name, manifest)

    print("\n=== Clean interleaved runtime ratio (contention-fair, frozen benchmark sample) ===")
    ratio = clean_runtime_ratio(frozen_manifest)
    print(f"  fine/baseline runtime ratio: {ratio:.2f}x")

    # Persist (drop the raw per-pair DataFrames from the JSON; save as CSV instead)
    summary = {}
    for name, r in results.items():
        baseline_df = r.pop("baseline_df")
        fine_df = r.pop("fine_df")
        baseline_df.to_csv(os.path.join(OUT_DIR, f"per_pair_baseline_{name}.csv"), index=False)
        fine_df.to_csv(os.path.join(OUT_DIR, f"per_pair_fine_{name}.csv"), index=False)
        summary[name] = r
    summary["clean_runtime_ratio"] = ratio
    with open(os.path.join(OUT_DIR, "revalidation_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote {OUT_DIR}/revalidation_results.json and per-pair CSVs.")


if __name__ == "__main__":
    main()
