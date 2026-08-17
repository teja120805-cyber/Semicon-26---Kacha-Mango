#!/usr/bin/env python
"""Is the remaining error algorithmic, or is the task ill-posed?

The oracle diagnostic showed every failure is a near-tie and that 45% of
failures never even get the true location proposed. Profiling those pairs
pointed at one metadata field: `uniqueness_score`. This experiment tests
the resulting hypothesis directly and algorithm-free.

Stages:
  1. Accuracy as a function of uniqueness_score, and the periodicity
     cross-tab that shows periodicity is a confound rather than a cause.
  2. Per-pair image-content measurements (multiplicity.py):
       identity(gt, pred) - are the two locations the same picture?
       K@thr               - how many distinct locations match ground truth?
     Both compare Search content against Search content, so they bound
     what ANY method could achieve on this data.
  3. The implied ceiling: expected accuracy under forced choice is
     sum(1/K) / n, since a pair with K equally-valid answers cannot be
     resolved better than by guessing among them.
  4. An abstention policy: use the pool-internal gap detector from
     experiments/oracle_ceiling_diagnostic to decline to answer instead of
     emitting a confident wrong coordinate, and measure the accuracy of
     what remains.

Never modifies pipeline/, generator/, model/, or data/.
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
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplicity import PATCH_PX, extract_patch, multiplicity, zncc  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
GAP_CSV = os.path.join(PROJECT_ROOT, "experiments", "oracle_ceiling_diagnostic",
                       "outputs", "margin_detector.csv")
THRESHOLDS = (0.85, 0.90, 0.95)


def stage1_uniqueness_table(b: pd.DataFrame) -> dict:
    print("=== Stage 1: accuracy vs uniqueness_score ===")
    t = b.groupby(b.uniqueness_score.round(2)).agg(n=("correct", "size"), acc=("correct", "mean"))
    print(t.round(3).to_string())
    b = b.assign(uz=b.uniqueness_score <= 0.001, per_hi=b.periodicity_score > 0.8)
    cross = b.groupby(["uz", "per_hi"]).agg(n=("correct", "size"), acc=("correct", "mean"))
    print("\nuniqueness x periodicity (periodicity is a confound, not a cause):")
    print(cross.round(3).to_string())
    return {
        "by_uniqueness": {str(k): {"n": int(v.n), "acc": float(v.acc)} for k, v in t.iterrows()},
        "uniqueness_zero": {"n": int((b.uz).sum()), "acc": float(b[b.uz].correct.mean())},
        "uniqueness_pos": {"n": int((~b.uz).sum()), "acc": float(b[~b.uz].correct.mean())},
        "cross_uniqueness_periodicity": {
            f"uz={k[0]},per_hi={k[1]}": {"n": int(v.n), "acc": float(v.acc)}
            for k, v in cross.iterrows()},
    }


def stage2_content(b: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Stage 2: image-content measurements (algorithm-free) ===")
    rows = []
    t0 = time.perf_counter()
    for i, row in b.iterrows():
        search = cv2.imread(os.path.join(DATA_ROOT, row["search_path"]), cv2.IMREAD_UNCHANGED)
        search = search.astype(np.float32)
        gt_patch = extract_patch(search, row["gt_x"], row["gt_y"])
        pred_patch = extract_patch(search, row["pred_x"], row["pred_y"])
        rec = {"pair_id": row["pair_id"], "split": row["split"],
               "family": row["pair_id"].rsplit("_", 1)[0],
               "uniqueness_score": float(row["uniqueness_score"]),
               "periodicity_score": float(row["periodicity_score"]),
               "error_px": float(row["error_px"]), "correct": bool(row["correct"]),
               "identity_gt_pred": zncc(gt_patch, pred_patch)}
        if gt_patch is not None:
            ks, _ = multiplicity(search, gt_patch, THRESHOLDS)
            for thr in THRESHOLDS:
                rec[f"K@{thr}"] = ks[thr]
        else:
            for thr in THRESHOLDS:
                rec[f"K@{thr}"] = np.nan
        rows.append(rec)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(b)} ({time.perf_counter() - t0:.0f}s)")
    return pd.DataFrame(rows)


def stage3_ceiling(m: pd.DataFrame, thr: float = 0.90) -> dict:
    print(f"\n=== Stage 3: implied ceiling from multiplicity (thr={thr}) ===")
    k = m[f"K@{thr}"].dropna()
    exp_acc = float((1.0 / k).mean())
    uniq = m[m[f"K@{thr}"] == 1]
    amb = m[m[f"K@{thr}"] > 1]
    print(f"  pairs with a UNIQUE origin (K==1): {len(uniq)}/{len(m)}  "
          f"observed acc {uniq.correct.mean():.4f}")
    print(f"  pairs that are AMBIGUOUS  (K>1) : {len(amb)}/{len(m)}  "
          f"observed acc {amb.correct.mean():.4f}   median K={amb[f'K@{thr}'].median():.0f}")
    print(f"  expected accuracy if ambiguous pairs are guessed uniformly: {exp_acc:.4f}")
    print(f"  actual pooled accuracy: {m.correct.mean():.4f}")
    return {
        "threshold": thr,
        "n_unique": int(len(uniq)), "acc_unique": float(uniq.correct.mean()),
        "n_ambiguous": int(len(amb)),
        "acc_ambiguous": float(amb.correct.mean()) if len(amb) else float("nan"),
        "median_K_ambiguous": float(amb[f"K@{thr}"].median()) if len(amb) else float("nan"),
        "expected_acc_uniform_guess": exp_acc,
        "actual_acc": float(m.correct.mean()),
    }


def stage4_abstention(m: pd.DataFrame) -> dict:
    print("\n=== Stage 4: abstention using the pool-internal gap detector ===")
    if not os.path.exists(GAP_CSV):
        print("  (gap CSV not found - run experiments/oracle_ceiling_diagnostic first)")
        return {}
    gap = pd.read_csv(GAP_CSV)[["pair_id", "gap"]]
    d = m.merge(gap, on="pair_id")
    out = {}
    print(f"  {'thr':>7} {'answered':>9} {'coverage':>9} {'acc|answered':>13} {'wrong kept':>11}")
    for thr in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
        ans = d[d.gap >= thr]
        out[str(thr)] = {
            "answered": int(len(ans)), "coverage": float(len(ans) / len(d)),
            "acc_on_answered": float(ans.correct.mean()) if len(ans) else float("nan"),
            "wrong_still_answered": int((~ans.correct).sum()),
        }
        print(f"  {thr:>7.3f} {len(ans):>9d} {len(ans) / len(d):>8.1%} "
              f"{ans.correct.mean():>12.4f} {int((~ans.correct).sum()):>11d}")
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    b = pd.read_csv(BASELINE_CSV)
    b["correct"] = b["error_px"] <= 5.0

    summary = {"stage1_uniqueness": stage1_uniqueness_table(b)}
    m = stage2_content(b.reset_index(drop=True))
    m.to_csv(os.path.join(OUT_DIR, "content_multiplicity.csv"), index=False)

    print("\n=== identity(gt, pred): is the predicted spot the same picture? ===")
    f = m[~m.correct]
    print(f"  failures  median {f.identity_gt_pred.median():.4f}   "
          f"q25 {f.identity_gt_pred.quantile(.25):.4f}   max {f.identity_gt_pred.max():.4f}")
    for t in (0.80, 0.90, 0.95):
        print(f"    failures with identity > {t:.2f}: {int((f.identity_gt_pred > t).sum())}/{len(f)}")
    summary["identity_gt_pred_failures"] = {
        "median": float(f.identity_gt_pred.median()),
        "q25": float(f.identity_gt_pred.quantile(.25)),
        "frac_above_0.90": float((f.identity_gt_pred > 0.90).mean()),
        "frac_above_0.95": float((f.identity_gt_pred > 0.95).mean()),
    }

    summary["stage3_ceiling"] = {str(t): stage3_ceiling(m, t) for t in THRESHOLDS}
    summary["stage4_abstention"] = stage4_abstention(m)

    with open(os.path.join(OUT_DIR, "uniqueness_ceiling_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {OUT_DIR}/uniqueness_ceiling_summary.json")


if __name__ == "__main__":
    main()
