"""Recalibrate the ambiguity flag. Fit on tuning surfaces, evaluate ONCE on
held-back data.

`reports/PROJECT_STATUS.md` lists `AMBIGUITY_THRESHOLD = 0.92` as
miscalibrated - "fires on 128/156 pairs at 31% precision". Measured here on
`tune_degraded`: 37/40 pairs flagged, precision 0.324. Reproduced
independently, so it is a real property of the constant and not of that one
benchmark.

What the measurement in run.py adds is WHY. `ambiguity_ratio` is a good
statistic - AUC 0.949 at separating correct from wrong - it is just being
thresholded in the wrong place. The observed range is 0.816-0.999 with a
median of 0.985, so a 0.92 cut sits far below the bulk of the distribution
and flags almost everything. The fix is a number, not a new statistic.

(This also settles P4's proposal in the negative: PSR was offered as the
principled replacement, and measures AUC 0.577 - barely above chance -
against the existing pool gap's 0.964. See REPORT.md §2.)

Protocol, so the threshold is not fitted to the surface it is scored on:

  fit    `development` (24) + `tune_degraded` (40), pooled
  test   `validate_fresh` (40), read exactly once, after the rule is fixed

The selection rule is fixed in code below BEFORE any test data is read.

    python -m experiments.psr_confidence.calibrate
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

FIT_SURFACES = ("development", "tune_degraded")
TEST_SURFACE = "validate_fresh"
PRODUCTION_THRESHOLD = 0.92

# Fixed before reading the test surface: among thresholds meeting a minimum
# failure-recall bar, take the one with the best precision. Recall is the
# safety property (a missed failure is silently wrong), precision is the
# usability property (over-flagging makes the flag meaningless), so recall
# is a constraint and precision the objective.
MIN_FAILURE_RECALL = 0.80


def load(surface: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUT, f"{surface}_psr.csv"))
    df["surface"] = surface
    return df


def curve(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    wrong = ~df.decisiveness_correct.values
    amb = df.ambiguity_ratio.values
    rows = []
    for t in thresholds:
        flagged = amb >= t
        n_flag = int(flagged.sum())
        tp = int((flagged & wrong).sum())
        rows.append({
            "threshold": float(t),
            "flag_rate": n_flag / len(df),
            "precision": tp / n_flag if n_flag else float("nan"),
            "failure_recall": tp / int(wrong.sum()) if wrong.sum() else float("nan"),
            "answered_frac": 1.0 - n_flag / len(df),
            "answered_accuracy": (float((~flagged & ~wrong).sum() / (len(df) - n_flag))
                                   if n_flag < len(df) else float("nan")),
        })
    return pd.DataFrame(rows)


def main() -> None:
    fit = pd.concat([load(s) for s in FIT_SURFACES], ignore_index=True)
    thresholds = np.round(np.arange(0.90, 1.0001, 0.002), 4)
    fit_curve = curve(fit, thresholds)
    fit_curve.to_csv(os.path.join(OUT, "calibration_fit_curve.csv"), index=False)

    print(f"FIT on {' + '.join(FIT_SURFACES)}  (n={len(fit)}, "
          f"failures={int((~fit.decisiveness_correct).sum())})")
    print(f"  production threshold {PRODUCTION_THRESHOLD}:")
    prod_row = curve(fit, [PRODUCTION_THRESHOLD]).iloc[0]
    print(f"    flag rate {prod_row.flag_rate:.3f}  precision {prod_row.precision:.3f}  "
          f"failure recall {prod_row.failure_recall:.3f}")

    eligible = fit_curve[fit_curve.failure_recall >= MIN_FAILURE_RECALL]
    if eligible.empty:
        raise SystemExit("no threshold meets the recall bar on the fit surfaces")
    chosen = eligible.loc[eligible.precision.idxmax()]
    t = float(chosen.threshold)
    print(f"\n  selection rule: max precision subject to failure recall >= {MIN_FAILURE_RECALL}")
    print(f"  CHOSEN THRESHOLD = {t:.4f}")
    print(f"    flag rate {chosen.flag_rate:.3f}  precision {chosen.precision:.3f}  "
          f"failure recall {chosen.failure_recall:.3f}")
    print(f"    answers {chosen.answered_frac:.1%} of pairs at "
          f"{chosen.answered_accuracy:.1%} accuracy")

    print("\n  fit curve (every 0.01)")
    print(f"{'thresh':>8} {'flag rate':>10} {'precision':>10} {'recall':>8} "
          f"{'answered':>9} {'acc on answered':>16}")
    for _, r in fit_curve[np.isclose(fit_curve.threshold % 0.01, 0, atol=1e-9)].iterrows():
        print(f"{r.threshold:>8.3f} {r.flag_rate:>10.3f} {r.precision:>10.3f} "
              f"{r.failure_recall:>8.3f} {r.answered_frac:>9.3f} {r.answered_accuracy:>16.3f}")

    # ---- test: read once, after the threshold is fixed ---------------------
    test = load(TEST_SURFACE)
    test_prod = curve(test, [PRODUCTION_THRESHOLD]).iloc[0]
    test_new = curve(test, [t]).iloc[0]
    print(f"\n\nTEST on {TEST_SURFACE} (n={len(test)}, "
          f"failures={int((~test.decisiveness_correct).sum())}) - read once, threshold already fixed")
    print(f"{'':>26} {'flag rate':>10} {'precision':>10} {'recall':>8} {'answered':>9} {'acc':>7}")
    for label, r in ((f"production ({PRODUCTION_THRESHOLD})", test_prod),
                     (f"recalibrated ({t:.3f})", test_new)):
        print(f"{label:>26} {r.flag_rate:>10.3f} {r.precision:>10.3f} {r.failure_recall:>8.3f} "
              f"{r.answered_frac:>9.3f} {r.answered_accuracy:>7.3f}")

    result = {
        "fit_surfaces": list(FIT_SURFACES), "test_surface": TEST_SURFACE,
        "selection_rule": f"max precision s.t. failure_recall >= {MIN_FAILURE_RECALL}",
        "production_threshold": PRODUCTION_THRESHOLD, "chosen_threshold": t,
        "fit": {k: float(chosen[k]) for k in
                ("flag_rate", "precision", "failure_recall", "answered_frac", "answered_accuracy")},
        "fit_production": {k: float(prod_row[k]) for k in
                            ("flag_rate", "precision", "failure_recall", "answered_frac", "answered_accuracy")},
        "test_recalibrated": {k: float(test_new[k]) for k in
                               ("flag_rate", "precision", "failure_recall", "answered_frac", "answered_accuracy")},
        "test_production": {k: float(test_prod[k]) for k in
                             ("flag_rate", "precision", "failure_recall", "answered_frac", "answered_accuracy")},
    }
    with open(os.path.join(OUT, "calibration_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'calibration_result.json')}")


if __name__ == "__main__":
    main()
