#!/usr/bin/env python
"""A2 compliance experiment: widen the scale search to the literal
9:1-11:1 range the Applied Materials help doc and sponsor pptx both state
("Nominal 10:1 scale; robustness tests may span ~9:1-11:1"), instead of
the shipped +-7-8% ([0.93,1.07] dataset drift range / [9.2,10.8] hypothesis
grid - both fall about 2 percentage points short of the stated edge).

Two independent things change together, both widened to the SAME literal
edge, since testing one without the other doesn't answer the compliance
question:
  (a) DATASET: the `_scale_range` override on the three families that
      already exercise scale drift (ho_scale_drift, ch_combined_acquisition,
      ch_worst_case) widens from (0.93, 1.07) to (0.90, 1.10) - literal
      9:1-11:1 given the generator's exact 10x base ratio.
  (b) PIPELINE: candidate_generation's scale hypothesis grid widens from
      (9.2 ... 10.8, 9 points, step 0.2) to (9.0 ... 11.0, 11 points, same
      step 0.2) - same density/step-size convention as the shipped
      finer_hypothesis_grid change, just a wider span.

Evaluated on two datasets, both ways, exactly like center_tiebreak_v2's
run_experiment.py:
  1. Frozen benchmark (data/, seed 777001, n=156) - dataset is NOT widened
     here (can't retroactively change already-frozen ground truth), so
     this only tests (b) in isolation: does searching a wider scale grid
     regress anything on data that never needed the wider grid? Baseline
     and candidate here differ ONLY in scale_hypotheses.
  2. A fresh, independently-seeded dataset (seed 913442 - distinct from
     production 777001 and every prior experiment seed) where the three
     scale-drift families ARE widened per (a). validation/held_out/challenge
     only (cross_generator is external/fixed - no fresh analogue, same
     reasoning as every prior experiment). Baseline = old grid on this
     harder data (shows the real compliance gap); candidate = new grid on
     the same data (shows whether widening the grid closes it).

Both baseline and candidate call the SAME harness function
(harness.py::evaluate_with_hypotheses), which itself calls only unmodified
pipeline/ functions - the only difference anywhere in this script is which
scale_hypotheses tuple gets passed in, and (for the fresh dataset only)
which _scale_range the data was generated with.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import benchmark  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from generator.dataset_generator import FAMILIES, generate_dataset  # noqa: E402
from pipeline import candidate_generation  # noqa: E402
from experiments.scale_range_v1.harness import evaluate_with_hypotheses  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
FRESH_DATA_DIR = os.path.join(EXP_DIR, "fresh_data")
FRESH_SEED = 913442  # distinct from production (777001) and center_tiebreak_v2's fresh seed (647301)
FRESH_SPLITS = ("validation", "held_out", "challenge")

GATE_SPLITS_FROZEN = ("validation", "held_out", "challenge", "cross_generator")
SCALE_DRIFT_FAMILIES = ("ho_scale_drift", "ch_combined_acquisition", "ch_worst_case")
WIDENED_SCALE_RANGE = (0.90, 1.10)  # literal 9:1-11:1 given exact 10x base ratio

BASELINE_SCALE_HYPOTHESES = candidate_generation.DEFAULT_SCALE_HYPOTHESES  # (9.2 ... 10.8), 9 points
WIDENED_SCALE_HYPOTHESES = (9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0)  # 11 points, same step


def widened_families() -> list[dict]:
    fams = copy.deepcopy(FAMILIES)
    touched = []
    for fam in fams:
        if fam["name"] in SCALE_DRIFT_FAMILIES:
            fam["overrides"]["_scale_range"] = WIDENED_SCALE_RANGE
            touched.append(fam["name"])
    assert set(touched) == set(SCALE_DRIFT_FAMILIES), f"expected to widen {SCALE_DRIFT_FAMILIES}, touched {touched}"
    return fams


def ensure_fresh_dataset() -> None:
    if os.path.isdir(FRESH_DATA_DIR) and all(
        os.path.isfile(os.path.join(FRESH_DATA_DIR, s, "ground_truth.json")) for s in FRESH_SPLITS
    ):
        print(f"Fresh dataset already present at {FRESH_DATA_DIR}, seed {FRESH_SEED} - skipping regeneration.")
        return
    print(f"Generating fresh dataset with widened scale range (seed={FRESH_SEED}) -> {FRESH_DATA_DIR}")
    fams = [f for f in widened_families() if f["split"] in FRESH_SPLITS]
    generate_dataset(FRESH_DATA_DIR, seed=FRESH_SEED, families=fams, only_splits=list(FRESH_SPLITS), verbose=False)


def load_pairs(data_root: str, splits: list[str]) -> pd.DataFrame:
    return pd.concat([load_manifest(data_root, s) for s in splits], ignore_index=True)


def compare_gate_criteria(gate_a: dict, gate_b: dict, shared_keys: list[str]) -> bool:
    return all(gate_a["criteria"][k] == gate_b["criteria"][k] for k in shared_keys)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_fresh_dataset()

    # ---- Part 1: frozen benchmark, grid-only change ----
    frozen_manifest = load_pairs(os.path.join(PROJECT_ROOT, "data"), list(GATE_SPLITS_FROZEN))
    print(f"\n=== FROZEN benchmark (n={len(frozen_manifest)}, unwidened dataset - grid change only) ===")
    frozen_baseline = evaluate_with_hypotheses(frozen_manifest, BASELINE_SCALE_HYPOTHESES, "frozen/baseline_grid")

    # Integrity check vs this sandbox's own already-established baseline
    # (per Task #1: this sandbox's environment doesn't byte-match the
    # original shipped 71.2%@5px run - opencv/numpy minor-version drift on
    # near-tied scores is the suspected cause - so we validate the harness
    # against OUR OWN sandbox baseline, not the original production CSV,
    # which was never staged into this sandbox).
    sandbox_baseline_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results_SANDBOX_BASELINE.csv")
    integrity = {"checked": False}
    if os.path.isfile(sandbox_baseline_path):
        shipped = pd.read_csv(sandbox_baseline_path)
        shipped = shipped[shipped["split"].isin(GATE_SPLITS_FROZEN)][["pair_id", "error_px"]].rename(
            columns={"error_px": "error_px_shipped"})
        cmp = frozen_baseline[["pair_id", "error_px"]].merge(shipped, on="pair_id", how="inner")
        max_diff = float((cmp["error_px"] - cmp["error_px_shipped"]).abs().max()) if len(cmp) else float("nan")
        integrity = {"checked": True, "n_compared": len(cmp), "max_abs_error_px_diff": max_diff,
                     "matches_sandbox_baseline": bool(max_diff < 1e-6)}
        print(f"  Integrity check vs this sandbox's own baseline CSV: n={len(cmp)} max|diff|={max_diff:.2e} "
              f"({'MATCHES' if integrity['matches_sandbox_baseline'] else 'DIVERGES FROM'} sandbox baseline)")

    frozen_candidate = evaluate_with_hypotheses(frozen_manifest, WIDENED_SCALE_HYPOTHESES, "frozen/widened_grid")
    frozen_baseline.to_csv(os.path.join(OUT_DIR, "per_pair_frozen_baseline.csv"), index=False)
    frozen_candidate.to_csv(os.path.join(OUT_DIR, "per_pair_frozen_candidate.csv"), index=False)

    frozen_gate = benchmark.run_integration_gate(frozen_baseline, frozen_candidate, seeds_agree=None)
    print(f"  Frozen gate (seeds_agree pending): passed={frozen_gate['passed']}")
    print(json.dumps(frozen_gate["criteria"], indent=4))

    # ---- Part 2: fresh dataset, dataset widened AND grid widened together ----
    print(f"\n=== FRESH dataset (seed={FRESH_SEED}, n={len(FRESH_SPLITS)} splits, widened scale range) ===")
    fresh_manifest = load_pairs(FRESH_DATA_DIR, list(FRESH_SPLITS))
    fresh_baseline = evaluate_with_hypotheses(fresh_manifest, BASELINE_SCALE_HYPOTHESES, "fresh/baseline_grid")
    fresh_candidate = evaluate_with_hypotheses(fresh_manifest, WIDENED_SCALE_HYPOTHESES, "fresh/widened_grid")
    fresh_baseline.to_csv(os.path.join(OUT_DIR, "per_pair_fresh_baseline.csv"), index=False)
    fresh_candidate.to_csv(os.path.join(OUT_DIR, "per_pair_fresh_candidate.csv"), index=False)

    fresh_gate = benchmark.run_integration_gate(fresh_baseline, fresh_candidate, seeds_agree=None)
    print(f"  Fresh gate (seeds_agree pending): passed={fresh_gate['passed']}")
    print(json.dumps(fresh_gate["criteria"], indent=4))

    # ---- Combine: criterion 7 stand-in (same reasoning as center_tiebreak_v2) ----
    shared_keys = ["1_improves_validation", "2_improves_held_out", "4_no_catastrophic_increase",
                   "5_no_per_family_regression", "6_acceptable_runtime"]
    seeds_agree = compare_gate_criteria(frozen_gate, fresh_gate, shared_keys)
    frozen_gate_final = benchmark.run_integration_gate(frozen_baseline, frozen_candidate, seeds_agree=seeds_agree)
    fresh_gate_final = benchmark.run_integration_gate(fresh_baseline, fresh_candidate, seeds_agree=seeds_agree)

    print(f"\n=== FINAL (seeds_agree across frozen/fresh = {seeds_agree}) ===")
    print(f"  Frozen gate passed: {frozen_gate_final['passed']}")
    print(f"  Fresh gate passed:  {fresh_gate_final['passed']}")

    summary = {
        "integrity_check": integrity,
        "frozen_gate": frozen_gate_final,
        "fresh_gate": fresh_gate_final,
        "seeds_agree": seeds_agree,
        "fresh_seed": FRESH_SEED,
        "widened_scale_range": WIDENED_SCALE_RANGE,
        "baseline_scale_hypotheses": list(BASELINE_SCALE_HYPOTHESES),
        "widened_scale_hypotheses": list(WIDENED_SCALE_HYPOTHESES),
    }
    with open(os.path.join(OUT_DIR, "gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/gate_summary.json")


if __name__ == "__main__":
    main()
