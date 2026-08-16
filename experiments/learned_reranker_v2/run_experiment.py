#!/usr/bin/env python
"""Evaluates the properly-data-scaled embedding re-ranker (trained by
train.py on the expanded ~180-pair development set) against the frozen
benchmark, for every one of the 3 training seeds independently - exactly
mirroring embedding_reranker_v1's own "all 3 seeds" evaluation discipline
(reports/V2_MODEL_EVALUATION_REPORT.md) and reusing the EXACT SAME
production code path a real integration would use:
pipeline.localize.localize(ranking_mode="learned", model=...) via
evaluation.evaluate.evaluate_all - no custom harness needed here, unlike
the other experiments in this campaign, because ranking_mode="learned" is
already a first-class, unmodified option in production localize().

`seeds_agree` (integration gate criterion 7) is computed honestly here: True
only if the gate's pass/fail verdict is identical across all 3 seeds - an
unverified or inconsistent claim of stability must not count as stable.

Never touches pipeline/, generator/, model/, or data/ - only reads/writes
under this experiment's own data/ and outputs/ + checkpoints/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
import torch

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import evaluate_all  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402
from model.architecture import EmbeddingNet  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

SEEDS = (20260101, 20260102, 20260103)
# rank_with_model only ever re-ranks the classical candidate pool of
# validation/held_out/challenge/cross_generator - development is excluded
# here for the SAME reason model/train.py never validates on it: it's
# training data (for the ORIGINAL production development split at least;
# this experiment's own expanded set lives entirely under
# experiments/learned_reranker_v2/data/, never data/development/, so there
# is no overlap either way, but the convention is kept identical to
# scripts/evaluate_model.py for direct comparability).
EVAL_SPLITS = ["validation", "held_out", "challenge", "cross_generator"]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline_df = pd.read_csv(BASELINE_CSV)

    per_seed_results = {}
    for seed in SEEDS:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"embedding_net_seed{seed}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: checkpoint {checkpoint_path} not found - skipping seed {seed} "
                  f"(run train.py first)")
            continue

        print(f"\n=== Evaluating seed {seed} ===")
        model = EmbeddingNet()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        t0 = time.perf_counter()
        candidate_df = evaluate_all(DATA_ROOT, EVAL_SPLITS, OUT_DIR, ranking_mode="learned",
                                     model=model, verbose=False)
        print(f"  evaluated {len(candidate_df)} pairs in {time.perf_counter() - t0:.1f}s")

        candidate_csv = os.path.join(OUT_DIR, f"per_pair_results_learned_seed{seed}.csv")
        candidate_df.to_csv(candidate_csv, index=False)

        gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
        acc5 = float((candidate_df["error_px"] <= 5).mean())
        b_acc5_matched = float((baseline_df[baseline_df["split"].isin(EVAL_SPLITS)]["error_px"] <= 5).mean())
        print(f"  candidate acc@5px (matched splits, n={len(candidate_df)}): {acc5:.4f} "
              f"(baseline on same splits: {b_acc5_matched:.4f})")
        print(f"  gate passed={gate['passed']}  criteria={gate['criteria']}")

        per_seed_results[seed] = {
            "acc_at_5px": acc5, "baseline_acc_at_5px_matched_splits": b_acc5_matched,
            "gate_passed": gate["passed"], "gate_criteria": gate["criteria"],
            "per_split": gate["per_split"], "csv": candidate_csv,
        }

    if not per_seed_results:
        print("\nNo checkpoints found - nothing evaluated. Run train.py first.")
        return

    verdicts = [r["gate_passed"] for r in per_seed_results.values()]
    seeds_agree = len(set(verdicts)) == 1
    print(f"\n=== Cross-seed stability (integration gate criterion 7) ===")
    print(f"seeds evaluated: {list(per_seed_results.keys())}")
    print(f"gate verdicts: {verdicts}")
    print(f"seeds_agree: {seeds_agree}")

    summary = {"per_seed": {str(k): v for k, v in per_seed_results.items()}, "seeds_agree": seeds_agree}
    with open(os.path.join(OUT_DIR, "cross_seed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
