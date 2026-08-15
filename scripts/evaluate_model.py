#!/usr/bin/env python
"""Run the classical baseline over every split, compute metrics/breakdowns,
and generate all required plots - the exact commands behind
reports/V2_BASELINE_REPORT.md. Pass --learned-checkpoint to also evaluate a
trained model and run the integration gate against the frozen classical
baseline (reports/V2_MODEL_EVALUATION_REPORT.md).

    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --learned-checkpoint experiments/embedding_reranker_v1/checkpoints/embedding_net_seed20260101.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation import benchmark, metrics, plots  # noqa: E402
from evaluation.evaluate import evaluate_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--splits", default="development,validation,held_out,challenge,cross_generator")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "reports"))
    parser.add_argument("--plots-out", default=os.path.join(PROJECT_ROOT, "outputs", "plots"))
    parser.add_argument("--learned-checkpoint", default=None)
    args = parser.parse_args()

    splits = args.splits.split(",")

    print("=== Classical baseline ===")
    baseline_df = evaluate_all(args.data_root, splits, args.out, ranking_mode="classical")
    baseline_report = metrics.full_report(baseline_df)
    with open(os.path.join(args.out, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_report, f, indent=2)
    plots.generate_all_plots(baseline_df, args.plots_out)
    print(json.dumps(baseline_report["overall"], indent=2))

    # Selected once, here, deterministically - the Streamlit app's Failure
    # Analysis page reads this file rather than re-picking cases itself.
    failure_cases = metrics.select_failure_cases(baseline_df)
    with open(os.path.join(args.out, "failure_analysis_cases.json"), "w") as f:
        json.dump(failure_cases, f, indent=2)

    if args.learned_checkpoint:
        import torch
        from model.architecture import EmbeddingNet

        print(f"\n=== Learned re-ranker ({args.learned_checkpoint}) ===")
        model = EmbeddingNet()
        model.load_state_dict(torch.load(args.learned_checkpoint, map_location="cpu"))
        model.eval()
        learned_splits = [s for s in splits if s != "development"]
        learned_df = evaluate_all(args.data_root, learned_splits, args.out, ranking_mode="learned", model=model)

        gate = benchmark.run_integration_gate(baseline_df, learned_df, seeds_agree=None)
        with open(os.path.join(args.out, "integration_gate.json"), "w") as f:
            json.dump(gate, f, indent=2)
        print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
