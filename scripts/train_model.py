#!/usr/bin/env python
"""Train the V2 embedding re-ranking model, optionally across the 3 seeds
`model/TRAINING_PROTOCOL.md` requires for the integration gate's stability
criterion.

    python scripts/train_model.py --seeds 20260101,20260102,20260103
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument(
        "--out", default=os.path.join(PROJECT_ROOT, "experiments", "embedding_reranker_v1", "checkpoints")
    )
    parser.add_argument("--seeds", default="20260101", help="Comma-separated list of seeds to train.")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    for seed in (int(s) for s in args.seeds.split(",")):
        print(f"=== training seed {seed} ===")
        summary = train(args.data_root, args.out, seed=seed, epochs=args.epochs)
        print(f"best_dev_early_stop_loss={summary['best_dev_early_stop_loss']:.4f} "
              f"checkpoint={summary['checkpoint_path']}")


if __name__ == "__main__":
    main()
