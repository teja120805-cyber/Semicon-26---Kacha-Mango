"""Trains the candidate-generator embedding network on the expanded,
16-family dev_data/ (experiments/learned_candidate_generator/dev_data/),
never on validation/held_out/challenge/cross_generator. Reuses
model/architecture.py::EmbeddingNet unmodified (same 64,992-param CNN the
rejected embedding_reranker_v1 used) - this experiment tests whether more
diverse training data changes the outcome, not whether a bigger model does.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.architecture import EmbeddingNet  # noqa: E402
from experiments.learned_candidate_generator.dataset import ExpandedTripletPatchDataset  # noqa: E402
from experiments.learned_candidate_generator.dataset_config import (  # noqa: E402
    ALL_FAMILY_NAMES, INTERNAL_EARLY_STOP_FAMILIES, TRAIN_FAMILIES,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_DATA_DIR = os.path.join(EXP_DIR, "dev_data")
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epoch_loss(model, loader, device, margin: float, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, n = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            if is_train:
                optimizer.zero_grad()
            ea, ep, en = model(anchor), model(positive), model(negative)
            loss = F.triplet_margin_loss(ea, ep, en, margin=margin)
            if is_train:
                loss.backward()
                optimizer.step()
            total += loss.item() * anchor.size(0)
            n += anchor.size(0)
    return total / max(n, 1)


def train(seed: int = 20260201, epochs: int = 60, batch_size: int = 16, lr: float = 1e-3,
          margin: float = 0.3, patience: int = 10, device: str = "cpu", verbose: bool = True) -> dict:
    set_seed(seed)
    dataset = ExpandedTripletPatchDataset(DEV_DATA_DIR, ALL_FAMILY_NAMES, seed=seed)

    train_idx = [i for i, it in enumerate(dataset.items) if it["structural_family"] in TRAIN_FAMILIES]
    early_stop_idx = [i for i, it in enumerate(dataset.items) if it["structural_family"] in INTERNAL_EARLY_STOP_FAMILIES]
    if verbose:
        print(f"Dataset: {len(dataset)} triplets total ({len(train_idx)} train, {len(early_stop_idx)} early-stop)")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(seed))
    early_stop_loader = DataLoader(Subset(dataset, early_stop_idx), batch_size=batch_size, shuffle=False)

    model = EmbeddingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_early_stop_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_loss(model, train_loader, device, margin, optimizer=optimizer)
        early_stop_loss = _epoch_loss(model, early_stop_loader, device, margin, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "early_stop_loss": early_stop_loss})
        if verbose:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  early_stop_loss={early_stop_loss:.4f}")

        if early_stop_loss < best_early_stop_loss - 1e-4:
            best_early_stop_loss = early_stop_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CKPT_DIR, f"candidate_gen_seed{seed}.pt")
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, checkpoint_path)

    summary = {
        "seed": seed, "checkpoint_path": checkpoint_path, "best_epoch": best_epoch,
        "best_early_stop_loss": best_early_stop_loss, "n_train_triplets": len(train_idx),
        "n_early_stop_triplets": len(early_stop_idx),
        "n_train_families": len(TRAIN_FAMILIES), "epochs_run": len(history),
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "margin": margin, "patience": patience},
        "history": history,
    }
    with open(os.path.join(CKPT_DIR, f"training_history_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260201)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    result = train(seed=args.seed, epochs=args.epochs)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
