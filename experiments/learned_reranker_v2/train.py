#!/usr/bin/env python
"""Trains the embedding re-ranker on the EXPANDED development set
(experiments/learned_reranker_v2/data/development/, generate_data.py) -
otherwise identical to model/train.py's logic (same architecture, same
optimizer/loss/early-stopping strategy, same dev-only-never-validation/
held_out/challenge discipline per model/TRAINING_PROTOCOL.md), reusing
model.architecture.EmbeddingNet and model.dataset.TripletPatchDataset
UNMODIFIED.

The only reason this isn't just a --data-root flag to model/train.py is
that EARLY_STOP_FAMILIES is a module-level constant there, hardcoded to the
production development split's family name ("dev_dense_periodic") - this
experiment's expanded set uses a different name for the analogous hardest
family ("ldev_dense_periodic"), so a thin copy of the training loop with
that one constant corrected is required. Every other line mirrors
model/train.py exactly.
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
sys.path.insert(0, PROJECT_ROOT)

from model.architecture import EmbeddingNet  # noqa: E402
from model.dataset import TripletPatchDataset  # noqa: E402

# The analogous hardest/most-ambiguous family in THIS expanded set (see
# generate_data.py) - held out from training, used only for early stopping,
# exactly mirroring model/train.py's EARLY_STOP_FAMILIES rationale.
EARLY_STOP_FAMILIES = ("ldev_dense_periodic",)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_family(dataset: TripletPatchDataset) -> tuple[list[int], list[int]]:
    train_idx, early_stop_idx = [], []
    for i, item in enumerate(dataset.items):
        if item["structural_family"] in EARLY_STOP_FAMILIES:
            early_stop_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, early_stop_idx


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


def train(data_root: str = DATA_ROOT, out_dir: str = CHECKPOINT_DIR, seed: int = 20260101,
          epochs: int = 40, batch_size: int = 8, lr: float = 1e-3, margin: float = 0.3,
          patience: int = 8, device: str = "cpu", verbose: bool = True) -> dict:
    set_seed(seed)
    dataset = TripletPatchDataset(data_root, seed=seed)
    train_idx, early_stop_idx = split_by_family(dataset)
    if verbose:
        print(f"  dataset: {len(dataset)} triplets total, {len(train_idx)} train, "
              f"{len(early_stop_idx)} early-stop")
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
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_early_stop_loss": early_stop_loss})
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  dev_early_stop_loss={early_stop_loss:.4f}")

        if early_stop_loss < best_early_stop_loss - 1e-4:
            best_early_stop_loss = early_stop_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"embedding_net_seed{seed}.pt")
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, checkpoint_path)

    summary = {
        "seed": seed, "checkpoint_path": checkpoint_path, "best_epoch": best_epoch,
        "best_dev_early_stop_loss": best_early_stop_loss, "n_train_triplets": len(train_idx),
        "n_early_stop_triplets": len(early_stop_idx), "epochs_run": len(history),
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                             "margin": margin, "patience": patience},
        "history": history,
    }
    with open(os.path.join(out_dir, f"training_history_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    for seed in (20260101, 20260102, 20260103):
        print(f"\n=== Training seed {seed} ===")
        result = train(seed=seed)
        print(f"  best_epoch={result['best_epoch']} best_dev_early_stop_loss={result['best_dev_early_stop_loss']:.4f} "
              f"n_train_triplets={result['n_train_triplets']}")
