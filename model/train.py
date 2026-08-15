"""Trains the embedding model on the development split only, with early
stopping based on an internal held-out slice OF the development split -
never the real validation/held_out/challenge splits (see
model/TRAINING_PROTOCOL.md, fixed before this script was run: checkpoint
selection must not be able to see, even indirectly, the splits the
integration gate later scores).
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .architecture import EmbeddingNet
from .dataset import TripletPatchDataset

# dev_dense_periodic is the hardest/most ambiguous development family
# (reports/V2_BASELINE_REPORT.md: 25% accuracy@5px, the worst of the three) - held
# out from training itself and used only to decide when to stop, so
# checkpoint selection is answering "does this generalize to the hardest
# in-scope case" rather than "did loss go down on what it was trained on".
EARLY_STOP_FAMILIES = ("dev_dense_periodic",)


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


def train(data_root: str = "data", out_dir: str = "experiments/embedding_reranker_v1/checkpoints", seed: int = 20260101,
          epochs: int = 40, batch_size: int = 8, lr: float = 1e-3, margin: float = 0.3,
          patience: int = 8, device: str = "cpu", verbose: bool = True) -> dict:
    set_seed(seed)
    dataset = TripletPatchDataset(data_root, seed=seed)
    train_idx, early_stop_idx = split_by_family(dataset)
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
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  dev_early_stop_loss={early_stop_loss:.4f}")

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
    parser = argparse.ArgumentParser(description="Train the V2 embedding re-ranking model.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="experiments/embedding_reranker_v1/checkpoints")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.data_root, args.out, seed=args.seed, epochs=args.epochs, device=args.device)
