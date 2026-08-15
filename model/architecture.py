"""Small CNN embedding network for candidate re-ranking (Siamese-style:
shared weights, called once per 100x100 patch, compared via cosine
similarity in pipeline/ranking.py::rank_with_model).

Chosen over a larger or attention-based matcher because
reports/V2_BASELINE_REPORT.md identifies the open question as whether a learned
embedding can rank the true location above near-identical structural
decoys - a hard-negative embedding problem the classical matcher's own
candidate pool already frames well. That is not a problem a bigger model
solves by having more capacity; on a development split of only 24 pairs,
more capacity mostly buys faster overfitting. See
reports/V2_ARCHITECTURE_PLAN.md section 7 for the full reasoning, including why a
cross-attention matcher was considered and not chosen as the first
experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = EmbeddingNet()
    print(f"EmbeddingNet parameter count: {count_parameters(net):,}")
    dummy = torch.randn(2, 1, 100, 100)
    out = net(dummy)
    print(f"Output shape: {tuple(out.shape)}, L2 norms: {out.norm(dim=1).tolist()}")
