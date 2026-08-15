"""Loads a trained checkpoint for use by pipeline/ranking.py::rank_with_model
or for ad-hoc embedding similarity checks in the Streamlit app. This is the
only place a checkpoint path turns into a ready-to-use model object."""
from __future__ import annotations

import torch

from .architecture import EmbeddingNet


def load_model(checkpoint_path: str, device: str = "cpu") -> EmbeddingNet:
    model = EmbeddingNet()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
