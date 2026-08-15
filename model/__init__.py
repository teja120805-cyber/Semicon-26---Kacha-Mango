"""V2 candidate re-ranking model: a small CNN embedding network, trained on
the development split only, evaluated through the mandatory integration
gate (evaluation/benchmark.py) before it may ever replace the classical
ranking default in pipeline/ranking.py. See model/TRAINING_PROTOCOL.md.
"""

from .architecture import EmbeddingNet

__all__ = ["EmbeddingNet"]
