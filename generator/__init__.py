"""DriftSense V2 dataset generator.

A macro-structured DRAM localization dataset generator: independently
rendered mat sub-arrays tiled with strip/peripheral regions on a shared
fine canvas, from which a Reference crop and a downsampled, degraded
Search image are both derived - so ground truth is exact by construction
and never touches the localization pipeline.

See ../reports/V2_ARCHITECTURE_PLAN.md for the full design rationale.
"""

from .dataset_generator import (
    GENERATOR_VERSION,
    SCALE_FACTOR,
    REFERENCE_SIZE_PX,
    FINE_CANVAS_SIZE_PX,
    FAMILIES,
    generate_pair,
    generate_dataset,
)

__all__ = [
    "GENERATOR_VERSION",
    "SCALE_FACTOR",
    "REFERENCE_SIZE_PX",
    "FINE_CANVAS_SIZE_PX",
    "FAMILIES",
    "generate_pair",
    "generate_dataset",
]
