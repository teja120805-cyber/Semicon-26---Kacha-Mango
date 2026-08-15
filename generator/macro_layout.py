"""Macro-scale layout: tiles independently-rendered DRAM mats with
strip/peripheral regions on both axes.

This is the core structural design choice V2 is built around: a generator
that draws one continuous periodic field per image doesn't match how real
chips look. A real field of view is discrete mats (sub-arrays) separated by strips
(routing / peripheral circuitry), and where a Reference crop lands relative
to that structure - deep inside one mat, straddling two, or centered on a
strip - is a qualitatively different localization problem each time. That
structural variety, not an injected fingerprint, is what should make one
crop distinguishable from another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import mat_generator as mg

STRIP_BASE_VALUE = 85.0
STRIP_ROUTE_PITCH_NM = 220.0
STRIP_ROUTE_WIDTH_NM = 9.0
STRIP_ROUTE_VALUE = 150.0


def _axis_spans(total_nm: int, mat_size_nm: int, strip_width_nm: int) -> list[dict]:
    """Deterministic alternating [mat, strip, mat, strip, ...] tiling along
    one axis. Deterministic (no RNG) so row/col spans are reproducible given
    only the size parameters; per-mat content is still independently random
    (see generate_macro_canvas)."""
    spans = []
    pos = 0
    is_mat = True
    while pos < total_nm:
        size = mat_size_nm if is_mat else strip_width_nm
        size = min(size, total_nm - pos)
        if size <= 0:
            break
        spans.append({"start": pos, "size": size, "kind": "mat" if is_mat else "strip"})
        pos += size
        is_mat = not is_mat
    return spans


def _strip_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Peripheral/routing-region fill: flat base plus sparse orthogonal
    routing lines at a coarser, non-cell pitch - structurally distinct from
    any mat's cell array, with a per-canvas random phase so strip content
    is not identical across regenerations."""
    canvas = np.full((h, w), STRIP_BASE_VALUE, dtype=np.float32)
    phase_y = rng.uniform(0, STRIP_ROUTE_PITCH_NM)
    for y in np.arange(phase_y, h, STRIP_ROUTE_PITCH_NM):
        y0, y1 = int(y), int(min(h, y + STRIP_ROUTE_WIDTH_NM))
        if y1 > y0:
            canvas[y0:y1, :] = STRIP_ROUTE_VALUE
    phase_x = rng.uniform(0, STRIP_ROUTE_PITCH_NM)
    for x in np.arange(phase_x, w, STRIP_ROUTE_PITCH_NM):
        x0, x1 = int(x), int(min(w, x + STRIP_ROUTE_WIDTH_NM))
        if x1 > x0:
            canvas[:, x0:x1] = np.maximum(canvas[:, x0:x1], STRIP_ROUTE_VALUE)
    return canvas


@dataclass
class MacroLayout:
    canvas: np.ndarray
    mat_rects: list = field(default_factory=list)   # [{x,y,w,h,mat_id,preset}]
    strip_rects: list = field(default_factory=list)  # [{x,y,w,h}]


def generate_macro_canvas(seed: int, *, canvas_size_nm: int = 10000,
                           mat_size_nm: int = 2400, strip_width_nm: int = 300,
                           force_preset: Optional[str] = None,
                           feature_size_scale: float = 1.0,
                           linewidth_bias_nm: float = 0.0,
                           collapse_enabled: bool = True,
                           collapse_threshold_nm: float = 10.0,
                           collapse_prob: float = 0.65,
                           corner_rounding_px: float = 0.0) -> MacroLayout:
    """Build one shared fine canvas (1 nm/px) tiling independently-rendered
    mats with strip regions. Every mat gets its own RNG spawned from the
    layout RNG, so mat content is independent even between mats sharing a
    preset - this independence (not an injected marker) is the source of
    mat-to-mat distinguishability.
    """
    rng = np.random.default_rng(seed)
    row_spans = _axis_spans(canvas_size_nm, mat_size_nm, strip_width_nm)
    col_spans = _axis_spans(canvas_size_nm, mat_size_nm, strip_width_nm)

    canvas = _strip_texture(canvas_size_nm, canvas_size_nm, rng)
    mat_rects: list[dict] = []
    strip_rects: list[dict] = []
    mat_id = 0
    for rs in row_spans:
        for cs in col_spans:
            x, y, w, h = cs["start"], rs["start"], cs["size"], rs["size"]
            if rs["kind"] == "mat" and cs["kind"] == "mat":
                preset = mg.pick_mat_preset(rng, force=force_preset)
                mat_seed = int(rng.integers(0, 2 ** 31 - 1))
                mat_rng = np.random.default_rng(mat_seed)
                side = max(h, w)
                mat_img = mg.generate_mat(
                    side, preset, mat_rng,
                    feature_size_scale=feature_size_scale,
                    linewidth_bias_nm=linewidth_bias_nm,
                    collapse_enabled=collapse_enabled,
                    collapse_threshold_nm=collapse_threshold_nm,
                    collapse_prob=collapse_prob,
                    corner_rounding_px=corner_rounding_px,
                )
                canvas[y:y + h, x:x + w] = mat_img[:h, :w]
                mat_rects.append({"x": x, "y": y, "w": w, "h": h, "mat_id": mat_id, "preset": preset})
                mat_id += 1
            else:
                strip_rects.append({"x": x, "y": y, "w": w, "h": h})
    return MacroLayout(canvas=canvas, mat_rects=mat_rects, strip_rects=strip_rects)


def _rect_overlaps(ax: float, ay: float, aw: float, ah: float,
                    bx: float, by: float, bw: float, bh: float) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def crop_membership(x0: int, y0: int, size: int, layout: MacroLayout) -> dict:
    """Structural membership of a candidate crop window - purely geometric,
    never influenced by any localization algorithm's output, so it is safe
    to report alongside ground truth without leaking anything about how
    hard the crop turned out to be for a specific matcher."""
    touched = [m for m in layout.mat_rects if _rect_overlaps(x0, y0, size, size, m["x"], m["y"], m["w"], m["h"])]
    mat_ids = [m["mat_id"] for m in touched]
    presets = sorted({m["preset"] for m in touched})
    crosses_strip = any(
        _rect_overlaps(x0, y0, size, size, s["x"], s["y"], s["w"], s["h"]) for s in layout.strip_rects
    )
    return {
        "mat_ids": mat_ids,
        "num_mats": len(mat_ids),
        "presets": presets,
        "crosses_mat_boundary": len(mat_ids) > 1,
        "crosses_strip_boundary": crosses_strip,
        "same_preset_boundary": len(mat_ids) > 1 and len(presets) == 1,
    }
