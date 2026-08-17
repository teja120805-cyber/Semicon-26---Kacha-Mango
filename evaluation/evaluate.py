"""Runs the localization pipeline over a dataset split (or the external
cross_generator surface) and writes per-pair predictions/errors.

Ground truth is read HERE, in the evaluation harness, purely to score an
already-produced prediction - exactly like a real held-out test set. The
pipeline itself (pipeline/localize.py) never sees it.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from pipeline.localize import localize

V2_SPLITS = ("development", "validation", "held_out", "challenge")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _portable_path(path: str) -> str:
    """Render an image path for the OUTPUT manifest as a forward-slash path
    relative to the project root.

    The manifest (`outputs/reports/per_pair_results.csv`) is a submission
    deliverable, and the submission checklist forbids hard-coded local paths.
    `--data-root` defaults to an absolute path, so without this every row
    shipped a per-machine absolute path such as
    `<drive>:\\...\\data\\development/x.png` - machine-specific, mixed-separator,
    and unreadable on any other checkout.

    Only the RECORDED path is rewritten; images are still read through the
    original absolute path, so this cannot change which files are evaluated.
    Falls back to the input unchanged when the data root lives outside the
    project (e.g. an evaluator pointing `--data-root` at their own directory),
    since a relative path would be wrong in that case."""
    try:
        rel = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    except ValueError:                      # different drive on Windows
        return path.replace("\\", "/")
    if rel.startswith(os.pardir):           # outside the project tree
        return path.replace("\\", "/")
    return rel.replace(os.sep, "/")


def _load_v2_manifest(data_root: str, split: str) -> pd.DataFrame:
    with open(os.path.join(data_root, split, "ground_truth.json")) as f:
        records = json.load(f)
    rows = []
    for r in records:
        dp = r["degradation_params"]
        rows.append({
            "pair_id": r["pair_id"], "split": split, "structural_family": r["structural_family"],
            "reference_path": os.path.join(data_root, r["reference_path"]),
            "search_path": os.path.join(data_root, r["search_path"]),
            "gt_x": r["gt_x"], "gt_y": r["gt_y"],
            "dose_search": dp["dose_search"], "rotation_deg": dp["rotation_deg"], "extra_scale": dp["extra_scale"],
            "crosses_mat_boundary": r["crosses_mat_boundary"], "crosses_strip_boundary": r["crosses_strip_boundary"],
            "periodicity_score": r["periodicity_score"], "uniqueness_score": r["uniqueness_score"],
            "num_mats": r["num_mats"],
        })
    return pd.DataFrame(rows)


def _load_cross_generator_manifest(data_root: str) -> pd.DataFrame:
    split_dir = os.path.join(data_root, "cross_generator")
    with open(os.path.join(split_dir, "ground_truth.json")) as f:
        records = json.load(f)
    rows = []
    for r in records:
        rows.append({
            "pair_id": f"cross_generator_{int(r['id']):05d}", "split": "cross_generator",
            "structural_family": "cross_generator_external",
            "reference_path": os.path.join(data_root, r["reference_path"]),
            "search_path": os.path.join(data_root, r["search_path"]),
            "gt_x": r["gt_x"], "gt_y": r["gt_y"],
            "dose_search": r.get("dose_search", np.nan),
            "rotation_deg": 0.0,   # the external reference generator has no rotation mechanism (verified in Phase 0)
            "extra_scale": 1.0,    # ...and no scale-drift mechanism either (fixed, exact 10x always)
            "crosses_mat_boundary": False, "crosses_strip_boundary": False,
            "periodicity_score": np.nan, "uniqueness_score": np.nan, "num_mats": np.nan,
        })
    return pd.DataFrame(rows)


def load_manifest(data_root: str, split: str) -> pd.DataFrame:
    if split == "cross_generator":
        return _load_cross_generator_manifest(data_root)
    if split in V2_SPLITS:
        return _load_v2_manifest(data_root, split)
    raise ValueError(f"Unknown split '{split}'")


def evaluate_split(data_root: str, split: str, *, ranking_mode: str = "classical", model=None,
                    device: str = "cpu", limit: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    manifest = load_manifest(data_root, split)
    if limit is not None:
        manifest = manifest.iloc[:limit]

    results = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(f"Could not read images for {row['pair_id']}")
        result = localize(ref, search, ranking_mode=ranking_mode, model=model, device=device)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        record = row.to_dict()
        # Record portable paths; the images above were already read through the
        # absolute ones, so this affects the manifest only.
        record["reference_path"] = _portable_path(record["reference_path"])
        record["search_path"] = _portable_path(record["search_path"])
        results.append({
            **record,
            "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
            "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
            "ambiguous": result.ambiguous, "runtime_s": result.runtime_s, "ranking_mode": ranking_mode,
        })
        if verbose:
            print(f"  [{split}] {row['pair_id']:28s} err={error_px:7.2f}px conf={result.confidence:.3f}")
    return pd.DataFrame(results)


def evaluate_all(data_root: str, splits: list[str], out_dir: str, *, ranking_mode: str = "classical",
                  model=None, device: str = "cpu", limit: Optional[int] = None,
                  verbose: bool = True) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    all_results = [
        evaluate_split(data_root, split, ranking_mode=ranking_mode, model=model, device=device,
                        limit=limit, verbose=verbose)
        for split in splits
    ]
    combined = pd.concat(all_results, ignore_index=True)
    suffix = "" if ranking_mode == "classical" else f"_{ranking_mode}"
    combined.to_csv(os.path.join(out_dir, f"per_pair_results{suffix}.csv"), index=False)
    combined.to_json(os.path.join(out_dir, f"per_pair_results{suffix}.json"), orient="records", indent=2)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the V2 localization pipeline over one or more splits.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="development,validation,held_out,challenge,cross_generator")
    parser.add_argument("--out", default="outputs/reports")
    parser.add_argument("--ranking-mode", default="classical", choices=["classical", "learned"])
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    loaded_model = None
    if args.ranking_mode == "learned":
        if not args.model_checkpoint:
            raise SystemExit("--model-checkpoint is required for --ranking-mode learned")
        import torch
        from model.architecture import EmbeddingNet
        loaded_model = EmbeddingNet()
        loaded_model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu"))
        loaded_model.eval()

    evaluate_all(args.data_root, args.splits.split(","), args.out,
                 ranking_mode=args.ranking_mode, model=loaded_model)
