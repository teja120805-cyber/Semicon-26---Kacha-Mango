"""Stage 3: combine the classical and learned candidate pools
(compute_classical.py / compute_learned.py outputs) into classical-only,
learned-only, and hybrid (union) predictions, apply the same production
subpixel refinement, and compute the full comparison: accuracy metrics,
candidate recall @1/2/5/10px, top-K learned recall, rescue/break vs. the
production classical baseline, and pool sizes. No torch import needed here
(refinement.py is pure cv2/numpy) - kept separate from both prior stages
for the same process-isolation reason.
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import metrics as eval_metrics  # noqa: E402
from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import refinement  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")
GT_TOLERANCE_PX = 5.0
TOP_K_VALUES = (5, 10, 20, 40)


class C:
    __slots__ = ("x", "y", "score", "scale", "rotation_deg", "template_size")

    def __init__(self, d):
        self.x, self.y, self.score = d["x"], d["y"], d.get("ncc_score", d.get("score"))
        self.scale, self.rotation_deg = d["scale"], d["rotation_deg"]
        self.template_size = d["template_size"]


def _dist(x1, y1, x2, y2) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def dedup(cands: list[C], radius_px: float = 10.0) -> list[C]:
    ordered = sorted(cands, key=lambda c: c.score, reverse=True)
    kept: list[C] = []
    r2 = radius_px ** 2
    for c in ordered:
        if all((c.x - k.x) ** 2 + (c.y - k.y) ** 2 > r2 for k in kept):
            kept.append(c)
    return kept


def recall_at(cands: list[C], gt_x, gt_y, tol, top_k=None) -> bool:
    pool = sorted(cands, key=lambda c: c.score, reverse=True)
    if top_k is not None:
        pool = pool[:top_k]
    return any(_dist(c.x, c.y, gt_x, gt_y) <= tol for c in pool)


def finalize(pool: list[C], reference: np.ndarray, search: np.ndarray, gt_x, gt_y) -> float:
    if not pool:
        return float("inf")
    winner = sorted(pool, key=lambda c: c.score, reverse=True)[0]
    rx, ry = refinement.refine(reference, search, winner)
    return _dist(rx, ry, gt_x, gt_y)


def main() -> None:
    with open(os.path.join(OUT_DIR, "classical_candidates.json")) as f:
        classical_data = json.load(f)
    with open(os.path.join(OUT_DIR, "learned_candidates.json")) as f:
        learned_data = json.load(f)

    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
    baseline_df = pd.read_csv(baseline_path)
    baseline_df = baseline_df[baseline_df["split"].isin(GATE_SPLITS)].set_index("pair_id")

    records = []
    manifests = {s: load_manifest(os.path.join(PROJECT_ROOT, "data"), s).set_index("pair_id") for s in GATE_SPLITS}

    for pair_id, cdata in classical_data.items():
        split = cdata["split"]
        ldata = learned_data[pair_id]
        row = manifests[split].loc[pair_id]
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_x, gt_y = cdata["gt_x"], cdata["gt_y"]

        classical = [C(d) for d in cdata["candidates"]]
        learned = [C(d) for d in ldata["candidates"]]
        hybrid = dedup(classical + learned)

        rec = {
            "pair_id": pair_id, "split": split, "gt_x": gt_x, "gt_y": gt_y,
            "error_classical": finalize(classical, ref, search, gt_x, gt_y),
            "error_learned": finalize(learned, ref, search, gt_x, gt_y),
            "error_hybrid": finalize(hybrid, ref, search, gt_x, gt_y),
            "n_classical": len(classical), "n_learned": len(learned), "n_hybrid": len(hybrid),
        }
        for tol in (1, 2, 5, 10):
            rec[f"recall_classical_{tol}px"] = recall_at(classical, gt_x, gt_y, tol)
            rec[f"recall_learned_{tol}px"] = recall_at(learned, gt_x, gt_y, tol)
            rec[f"recall_hybrid_{tol}px"] = recall_at(hybrid, gt_x, gt_y, tol)
        for k in TOP_K_VALUES:
            rec[f"recall_learned_top{k}_5px"] = recall_at(learned, gt_x, gt_y, 5, top_k=k)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "per_pair_comparison.csv"), index=False)

    def summarize(col: str) -> dict:
        tmp = df.rename(columns={col: "error_px"})[["error_px"]].copy()
        tmp["runtime_s"] = 0.0
        return eval_metrics.summarize(tmp)

    merged = baseline_df.join(df.set_index("pair_id"), rsuffix="_new")

    def rescue_break(cand_col: str) -> dict:
        base = merged["error_px"]
        cand = merged[cand_col]
        rescued = int(((base > GT_TOLERANCE_PX) & (cand <= GT_TOLERANCE_PX)).sum())
        broken = int(((base <= GT_TOLERANCE_PX) & (cand > GT_TOLERANCE_PX)).sum())
        cat_rescued = int(((base > 50) & (cand <= 50)).sum())
        cat_new = int(((base <= 50) & (cand > 50)).sum())
        return {"rescue": rescued, "break": broken, "net": rescued - broken,
                "catastrophic_rescued": cat_rescued, "catastrophic_new": cat_new}

    summary = {
        "n": len(df),
        "classical_(this_eval, should match production)": summarize("error_classical"),
        "learned_only": summarize("error_learned"),
        "hybrid": summarize("error_hybrid"),
        "candidate_recall": {
            str(tol): {
                variant: float(df[f"recall_{variant}_{tol}px"].mean())
                for variant in ("classical", "learned", "hybrid")
            } for tol in (1, 2, 5, 10)
        },
        "learned_top_k_recall_at_5px": {str(k): float(df[f"recall_learned_top{k}_5px"].mean()) for k in TOP_K_VALUES},
        "rescue_break_vs_production_classical": {
            "learned_only": rescue_break("error_learned"),
            "hybrid": rescue_break("error_hybrid"),
        },
        "mean_pool_size": {
            "classical": float(df["n_classical"].mean()),
            "learned": float(df["n_learned"].mean()),
            "hybrid": float(df["n_hybrid"].mean()),
        },
        "per_split_accuracy_at_5px": {
            split: {
                "classical": float((df[df["split"] == split]["error_classical"] <= 5).mean()),
                "learned": float((df[df["split"] == split]["error_learned"] <= 5).mean()),
                "hybrid": float((df[df["split"] == split]["error_hybrid"] <= 5).mean()),
            } for split in GATE_SPLITS
        },
    }
    with open(os.path.join(OUT_DIR, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
