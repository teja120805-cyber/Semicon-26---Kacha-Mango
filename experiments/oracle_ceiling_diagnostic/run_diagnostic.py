#!/usr/bin/env python
"""Oracle ceiling diagnostic.

For every pair the production baseline gets WRONG (error > 5px), plus a
control set of pairs it gets RIGHT, measure three quantities:

  grid_gt     - best ZNCC at ground truth over the PRODUCTION 11x9 grid
                (what the pipeline can currently see at the true spot)
  oracle_gt   - best ZNCC at ground truth over a dense continuous warp
                sweep (what is achievable there under an ideal warp)
  oracle_win  - best ZNCC at the baseline's (wrong) predicted location
                over that same dense sweep

The decisive comparison is oracle_gt vs oracle_win:

  oracle_gt > oracle_win  -> GEOMETRIC. Under an ideal warp the true
      location outscores the decoy, so the failure is the hypothesis grid
      being too coarse/misaligned, NOT the ZNCC scoring function. A
      continuous/refined warp search at candidate sites would fix it.

  oracle_gt < oracle_win  -> PHOTOMETRIC / genuine self-similarity. Even
      with a perfect warp the decoy still wins under ZNCC. No amount of
      warp search or ZNCC re-ranking can fix these; they need a different
      scoring representation.

Reads ground truth by design - this is a diagnostic, never a candidate
localization method. Never touches pipeline/, generator/, model/, data/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import candidate_generation  # noqa: E402

from oracle import sweep_best, verify_equivalence  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")

TOLERANCE_PX = 4.0        # a hit found within this of GT would still count as correct (<5px)
N_CONTROL = 20            # currently-correct pairs, as a sanity control

# Dense sweep: ~4x finer in scale and ~12x finer in rotation than production,
# and wider on both axes so the answer cannot be an artifact of the span.
COARSE_SCALES = np.round(np.arange(8.60, 11.4001, 0.05), 4)
COARSE_ROTS = np.round(np.arange(-6.0, 6.0001, 0.25), 4)
FINE_SCALE_HALFWIDTH, FINE_SCALE_STEP = 0.05, 0.01
FINE_ROT_HALFWIDTH, FINE_ROT_STEP = 0.25, 0.05


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV)
    df["failed"] = df["error_px"] > 5.0
    fails = df[df["failed"]].copy()
    oks = df[~df["failed"]].copy()
    # Deterministic, family-spread control sample.
    oks["fam"] = oks["pair_id"].str.rsplit("_", n=1).str[0]
    control = (oks.sort_values("pair_id").groupby("fam", group_keys=False)
                  .head(2).sort_values("pair_id").head(N_CONTROL).copy())
    fails["role"] = "fail"
    control["role"] = "control"
    return pd.concat([fails, control], ignore_index=True)


def refine_grid(center_scale: float, center_rot: float):
    scales = np.round(np.arange(center_scale - FINE_SCALE_HALFWIDTH,
                                center_scale + FINE_SCALE_HALFWIDTH + 1e-9, FINE_SCALE_STEP), 4)
    rots = np.round(np.arange(center_rot - FINE_ROT_HALFWIDTH,
                              center_rot + FINE_ROT_HALFWIDTH + 1e-9, FINE_ROT_STEP), 4)
    return scales, rots


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_pairs()
    print(f"Diagnosing {int((pairs.role == 'fail').sum())} failing + "
          f"{int((pairs.role == 'control').sum())} control pairs\n")

    prod_scales = np.array(candidate_generation.DEFAULT_SCALE_HYPOTHESES, dtype=float)
    prod_rots = np.array(candidate_generation.DEFAULT_ROTATION_HYPOTHESES, dtype=float)

    rows = []
    verified = False
    t_start = time.perf_counter()
    for i, row in pairs.reset_index(drop=True).iterrows():
        ref = cv2.imread(os.path.join(DATA_ROOT, row["reference_path"]), cv2.IMREAD_UNCHANGED)
        search = cv2.imread(os.path.join(DATA_ROOT, row["search_path"]), cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(row["pair_id"])
        ref = ref.astype(np.float32)
        search = search.astype(np.float32)
        if not verified:
            verify_equivalence(ref)
            verified = True
            print("cached-template decomposition verified identical to "
                  "pipeline.matching.build_template\n")

        gt = (float(row["gt_x"]), float(row["gt_y"]))
        win = (float(row["pred_x"]), float(row["pred_y"]))
        targets = {"gt": gt, "win": win}

        # 1. What the PRODUCTION grid can see at ground truth.
        grid = sweep_best(ref, search, {"gt": gt}, prod_scales, prod_rots, TOLERANCE_PX)

        # 2. Dense coarse sweep at both locations, then a local fine refine.
        coarse = sweep_best(ref, search, targets, COARSE_SCALES, COARSE_ROTS, TOLERANCE_PX)
        fine = {}
        for name in targets:
            if not np.isfinite(coarse[name]["score"]):
                fine[name] = coarse[name]
                continue
            fs, fr = refine_grid(coarse[name]["scale"], coarse[name]["rotation_deg"])
            got = sweep_best(ref, search, {name: targets[name]}, fs, fr, TOLERANCE_PX)[name]
            fine[name] = got if got["score"] > coarse[name]["score"] else coarse[name]

        rows.append({
            "pair_id": row["pair_id"], "split": row["split"],
            "family": row["pair_id"].rsplit("_", 1)[0], "role": row["role"],
            "error_px": float(row["error_px"]),
            "true_rotation_deg": float(row.get("rotation_deg", np.nan)),
            "true_extra_scale": float(row.get("extra_scale", np.nan)),
            "periodicity_score": float(row.get("periodicity_score", np.nan)),
            "winner_score": float(row["confidence"]),
            "grid_gt": grid["gt"]["score"],
            "grid_gt_scale": grid["gt"]["scale"], "grid_gt_rot": grid["gt"]["rotation_deg"],
            "oracle_gt": fine["gt"]["score"],
            "oracle_gt_scale": fine["gt"]["scale"], "oracle_gt_rot": fine["gt"]["rotation_deg"],
            "oracle_win": fine["win"]["score"],
            "oracle_win_scale": fine["win"]["scale"], "oracle_win_rot": fine["win"]["rotation_deg"],
        })
        r = rows[-1]
        verdict = ("GEOMETRIC" if r["oracle_gt"] > r["oracle_win"] else "PHOTOMETRIC") \
            if row["role"] == "fail" else "control"
        print(f"[{i + 1:>2}/{len(pairs)}] {row['pair_id']:<32} {row['role']:<7} "
              f"err={row['error_px']:>8.2f}px  grid_gt={r['grid_gt']:.3f}  "
              f"oracle_gt={r['oracle_gt']:.3f}  oracle_win={r['oracle_win']:.3f}  {verdict}")

    out = pd.DataFrame(rows)
    out["gt_headroom"] = out["oracle_gt"] - out["grid_gt"]
    out["oracle_margin"] = out["oracle_gt"] - out["oracle_win"]
    out["fixable_by_warp"] = out["oracle_margin"] > 0
    out.to_csv(os.path.join(OUT_DIR, "oracle_diagnostic.csv"), index=False)

    fails = out[out.role == "fail"]
    ctrl = out[out.role == "control"]
    summary = {
        "n_fail": int(len(fails)), "n_control": int(len(ctrl)),
        "tolerance_px": TOLERANCE_PX,
        "sweep": {"scales": [float(COARSE_SCALES[0]), float(COARSE_SCALES[-1]), FINE_SCALE_STEP],
                  "rotations": [float(COARSE_ROTS[0]), float(COARSE_ROTS[-1]), FINE_ROT_STEP]},
        "fail": {
            "geometric": int(fails.fixable_by_warp.sum()),
            "photometric": int((~fails.fixable_by_warp).sum()),
            "median_grid_gt": float(fails.grid_gt.median()),
            "median_oracle_gt": float(fails.oracle_gt.median()),
            "median_oracle_win": float(fails.oracle_win.median()),
            "median_gt_headroom": float(fails.gt_headroom.median()),
        },
        "control": {
            "geometric_like": int(ctrl.fixable_by_warp.sum()),
            "median_grid_gt": float(ctrl.grid_gt.median()),
            "median_oracle_gt": float(ctrl.oracle_gt.median()),
            "median_gt_headroom": float(ctrl.gt_headroom.median()),
        },
        "by_family": {},
    }
    for fam, g in fails.groupby("family"):
        summary["by_family"][fam] = {
            "n_fail": int(len(g)), "geometric": int(g.fixable_by_warp.sum()),
            "photometric": int((~g.fixable_by_warp).sum()),
            "median_oracle_margin": float(g.oracle_margin.median()),
            "median_gt_headroom": float(g.gt_headroom.median()),
        }
    with open(os.path.join(OUT_DIR, "oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(json.dumps(summary, indent=2))
    print(f"\nElapsed: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
