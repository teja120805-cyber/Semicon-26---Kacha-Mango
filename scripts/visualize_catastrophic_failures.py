"""Part 9 visual failure analysis: for the top-10 catastrophic failures on
the frozen benchmark (by outputs/reports/baseline_failure_decomposition.csv),
render Reference / Search-with-GT-and-predicted-markers side by side, plus
the diagnostic numbers (candidate rank/score, score margin, rotation/scale
hypothesis, periodicity score) needed to see WHY each one is wrong.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations", "catastrophic_failures")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "reports", "baseline_failure_decomposition.csv"))
    top10 = df.sort_values("error_px", ascending=False).head(10).reset_index(drop=True)

    rows_txt = []
    for rank, row in top10.iterrows():
        ref = cv2.imread(os.path.join(PROJECT_ROOT, row["reference_path"]), cv2.IMREAD_UNCHANGED)
        search = cv2.imread(os.path.join(PROJECT_ROOT, row["search_path"]), cv2.IMREAD_UNCHANGED)
        display = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(display, (int(row["gt_x"]), int(row["gt_y"])), (0, 200, 0),
                        markerType=cv2.MARKER_DIAMOND, markerSize=30, thickness=3)
        cv2.drawMarker(display, (int(row["pred_x"]), int(row["pred_y"])), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)

        ref_bgr = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        ref_padded = cv2.copyMakeBorder(ref_bgr, 0, display.shape[0] - ref_bgr.shape[0], 0,
                                         20, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        combined = np.hstack([ref_padded, display])

        pair_id = row["pair_id"]
        out_path = os.path.join(OUT_DIR, f"{rank+1:02d}_{pair_id}.png")
        cv2.imwrite(out_path, combined)

        line = (
            f"#{rank+1} {pair_id} (split={row['split']}, family={row['structural_family']})\n"
            f"  error_px={row['error_px']:.1f}  failure_location={row['failure_location']}\n"
            f"  gt_in_pool={row['gt_in_pool']}  gt_candidate_rank={row.get('gt_candidate_rank')}  "
            f"gt_candidate_score={row.get('gt_candidate_score')}\n"
            f"  winner_score={row['winner_score']:.4f}  top2_score_margin={row['top2_score_margin']:.4f}  "
            f"winner_scale={row['winner_scale']}  winner_rotation_deg={row['winner_rotation_deg']}\n"
            f"  true rotation_deg={row['rotation_deg']:.2f}  true extra_scale={row['extra_scale']:.3f}  "
            f"periodicity_score={row['periodicity_score']}  "
            f"boundary(mat={row['crosses_mat_boundary']},strip={row['crosses_strip_boundary']})\n"
            f"  -> {out_path}\n"
        )
        rows_txt.append(line)
        print(line)

    with open(os.path.join(OUT_DIR, "SUMMARY.txt"), "w") as f:
        f.write("\n".join(rows_txt))
    print(f"Wrote {len(top10)} annotated images + SUMMARY.txt to {OUT_DIR}")


if __name__ == "__main__":
    main()
