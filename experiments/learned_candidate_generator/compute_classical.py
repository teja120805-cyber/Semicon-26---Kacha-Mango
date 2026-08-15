"""Stage 1 of the classical-vs-learned-vs-hybrid comparison: compute and
save classical candidate pools for every gate-split pair. Deliberately
NEVER imports torch - kept in its own process/script because, on this
Windows environment, many repeated cv2.matchTemplate calls (as
pipeline/candidate_generation.py's 25-hypothesis search performs) segfault
if run in the same process as a loaded PyTorch model (a native
OpenCV/PyTorch threading conflict, not a logic bug - confirmed by isolating
each piece: classical alone works, learned-model alone works, torch model
load followed by the classical bulk search crashes reliably).
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402
from pipeline import candidate_generation  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    all_out = {}
    for split in GATE_SPLITS:
        manifest = load_manifest(os.path.join(PROJECT_ROOT, "data"), split)
        print(f"=== {split} ===", flush=True)
        for _, row in manifest.iterrows():
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            raw = candidate_generation.build_candidate_pool(ref, search)
            deduped = candidate_generation.deduplicate_by_location(raw)
            all_out[row["pair_id"]] = {
                "gt_x": float(row["gt_x"]), "gt_y": float(row["gt_y"]), "split": split,
                "candidates": [
                    {"x": c.x, "y": c.y, "score": c.score, "scale": c.scale,
                     "rotation_deg": c.rotation_deg, "template_size": c.template_size}
                    for c in deduped
                ],
            }
        print(f"  {split} done, {sum(1 for k,v in all_out.items() if v['split']==split)} pairs", flush=True)

    with open(os.path.join(OUT_DIR, "classical_candidates.json"), "w") as f:
        json.dump(all_out, f)
    print(f"Wrote {OUT_DIR}/classical_candidates.json ({len(all_out)} pairs)", flush=True)


if __name__ == "__main__":
    main()
