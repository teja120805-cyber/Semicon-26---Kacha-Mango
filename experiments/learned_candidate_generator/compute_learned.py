"""Stage 2: compute learned candidate proposals for every gate-split pair,
using the trained embedding net.

IMPORTANT ordering: all pandas manifest loading happens BEFORE torch is
imported. Root-caused via `faulthandler` (not a guess): pandas'
`DataFrame.__init__` (specifically its pyarrow-backed string-array
construction, `pandas.core.arrays.string_arrow.py::_from_sequence`)
segfaults on this Windows environment if a new DataFrame is constructed
*after* PyTorch has been imported/used in the same process - a native
allocator/threading conflict between PyTorch's ATen backend and PyArrow,
not a bug in this project's own code. Loading every manifest into plain
Python dicts first, then importing torch, avoids ever constructing a
DataFrame after torch is active.
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluate import load_manifest  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
CKPT_PATH = os.path.join(EXP_DIR, "checkpoints", "candidate_gen_seed20260201.pt")
GATE_SPLITS = ("validation", "held_out", "challenge", "cross_generator")


def _load_all_rows() -> list[dict]:
    """Load every gate-split manifest into plain dicts - the last pandas
    operation this process performs, before torch is ever imported."""
    rows = []
    for split in GATE_SPLITS:
        manifest = load_manifest(os.path.join(PROJECT_ROOT, "data"), split)
        for _, row in manifest.iterrows():
            rows.append({
                "pair_id": row["pair_id"], "split": split,
                "reference_path": row["reference_path"], "search_path": row["search_path"],
                "gt_x": float(row["gt_x"]), "gt_y": float(row["gt_y"]),
            })
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = _load_all_rows()
    print(f"Loaded {len(rows)} rows across {len(GATE_SPLITS)} splits (pre-torch).", flush=True)

    import cv2
    import numpy as np
    import torch
    from model.architecture import EmbeddingNet
    from experiments.learned_candidate_generator.propose_candidates import propose_learned_candidates

    model = EmbeddingNet()
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    all_out = {}
    current_split = None
    for row in rows:
        if row["split"] != current_split:
            current_split = row["split"]
            print(f"=== {current_split} ===", flush=True)
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        cands = propose_learned_candidates(model, ref.astype(np.float32), search.astype(np.float32), top_k=20)
        all_out[row["pair_id"]] = {"gt_x": row["gt_x"], "gt_y": row["gt_y"], "split": row["split"], "candidates": cands}

    with open(os.path.join(OUT_DIR, "learned_candidates.json"), "w") as f:
        json.dump(all_out, f, default=str)
    print(f"Wrote {OUT_DIR}/learned_candidates.json ({len(all_out)} pairs)", flush=True)


if __name__ == "__main__":
    main()
