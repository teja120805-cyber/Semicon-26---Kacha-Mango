#!/usr/bin/env python
"""Run the frozen DriftSense V2 localization pipeline on Reference/Search
image pairs and print the predicted center coordinates. No source-code
changes are needed to process a new pair or batch - just point the CLI
flags at the image files.

Single pair:
    python scripts/localize_pair.py --reference ref.png --search search.png

Batch (a CSV with reference_path,search_path columns, each resolved
relative to the CSV's own directory unless absolute):
    python scripts/localize_pair.py --batch-csv pairs.csv --out predictions.csv

Output columns/fields: pred_x, pred_y (Search-image pixels, origin
top-left, per README section 8), confidence (winning classical match
score), ambiguous, ambiguity_ratio, runtime_s. Ground truth is never read
or required - this script is for genuinely unknown pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.localize import localize  # noqa: E402


def _load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def localize_one(reference_path: str, search_path: str) -> dict:
    reference = _load_image(reference_path)
    search = _load_image(search_path)
    result = localize(reference, search)
    return {
        "reference_path": reference_path,
        "search_path": search_path,
        "pred_x": result.x,
        "pred_y": result.y,
        "confidence": result.confidence,
        "ambiguous": result.ambiguous,
        "ambiguity_ratio": result.ambiguity_ratio,
        "runtime_s": result.runtime_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", help="Path to a single Reference image (100x close-up).")
    parser.add_argument("--search", help="Path to a single Search image (10x field of view).")
    parser.add_argument("--batch-csv", help="CSV with reference_path,search_path columns for batch processing.")
    parser.add_argument("--out", help="Write predictions to this CSV (batch mode) or this JSON (single-pair mode).")
    args = parser.parse_args()

    if args.batch_csv:
        manifest = pd.read_csv(args.batch_csv)
        base_dir = os.path.dirname(os.path.abspath(args.batch_csv))
        rows = []
        for _, row in manifest.iterrows():
            ref_path = row["reference_path"]
            search_path = row["search_path"]
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(base_dir, ref_path)
            if not os.path.isabs(search_path):
                search_path = os.path.join(base_dir, search_path)
            result = localize_one(ref_path, search_path)
            rows.append(result)
            print(f"{row['reference_path']:40s} -> (x={result['pred_x']:.2f}, y={result['pred_y']:.2f}) "
                  f"conf={result['confidence']:.3f} runtime={result['runtime_s']:.3f}s")
        out_df = pd.DataFrame(rows)
        if args.out:
            out_df.to_csv(args.out, index=False)
            print(f"Wrote {len(out_df)} predictions -> {args.out}")
    elif args.reference and args.search:
        result = localize_one(args.reference, args.search)
        print(json.dumps(result, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
    else:
        parser.error("Provide either --reference and --search, or --batch-csv.")


if __name__ == "__main__":
    main()
