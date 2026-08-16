#!/usr/bin/env python
"""Second-seed validation of psf_matched_template's fixed sigma_extra=1.6.

Why this exists. Four full frozen-benchmark runs were made across
psf_matched_template and psf_matched_adaptive. Each individual choice was
disciplined, but four looks at one 156-pair benchmark can select on noise,
and the spread between the leading configurations is ~2pp. This project has
a precedent for exactly that failure mode: pitch_aware_prominence passed
6/7 gate criteria on the production seed and then failed outright on an
independently-seeded dataset.

So: regenerate the dataset with a DIFFERENT seed, run the unmodified
production pipeline and the sigma=1.6 candidate over it, and check whether
the +2.56pp gain survives.

Seed 618234 is reused deliberately - it is the same second seed
pitch_aware_prominence validated against, so the two results are directly
comparable and the seed cannot have been picked to flatter this candidate.

The generator's per-pair RNG folds (seed, split, family, pair_index)
together, so there is zero cross-seed leakage from the production set.

Note the split composition differs from the production benchmark: FAMILIES
covers development/validation/held_out/challenge = 136 pairs.
`cross_generator`'s 20 pairs are produced by a separate routine and are not
regenerated here, so pooled numbers below are over n=136 and are NOT
directly comparable to the 156-pair production figure - only the
baseline-vs-candidate DELTA on this seed is meaningful.

Never modifies pipeline/, generator/, model/, or the production data/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "psf_matched_template"))

from generator.dataset_generator import FAMILIES, generate_pair  # noqa: E402
from pipeline.localize import localize  # noqa: E402

from harness import localize_psf  # noqa: E402

SEED = 618234
SIGMA = 1.6
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
DATA_DIR = os.path.join(HERE, "validation_data")


def build_dataset() -> pd.DataFrame:
    """Generate the second-seed pairs to disk, returning their manifest."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for fam in FAMILIES:
        for i in range(fam["n"]):
            pair_id = f"{fam['name']}_{i:03d}"
            rp = os.path.join(DATA_DIR, f"{pair_id}_reference.png")
            sp = os.path.join(DATA_DIR, f"{pair_id}_search.png")
            if os.path.exists(rp) and os.path.exists(sp):
                meta_path = os.path.join(DATA_DIR, f"{pair_id}_meta.json")
                rows.append(json.load(open(meta_path)))
                continue
            ref, search, meta = generate_pair(i, SEED, fam)
            cv2.imwrite(rp, ref)
            cv2.imwrite(sp, search)
            rec = {"pair_id": pair_id, "split": fam["split"], "family": fam["name"],
                   "reference_path": rp, "search_path": sp,
                   "gt_x": float(meta["gt_x"]), "gt_y": float(meta["gt_y"])}
            json.dump(rec, open(os.path.join(DATA_DIR, f"{pair_id}_meta.json"), "w"))
            rows.append(rec)
        print(f"  generated {fam['name']} ({fam['n']}) [{time.perf_counter() - t0:.0f}s]", flush=True)
    return pd.DataFrame(rows)


def evaluate(manifest: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    t0 = time.perf_counter()
    for n, (_, row) in enumerate(manifest.iterrows(), 1):
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        res = (localize(ref, search) if mode == "baseline"
               else localize_psf(ref, search, sigma_extra=SIGMA))
        rows.append({**row.to_dict(), "pred_x": res.x, "pred_y": res.y,
                     "error_px": float(np.hypot(res.x - row["gt_x"], res.y - row["gt_y"])),
                     "confidence": res.confidence, "runtime_s": res.runtime_s})
        if n % 40 == 0:
            print(f"    {mode} {n}/{len(manifest)} [{time.perf_counter() - t0:.0f}s]", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== Second-seed validation (seed={SEED}, sigma_extra={SIGMA}) ===")
    print("Generating dataset...")
    manifest = build_dataset()
    print(f"  {len(manifest)} pairs\n")

    print("Evaluating BASELINE (unmodified production pipeline)...")
    base = evaluate(manifest, "baseline")
    base.to_csv(os.path.join(OUT_DIR, "second_seed_baseline.csv"), index=False)

    print("\nEvaluating CANDIDATE (psf_matched_template, sigma=1.6)...")
    cand = evaluate(manifest, "psf")
    cand.to_csv(os.path.join(OUT_DIR, "second_seed_psf.csv"), index=False)

    b_acc = float((base.error_px <= 5).mean())
    c_acc = float((cand.error_px <= 5).mean())
    bi = base.set_index("pair_id").error_px
    ci = cand.set_index("pair_id").error_px
    common = bi.index.intersection(ci.index)
    rescued = int(((bi[common] > 5) & (ci[common] <= 5)).sum())
    broken = int(((bi[common] <= 5) & (ci[common] > 5)).sum())

    print("\n" + "=" * 70)
    print(f"pooled (n={len(base)})   baseline {b_acc:.4f}  ->  candidate {c_acc:.4f}   ({c_acc - b_acc:+.4f})")
    print(f"rescued {rescued}   broken {broken}   net {rescued - broken:+d}")

    print("\n=== per split ===")
    per_split = {}
    for s in sorted(base.split.unique()):
        ba = float((base[base.split == s].error_px <= 5).mean())
        ca = float((cand[cand.split == s].error_px <= 5).mean())
        per_split[s] = {"baseline": ba, "candidate": ca, "delta": ca - ba}
        print(f"  {s:<16} {ba:.3f} -> {ca:.3f}  ({ca - ba:+.3f})")

    print("\n=== per family ===")
    per_fam = {}
    bf = base.assign(ok=base.error_px <= 5).groupby("family").ok.mean()
    cf = cand.assign(ok=cand.error_px <= 5).groupby("family").ok.mean()
    for fam in sorted(bf.index):
        d = cf[fam] - bf[fam]
        per_fam[fam] = {"baseline": float(bf[fam]), "candidate": float(cf[fam]), "delta": float(d)}
        mark = "  <-- REGRESSION" if d < -1e-9 else ("  <-- gain" if d > 1e-9 else "")
        print(f"  {fam:<28} {bf[fam]:.3f} -> {cf[fam]:.3f}  ({d:+.3f}){mark}")

    summary = {
        "seed": SEED, "sigma_extra": SIGMA, "n": len(base),
        "baseline_acc5": b_acc, "candidate_acc5": c_acc, "delta": c_acc - b_acc,
        "rescued": rescued, "broken": broken,
        "production_seed_delta_for_comparison": 0.0256,
        "per_split": per_split, "per_family": per_fam,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "second_seed_summary.json"), "w"), indent=2)
    print(f"\nProduction-seed delta was +0.0256. This seed: {c_acc - b_acc:+.4f}")


if __name__ == "__main__":
    main()
