"""Parameter sweep for P3, on tuning surfaces only.

Builds each pair's candidate pool once with production code, caches it, then
replays only the stage-2 re-scoring for every configuration. The pool cannot
depend on P3's parameters (it is built before stage 2 runs), so this is an
exact speedup, not an approximation - `--verify-cache` re-runs a sample
through the single-shot path and asserts identical output.

    python -m experiments.discriminability_weighted.sweep --surface development
    python -m experiments.discriminability_weighted.sweep --surface tune_degraded
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation

from .harness import _build_pool, finish_from_pool, localize_weighted

ROOT = os.path.dirname(os.path.abspath(__file__))

SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(ROOT, "data", "tune_degraded"),
    "validate_fresh": os.path.join(ROOT, "data", "validate_fresh"),
}


def build_cache(data_root: str, split: str = "development", verbose: bool = True):
    manifest = load_manifest(data_root, split)
    cache = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        t0 = time.perf_counter()
        psf_sigma, _, candidates = _build_pool(
            ref, search, candidate_generation.DEFAULT_SCALE_HYPOTHESES,
            candidate_generation.DEFAULT_ROTATION_HYPOTHESES, True)
        cache.append({"row": row, "ref": ref, "search": search, "psf_sigma": psf_sigma,
                       "candidates": candidates, "pool_time_s": time.perf_counter() - t0})
        if verbose:
            print(f"  pooled {row['pair_id']:32s} ({len(candidates)} cands)", flush=True)
    return cache


def run_config(cache, **cfg) -> pd.DataFrame:
    rows = []
    for entry in cache:
        row = entry["row"]
        res = finish_from_pool(entry["ref"], entry["search"], entry["psf_sigma"],
                               entry["candidates"], pool_time_s=entry["pool_time_s"], **cfg)
        rows.append({
            "pair_id": row["pair_id"], "structural_family": row["structural_family"],
            "split": row["split"], "error_px": float(np.hypot(res.x - row["gt_x"], res.y - row["gt_y"])),
            "pred_x": res.x, "pred_y": res.y, "confidence": res.confidence,
            "ambiguity_ratio": res.ambiguity_ratio, "runtime_s": res.runtime_s,
            "psf_sigma": res.psf_sigma, "tie_group_size": res.tie_group_size,
            "rescored": res.rescored, "winner_changed": res.winner_changed,
        })
    return pd.DataFrame(rows)


def verify_cache(cache, cfg, n: int = 3) -> dict:
    """The cached-pool path must equal the single-shot path exactly."""
    bad = []
    for entry in cache[:n]:
        direct = localize_weighted(entry["ref"], entry["search"], **cfg)
        cached = finish_from_pool(entry["ref"], entry["search"], entry["psf_sigma"],
                                   entry["candidates"], **cfg)
        if not (direct.x == cached.x and direct.y == cached.y
                and direct.confidence == cached.confidence):
            bad.append(entry["row"]["pair_id"])
    return {"checked": min(n, len(cache)), "mismatches": bad}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="development", choices=sorted(SURFACES))
    p.add_argument("--out", default=None)
    p.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--tie-eps", default="0.0,0.01,0.02,0.05")
    p.add_argument("--schemes", default="lattice_shift,confuser_variance")
    p.add_argument("--smooth", default="1.0")
    args = p.parse_args()

    data_root = SURFACES[args.surface]
    out_dir = args.out or os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.surface}] building candidate-pool cache (production code, once per pair)")
    cache = build_cache(data_root)
    print(f"  cached {len(cache)} pairs\n")

    chk = verify_cache(cache, dict(alpha=0.5, scheme="lattice_shift", tie_eps=0.02))
    print(f"cache-equivalence check: {chk['checked']} pairs, mismatches={chk['mismatches']}\n")
    if chk["mismatches"]:
        raise SystemExit("cached-pool path diverged from single-shot path - aborting")

    alphas = [float(a) for a in args.alphas.split(",")]
    tie_epses = [float(t) for t in args.tie_eps.split(",")]
    schemes = args.schemes.split(",")
    smooths = [float(s) for s in args.smooth.split(",")]

    baseline = run_config(cache, alpha=0.0, scheme="lattice_shift", tie_eps=0.0)
    base_acc = float((baseline.error_px <= 5).mean())
    baseline.to_csv(os.path.join(out_dir, f"{args.surface}_baseline.csv"), index=False)
    print(f"baseline (production) acc@5px = {base_acc:.4f}  n={len(baseline)}\n")

    summary = []
    for scheme, alpha, tie_eps, smooth in itertools.product(schemes, alphas, tie_epses, smooths):
        if alpha == 0.0 or tie_eps == 0.0:
            continue  # null settings, already verified identical to baseline
        df = run_config(cache, alpha=alpha, scheme=scheme, tie_eps=tie_eps, smooth_sigma=smooth)
        merged = baseline[["pair_id", "error_px"]].merge(
            df[["pair_id", "error_px"]], on="pair_id", suffixes=("_base", "_cand"))
        rescued = int(((merged.error_px_base > 5) & (merged.error_px_cand <= 5)).sum())
        broken = int(((merged.error_px_base <= 5) & (merged.error_px_cand > 5)).sum())
        acc = float((df.error_px <= 5).mean())
        rec = {"scheme": scheme, "alpha": alpha, "tie_eps": tie_eps, "smooth_sigma": smooth,
               "acc_5px": acc, "delta_pp": 100.0 * (acc - base_acc),
               "rescued": rescued, "broken": broken, "net": rescued - broken,
               "n_rescored": int(df.rescored.sum()), "n_winner_changed": int(df.winner_changed.sum()),
               "mean_runtime_s": float(df.runtime_s.mean())}
        summary.append(rec)
        df.to_csv(os.path.join(out_dir, f"{args.surface}_{scheme}_a{alpha}_t{tie_eps}_s{smooth}.csv"),
                  index=False)
        print(f"{scheme:19s} a={alpha:<5} t={tie_eps:<6} s={smooth:<4} "
              f"acc={acc:.4f} ({rec['delta_pp']:+5.1f}pp) rescue={rescued} break={broken} "
              f"net={rec['net']:+d} fired={rec['n_rescored']}/{len(df)}", flush=True)

    sdf = pd.DataFrame(summary).sort_values(["net", "acc_5px"], ascending=False)
    sdf.to_csv(os.path.join(out_dir, f"{args.surface}_sweep_summary.csv"), index=False)
    with open(os.path.join(out_dir, f"{args.surface}_sweep_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "baseline_acc_5px": base_acc,
                   "n_pairs": len(baseline), "configs": summary}, f, indent=2)
    print("\n=== best by net rescue ===")
    print(sdf.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
