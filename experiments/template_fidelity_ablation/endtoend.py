"""End-to-end test of the high-pass / band-pass preprocessing.

`run.py` measured truth-vs-decoy margins on failures only, and found that
high-pass filtering both images makes 2 of 12 failures *winnable* against
the control's 0 of 12 - the best of seven interventions tried. That is a
necessary condition, not a sufficient one: the margin test compares truth
only against the CURRENTLY winning decoy, and says nothing about (a) other
decoys that might also outscore truth, or (b) damage to the 28 pairs that
are currently correct.

Only an end-to-end run answers both. This is that run.

The change is preprocessing: high-pass both images before the pipeline sees
them, then run production unmodified. The Reference sigma is 10x the Search
sigma because the Reference is 10x finer, so the same physical spatial
frequency needs a 10x larger pixel sigma - filtering both at one sigma would
attack different physical scales in each image.

Null control: `--hp-search 0` skips filtering entirely and must reproduce
production bit-for-bit.

    python -m experiments.template_fidelity_ablation.endtoend --surface tune_degraded
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, feature_extraction, ranking, refinement
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness, localize

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}


def highpass(img: np.ndarray, sigma: float, lowpass: float = 0.0) -> np.ndarray:
    if sigma <= 0.0:
        return img
    out = img - cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    if lowpass > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), lowpass, borderType=cv2.BORDER_REPLICATE)
    return out


def localize_hp(ref, srch, hp_search, lp_search=0.0):
    """Production pipeline, unmodified, on filtered inputs."""
    t0 = time.perf_counter()
    r = highpass(ref, hp_search * 10.0, lp_search * 10.0 if lp_search else 0.0)
    s = highpass(srch, hp_search, lp_search)
    best = None
    for sigma in (0.0, PSF_MATCH_SIGMA):
        pool = candidate_generation.deduplicate_by_location(
            candidate_generation.build_candidate_pool(r, s, psf_sigma=sigma))
        g = _decisiveness(pool)
        if best is None or g > best[0]:
            best = (g, sigma, pool)
    _, psf, pool = best
    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(pool), s.shape)
    x, y = refinement.refine(r, s, ranked[0], psf)
    scores = sorted((c.score for c in pool), reverse=True)
    return x, y, float(ranked[0].score), feature_extraction.ambiguity_ratio(scores), \
        psf, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    p.add_argument("--configs", default="0:0,16:0,16:0.5,8:0,32:0",
                   help="comma-separated hp_search:lp_search pairs")
    args = p.parse_args()

    manifest = load_manifest(SURFACES[args.surface], "development")
    imgs = [(r, cv2.imread(r["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32),
             cv2.imread(r["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32))
            for _, r in manifest.iterrows()]

    # Null control first: hp=0 must equal production exactly.
    print("null control: hp_search=0 must reproduce production bit-for-bit")
    bad = []
    for row, ref, srch in imgs[:6]:
        base = localize(ref.astype(np.uint8), srch.astype(np.uint8))
        x, y, conf, _amb, _psf, _t = localize_hp(ref, srch, 0.0)
        if not (x == base.x and y == base.y and conf == base.confidence):
            bad.append(row["pair_id"])
    print(f"  checked 6, mismatches={len(bad)} {bad}")
    if bad:
        raise SystemExit("null control failed")
    print("  PASS\n")

    def run(hp, lp):
        rows = []
        for row, ref, srch in imgs:
            x, y, conf, amb, psf, rt = localize_hp(ref, srch, hp, lp)
            rows.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                          "error_px": float(np.hypot(x - row["gt_x"], y - row["gt_y"])),
                          "confidence": conf, "psf_sigma": psf, "runtime_s": rt})
        return pd.DataFrame(rows)

    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)

    base_df = run(0.0, 0.0)
    base_acc = float((base_df.error_px <= 5).mean())
    base_df.to_csv(os.path.join(out, f"{args.surface}_e2e_baseline.csv"), index=False)
    print(f"baseline (no filtering) acc@5px = {base_acc:.4f}  n={len(base_df)}\n")

    results = []
    for spec in args.configs.split(","):
        hp, lp = (float(v) for v in spec.split(":"))
        if hp == 0.0:
            continue
        df = run(hp, lp)
        m = base_df[["pair_id", "error_px"]].merge(df[["pair_id", "error_px"]], on="pair_id",
                                                    suffixes=("_b", "_c"))
        rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
        broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())
        acc = float((df.error_px <= 5).mean())
        rec = {"hp_search": hp, "lp_search": lp, "acc_5px": acc,
               "delta_pp": 100 * (acc - base_acc), "rescued": rescued, "broken": broken,
               "net": rescued - broken, "mean_runtime_s": float(df.runtime_s.mean())}
        results.append(rec)
        df.to_csv(os.path.join(out, f"{args.surface}_e2e_hp{hp}_lp{lp}.csv"), index=False)
        print(f"hp={hp:<5} lp={lp:<4} acc={acc:.4f} ({rec['delta_pp']:+5.1f}pp) "
              f"R={rescued} B={broken} net={rec['net']:+d} rt={rec['mean_runtime_s']:.2f}s",
              flush=True)

    sdf = pd.DataFrame(results).sort_values(["net", "acc_5px"], ascending=False)
    sdf.to_csv(os.path.join(out, f"{args.surface}_e2e_summary.csv"), index=False)
    with open(os.path.join(out, f"{args.surface}_e2e_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "baseline_acc": base_acc, "n": len(base_df),
                   "configs": results}, f, indent=2)
    print("\n=== ranked ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
