"""Is the candidate pool spatially degenerate?

`matching.top_k_peaks` does greedy max-then-suppress with
`SUPPRESSION_RADIUS_PX = 8`. On a DRAM array the correlation surface is
itself periodic, with peaks spaced at the lattice pitch. **If that pitch
exceeds 8px at Search resolution, the "top 2 peaks" of a hypothesis can be
two adjacent lattice peaks in the SAME neighbourhood** rather than two
genuinely different candidate locations — so the pool would be far less
spatially diverse than its size suggests, and recall would be capped for
reasons nothing to do with how many peaks are retained.

This is a real gap in the project's evidence. Both prior experiments that
touched pool width moved the radius **down**, not up:
`experiments/wider_candidate_pool/` used 4px, and
`experiments/discriminability_weighted/pool_recall.py` tested 8 and 4. A
*larger* radius has never been measured.

Efficiency: the correlation map for each (scale, rotation, psf) is computed
ONCE and peaks are extracted from it at every radius and k in the same pass,
so the whole grid costs about one pool build per pair rather than one per
configuration.

Stage 1 (this file) is a diagnostic: it measures the lattice pitch of the
correlation surface, the pool's actual spatial spread, and recall across the
radius grid. It selects nothing. An end-to-end run is only warranted if
recall responds to radius.

    python -m experiments.nms_spatial_diversity.run --surface development
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching
from pipeline.candidate_generation import Candidate
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}

RADII = (8, 16, 32, 64, 128)
KS = (2, 4, 8)
MAX_K = max(KS)
HIT_PX = 5.0


def correlation_pitch(score_map: np.ndarray) -> float:
    """Dominant peak spacing of the correlation surface itself, via its
    autocorrelation. This is the quantity the NMS radius should be compared
    against — if it exceeds the radius, suppression cannot separate lattice
    repeats."""
    m = score_map.astype(np.float32)
    m = m - m.mean()
    if m.std() < 1e-6:
        return float("nan")
    # Subsample for speed; pitch is a coarse property.
    m = m[::2, ::2]
    f = np.fft.rfft2(m)
    acf = np.fft.irfft2(f * np.conj(f), s=m.shape)
    acf = np.fft.fftshift(acf)
    acf /= acf.max()
    h, w = acf.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    ring = (r >= 3) & (r <= min(h, w) * 0.4)
    if not ring.any():
        return float("nan")
    masked = np.where(ring, acf, -np.inf)
    idx = int(np.argmax(masked))
    py, px = divmod(idx, w)
    return float(np.hypot(py - cy, px - cx) * 2)  # undo the ::2 subsample


def analyse_pair(ref, srch, gt):
    """One pass over all hypotheses; peaks extracted at every radius."""
    peaks = {(r, s): [] for r in RADII for s in (0.0, PSF_MATCH_SIGMA)}
    pitches = []
    for sigma in (0.0, PSF_MATCH_SIGMA):
        for scale in candidate_generation.DEFAULT_SCALE_HYPOTHESES:
            for rot in candidate_generation.DEFAULT_ROTATION_HYPOTHESES:
                t = matching.build_template(ref, scale, rot, sigma)
                smap = matching.correlate(srch, t)
                if sigma == 0.0 and scale == 10.0 and rot == 0.0:
                    pitches.append(correlation_pitch(smap))
                for radius in RADII:
                    got = matching.top_k_peaks(smap, MAX_K, radius)
                    peaks[(radius, sigma)].append([
                        Candidate(x=px + t.shape[1] / 2.0, y=py + t.shape[0] / 2.0, score=sc,
                                   scale=scale, rotation_deg=rot, template_size=t.shape[0])
                        for px, py, sc in got])
    out = {"pitch_px": float(np.nanmean(pitches)) if pitches else float("nan")}
    for radius in RADII:
        # Production selects the arm at k=2 by decisiveness; keep that fixed
        # so this isolates the radius and does not also change the PSF rule.
        gaps = {}
        for sigma in (0.0, PSF_MATCH_SIGMA):
            p2 = candidate_generation.deduplicate_by_location(
                [c for hyp in peaks[(radius, sigma)] for c in hyp[:2]])
            gaps[sigma] = _decisiveness(p2)
        sel = max((0.0, PSF_MATCH_SIGMA), key=lambda s: gaps[s])
        for k in KS:
            pool = candidate_generation.deduplicate_by_location(
                [c for hyp in peaks[(radius, sel)] for c in hyp[:k]])
            d = min(np.hypot(c.x - gt[0], c.y - gt[1]) for c in pool)
            out[f"r{radius}_k{k}_hit"] = bool(d <= HIT_PX)
            out[f"r{radius}_k{k}_n"] = len(pool)
            if k == 2:
                # spatial spread: median nearest-neighbour distance in the pool
                pts = np.array([[c.x, c.y] for c in pool])
                if len(pts) > 1:
                    dd = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                                  pts[:, None, 1] - pts[None, :, 1])
                    np.fill_diagonal(dd, np.inf)
                    out[f"r{radius}_nn_px"] = float(np.median(dd.min(axis=1)))
                else:
                    out[f"r{radius}_nn_px"] = float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="development", choices=sorted(SURFACES))
    args = p.parse_args()

    manifest = load_manifest(SURFACES[args.surface], "development")
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        r = analyse_pair(ref, srch, (float(row["gt_x"]), float(row["gt_y"])))
        r.update({"pair_id": row["pair_id"], "structural_family": row["structural_family"]})
        rows.append(r)
        print(f"  {row['pair_id']:30s} pitch={r['pitch_px']:6.1f}px  "
              f"r8_nn={r['r8_nn_px']:6.1f}  r128_nn={r['r128_nn_px']:6.1f}  "
              f"hit r8k2={r['r8_k2_hit']} r128k2={r['r128_k2_hit']}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{args.surface}_nms.csv"), index=False)

    print("\n" + "=" * 64)
    print(f"SPATIAL DIVERSITY OF THE CANDIDATE POOL  ({args.surface}, n={len(df)})")
    print("=" * 64)
    print(f"  correlation-surface pitch : median {df.pitch_px.median():.1f} px "
          f"(range {df.pitch_px.min():.0f}-{df.pitch_px.max():.0f})")
    print(f"  production NMS radius     : {candidate_generation.SUPPRESSION_RADIUS_PX} px")
    print(f"  -> pitch exceeds radius on {int((df.pitch_px > 8).sum())}/{len(df)} pairs\n")

    print(f"  {'radius':>7} {'median NN dist':>15} {'recall k=2':>11} {'k=4':>7} {'k=8':>7} "
          f"{'pool n (k=2)':>13}")
    grid = []
    for radius in RADII:
        nn = df[f"r{radius}_nn_px"].median()
        rec = {k: float(df[f"r{radius}_k{k}_hit"].mean()) for k in KS}
        n2 = float(df[f"r{radius}_k2_n"].mean())
        grid.append({"radius": radius, "median_nn_px": float(nn),
                     **{f"recall_k{k}": rec[k] for k in KS}, "mean_pool_k2": n2})
        print(f"  {radius:>7} {nn:>15.1f} {rec[2]:>11.4f} {rec[4]:>7.4f} {rec[8]:>7.4f} "
              f"{n2:>13.1f}")

    base = float(df["r8_k2_hit"].mean())
    best = max(grid, key=lambda g: g["recall_k2"])
    print(f"\n  production (r=8, k=2) recall : {base:.4f}")
    print(f"  best radius at k=2           : r={best['radius']} -> {best['recall_k2']:.4f} "
          f"({100*(best['recall_k2']-base):+.1f}pp)")
    print("\n  GO only if a larger radius raises recall at EQUAL k — that would mean the")
    print("  pool was spatially degenerate. Equal recall means the radius was never the")
    print("  constraint and this direction is closed.")

    with open(os.path.join(out, f"{args.surface}_nms.json"), "w") as f:
        json.dump({"surface": args.surface, "n": len(df),
                   "median_pitch_px": float(df.pitch_px.median()), "grid": grid}, f, indent=2)


if __name__ == "__main__":
    main()
