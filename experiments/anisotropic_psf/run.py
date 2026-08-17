"""Anisotropic template PSF: match the horizontal-only raster jitter.

Mechanism, read off the generator rather than hypothesised:

`generator/degradation_models.py::image_search` composes the Search image as
blur -> **10x area-average downsample** -> optional rotation/scale ->
`apply_raster_shear_drift` -> ... . The raster drift therefore runs at
SEARCH resolution, not fine-canvas resolution, and it shifts each row
horizontally by

    shift(row) = shear_amplitude_px * (row/(h-1) - 0.5) + N(0, jitter_std_px)

with `DEFAULT_PARAMS` giving `shear_amplitude_px = 1.0` and
`jitter_std_px = 0.4` on **every pair**, not just a special family.

Two consequences:

  * The deterministic shear is negligible at template scale - 1.0px across
    1000 rows is 0.1px across a 100px template - so a shear hypothesis grid
    would be chasing nothing. That is worth stating, because "add shear
    hypotheses" is the obvious move and the arithmetic says not to.
  * The per-row jitter is NOT negligible and is **horizontal only**. Each
    template row is displaced ~0.4px relative to its neighbours, randomly.
    No global affine hypothesis can model it, but its effect on the
    correlation is close to an extra horizontal blur.

`pipeline/matching.build_template` blurs the template isotropically
(`cv2.GaussianBlur(t, (0,0), psf_sigma)` sets sigmaY = sigmaX). So the
effective Search PSF is wider horizontally than vertically and the template
cannot match both axes at once. `reports/PROJECT_STATUS.md` names exactly
this - "the raster shear/jitter the template never models" - as a remaining
source of the template-fidelity gap.

The change: blur the template with separate (sigma_x, sigma_y), sigma_x >
sigma_y. Cost is identical to production - the same one blur, different
kernel - so unlike a wider hypothesis grid this is free at runtime.

Null control: sigma_x = sigma_y = PSF_MATCH_SIGMA reproduces production
bit-for-bit (verified per pair, not assumed).

    python -m experiments.anisotropic_psf.run --surface tune_degraded
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
from pipeline import candidate_generation, feature_extraction, matching, ranking, refinement
from pipeline.candidate_generation import Candidate
from pipeline.localize import (AMBIGUITY_THRESHOLD, PSF_MATCH_SIGMA, _decisiveness, localize)

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}


def build_template_aniso(reference, scale_factor, rotation_deg, sigma_x, sigma_y):
    """pipeline.matching.build_template with an anisotropic blur.

    Everything up to the blur is production code verbatim - resize with
    INTER_AREA, then rotate with BORDER_REPLICATE - so the only difference
    is the kernel. The blur is applied AFTER the rotation, i.e. in the
    template's final frame, which is the frame aligned to the Search image.
    That is the correct place: the jitter being corrected is horizontal in
    SEARCH coordinates.
    """
    h, w = reference.shape
    size = max(8, int(round(w / scale_factor)))
    t = cv2.resize(reference.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    if rotation_deg != 0.0:
        m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), rotation_deg, 1.0)
        t = cv2.warpAffine(t, m, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    if sigma_x > 0.0 or sigma_y > 0.0:
        t = cv2.GaussianBlur(t, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y,
                              borderType=cv2.BORDER_REPLICATE)
    return t.astype(np.float32)


def pool_for(reference, search, sigma_x, sigma_y):
    cands = []
    for scale in candidate_generation.DEFAULT_SCALE_HYPOTHESES:
        for rot in candidate_generation.DEFAULT_ROTATION_HYPOTHESES:
            t = build_template_aniso(reference, scale, rot, sigma_x, sigma_y)
            smap = matching.correlate(search, t)
            for px, py, s in matching.top_k_peaks(smap, candidate_generation.PEAKS_PER_HYPOTHESIS,
                                                   candidate_generation.SUPPRESSION_RADIUS_PX):
                cands.append(Candidate(x=px + t.shape[1] / 2.0, y=py + t.shape[0] / 2.0, score=s,
                                        scale=scale, rotation_deg=rot, template_size=t.shape[0]))
    return candidate_generation.deduplicate_by_location(cands)


def localize_aniso(reference, search, sigma_x, sigma_y):
    """Production's dual-arm structure, with the blurred arm made
    anisotropic. The sharp arm (0,0) is untouched and always evaluated, so
    families that the blur harms still revert to baseline exactly."""
    t0 = time.perf_counter()
    best = None
    for sx, sy in ((0.0, 0.0), (sigma_x, sigma_y)):
        pool = pool_for(reference, search, sx, sy)
        gap = _decisiveness(pool)
        if best is None or gap > best[0]:
            best = (gap, (sx, sy), pool)
    _, (sx, sy), pool = best
    ranked = ranking.apply_center_tiebreak(ranking.rank_classical(pool), search.shape)
    winner = ranked[0]
    t = build_template_aniso(reference, winner.scale, winner.rotation_deg, sx, sy)
    smap = matching.correlate(search, t)
    px = int(np.clip(round(winner.x - t.shape[1] / 2.0), 1, smap.shape[1] - 2))
    py = int(np.clip(round(winner.y - t.shape[0] / 2.0), 1, smap.shape[0] - 2))
    dx = refinement._parabolic_offset(smap[py, px - 1], smap[py, px], smap[py, px + 1])
    dy = refinement._parabolic_offset(smap[py - 1, px], smap[py, px], smap[py + 1, px])
    return (float(px + dx + t.shape[1] / 2.0), float(py + dy + t.shape[0] / 2.0),
            float(winner.score), sx, sy, len(pool), time.perf_counter() - t0)


def verify_null(data_root, n=6) -> dict:
    """sigma_x = sigma_y = PSF_MATCH_SIGMA must equal production exactly."""
    manifest = load_manifest(data_root, "development").iloc[:n]
    bad = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        base = localize(ref, search)
        x, y, conf, _sx, _sy, _n, _t = localize_aniso(
            ref.astype(np.float32), search.astype(np.float32), PSF_MATCH_SIGMA, PSF_MATCH_SIGMA)
        if not (x == base.x and y == base.y and conf == base.confidence):
            bad.append({"pair_id": row["pair_id"], "dx": x - base.x, "dy": y - base.y})
    return {"checked": len(manifest), "mismatches": bad}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    p.add_argument("--sigma-x", default="1.6,1.8,2.0,2.2")
    p.add_argument("--sigma-y", default="1.2,1.4,1.6")
    args = p.parse_args()
    data_root = SURFACES[args.surface]

    print("null control: sigma_x = sigma_y = %.1f must reproduce production" % PSF_MATCH_SIGMA)
    nc = verify_null(data_root)
    print(f"  checked {nc['checked']} pairs, mismatches={len(nc['mismatches'])}")
    if nc["mismatches"]:
        for m in nc["mismatches"]:
            print("   ", m)
        raise SystemExit("null control failed - aborting before any tuning")
    print("  PASS\n")

    manifest = load_manifest(data_root, "development")
    imgs = [(r, cv2.imread(r["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32),
             cv2.imread(r["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32))
            for _, r in manifest.iterrows()]

    def run(sx, sy):
        rows = []
        for row, ref, search in imgs:
            x, y, conf, usx, usy, npool, rt = localize_aniso(ref, search, sx, sy)
            rows.append({"pair_id": row["pair_id"], "structural_family": row["structural_family"],
                          "error_px": float(np.hypot(x - row["gt_x"], y - row["gt_y"])),
                          "used_sigma_x": usx, "used_sigma_y": usy, "runtime_s": rt})
        return pd.DataFrame(rows)

    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)

    base = run(PSF_MATCH_SIGMA, PSF_MATCH_SIGMA)
    base_acc = float((base.error_px <= 5).mean())
    base.to_csv(os.path.join(out, f"{args.surface}_baseline.csv"), index=False)
    print(f"baseline (isotropic {PSF_MATCH_SIGMA}) acc@5px = {base_acc:.4f}  n={len(base)}\n")

    results = []
    for sx, sy in itertools.product([float(v) for v in args.sigma_x.split(",")],
                                     [float(v) for v in args.sigma_y.split(",")]):
        if sx == PSF_MATCH_SIGMA and sy == PSF_MATCH_SIGMA:
            continue
        df = run(sx, sy)
        m = base[["pair_id", "error_px"]].merge(df[["pair_id", "error_px"]], on="pair_id",
                                                 suffixes=("_b", "_c"))
        rescued = int(((m.error_px_b > 5) & (m.error_px_c <= 5)).sum())
        broken = int(((m.error_px_b <= 5) & (m.error_px_c > 5)).sum())
        acc = float((df.error_px <= 5).mean())
        rec = {"sigma_x": sx, "sigma_y": sy, "ratio": round(sx / sy, 3), "acc_5px": acc,
               "delta_pp": 100 * (acc - base_acc), "rescued": rescued, "broken": broken,
               "net": rescued - broken,
               "aniso_arm_used": int((df.used_sigma_x > 0).sum()),
               "mean_runtime_s": float(df.runtime_s.mean())}
        results.append(rec)
        df.to_csv(os.path.join(out, f"{args.surface}_sx{sx}_sy{sy}.csv"), index=False)
        print(f"sx={sx:<4} sy={sy:<4} (ratio {rec['ratio']:<5}) acc={acc:.4f} "
              f"({rec['delta_pp']:+5.1f}pp) R={rescued} B={broken} net={rec['net']:+d} "
              f"armused={rec['aniso_arm_used']}/{len(df)}", flush=True)

    sdf = pd.DataFrame(results).sort_values(["net", "acc_5px"], ascending=False)
    sdf.to_csv(os.path.join(out, f"{args.surface}_summary.csv"), index=False)
    with open(os.path.join(out, f"{args.surface}_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "baseline_acc": base_acc, "n": len(base),
                   "configs": results}, f, indent=2)
    print("\n=== ranked ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
