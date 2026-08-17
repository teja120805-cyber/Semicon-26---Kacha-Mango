"""Which image-domain intervention actually lifts the TRUE location's score?

`experiments/REACHABILITY_CAMPAIGN.md` concluded that neither better scoring
of the pool nor a bigger pool helps, and that the only lever the evidence
supports is **template fidelity** - making the true location score highly
enough to win outright. `reports/PROJECT_STATUS.md` item 1 says the same:
the PSF fix lifted fidelity at the true location from ~0.78 to ~0.85 while
the Search content itself supports ~0.95, and names the remaining sources as
dose/noise mismatch and unmodelled raster shear/jitter.

Rather than guess an intervention and measure it end-to-end (which is how
`experiments/anisotropic_psf/` produced a dev-only mirage), this measures the
quantity that has to move first:

    margin = ZNCC(truth) - ZNCC(winning decoy)

If an intervention does not increase this margin, it cannot possibly help,
and no end-to-end run is warranted. If it does, an end-to-end run is.

Interventions tested, each with a stated reason:

  none            control - reproduces the production score exactly
  blur_search     mild Gaussian on the Search image. Shot noise at
                  dose_search 40-220 is real; blurring trades resolution for
                  SNR. Applied to Search only, so it is NOT the same lever as
                  the integrated template PSF match.
  median_search   3x3 median. Targets impulse noise specifically
                  (salt-and-pepper, charging streaks) which a Gaussian
                  handles badly and which two challenge families carry.
  highpass_both   subtract a heavily-blurred copy from both images. ZNCC
                  removes the window MEAN but not a window-scale GRADIENT,
                  so vignette and gamma leave a slow ramp inside the
                  template footprint that survives normalization. This
                  removes it.
  bandpass_both   high-pass plus a mild low-pass - the band-limited variant
                  `experiments/parallel_pipeline/` §5 recommended as P1's
                  successor, but in the SPATIAL domain, where it cannot
                  amplify the noise floor the way dividing by |F|^2rho did.

Every intervention is applied identically to both the true and the decoy
location, so a uniform score change cancels and only genuine differential
movement registers.

    python -m experiments.template_fidelity_ablation.run --surface tune_degraded
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching, ranking
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}


def _hp(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


INTERVENTIONS = {
    "none": lambda r, s: (r, s),
    "blur_search_0.5": lambda r, s: (r, cv2.GaussianBlur(s, (0, 0), 0.5, borderType=cv2.BORDER_REPLICATE)),
    "blur_search_1.0": lambda r, s: (r, cv2.GaussianBlur(s, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)),
    "median_search_3": lambda r, s: (r, cv2.medianBlur(s.astype(np.uint8), 3).astype(np.float32)),
    "highpass_both_8": lambda r, s: (_hp(r, 80.0), _hp(s, 8.0)),
    "highpass_both_16": lambda r, s: (_hp(r, 160.0), _hp(s, 16.0)),
    "bandpass_both": lambda r, s: (
        cv2.GaussianBlur(_hp(r, 160.0), (0, 0), 5.0, borderType=cv2.BORDER_REPLICATE),
        cv2.GaussianBlur(_hp(s, 16.0), (0, 0), 0.5, borderType=cv2.BORDER_REPLICATE)),
}
# highpass sigmas are 10x apart between Reference and Search because the
# Reference is 10x finer: the same physical spatial frequency corresponds to
# a 10x larger pixel sigma there. Using one sigma for both would high-pass
# the two images at different physical scales and confound the test.


def zncc_at(reference, search, scale, rot, psf, cx, cy):
    t = matching.build_template(reference, scale, rot, psf)
    th, tw = t.shape
    x0, y0 = int(round(cx - tw / 2.0)), int(round(cy - th / 2.0))
    if x0 < 0 or y0 < 0 or y0 + th > search.shape[0] or x0 + tw > search.shape[1]:
        return None
    patch = search[y0:y0 + th, x0:x0 + tw]
    t64, p64 = t.astype(np.float64), patch.astype(np.float64)
    t64 -= t64.mean()
    p64 -= p64.mean()
    d = float(np.sqrt((t64 * t64).sum() * (p64 * p64).sum()))
    return float((t64 * p64).sum() / d) if d > 1e-12 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    args = p.parse_args()

    manifest = load_manifest(SURFACES[args.surface], "development")
    rows = []
    for _, row in manifest.iterrows():
        ref0 = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        srch0 = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = (float(row["gt_x"]), float(row["gt_y"]))

        # Production pool, to identify the decoy that actually wins.
        best = None
        for sigma in (0.0, PSF_MATCH_SIGMA):
            pool = candidate_generation.deduplicate_by_location(
                candidate_generation.build_candidate_pool(ref0, srch0, psf_sigma=sigma))
            g = _decisiveness(pool)
            if best is None or g > best[0]:
                best = (g, sigma, pool)
        _, psf, pool = best
        chosen = ranking.rank_classical(pool)[0]
        err = float(np.hypot(chosen.x - gt[0], chosen.y - gt[1]))
        if err <= 5.0:
            continue  # only failures carry a meaningful truth-vs-decoy margin

        # Truth is scored under the hypothesis that suits IT best, searched
        # over the same grid production uses - not under the decoy's.
        rec = {"pair_id": row["pair_id"], "structural_family": row["structural_family"],
               "chosen_err_px": err}
        for name, fn in INTERVENTIONS.items():
            ref, srch = fn(ref0.copy(), srch0.copy())
            s_decoy = zncc_at(ref, srch, chosen.scale, chosen.rotation_deg, psf, chosen.x, chosen.y)
            s_truth = None
            for sc in candidate_generation.DEFAULT_SCALE_HYPOTHESES:
                for rt in candidate_generation.DEFAULT_ROTATION_HYPOTHESES:
                    v = zncc_at(ref, srch, sc, rt, psf, gt[0], gt[1])
                    if v is not None and (s_truth is None or v > s_truth):
                        s_truth = v
            if s_decoy is None or s_truth is None:
                continue
            rec[f"{name}_truth"] = s_truth
            rec[f"{name}_decoy"] = s_decoy
            rec[f"{name}_margin"] = s_truth - s_decoy
        rows.append(rec)
        base = rec.get("none_margin", float("nan"))
        print(f"  {row['pair_id']:32s} err={err:8.2f} base_margin={base:+.4f} "
              + " ".join(f"{k.split('_')[0][:4]}={rec.get(k, float('nan')):+.4f}"
                          for k in ("highpass_both_16_margin", "bandpass_both_margin",
                                    "median_search_3_margin")), flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{args.surface}_ablation.csv"), index=False)

    print(f"\n=========== TRUTH-vs-DECOY MARGIN ON {len(df)} FAILURES "
          f"({args.surface}) ===========")
    print("positive margin => truth would outscore the decoy => the pair becomes winnable")
    print(f"{'intervention':>20} {'mean margin':>13} {'median':>10} {'n>0':>7} {'vs control':>12}")
    ctrl = df["none_margin"]
    summary = {}
    for name in INTERVENTIONS:
        col = f"{name}_margin"
        if col not in df:
            continue
        v = df[col]
        better = int((v > ctrl).sum())
        summary[name] = {"mean": float(v.mean()), "median": float(v.median()),
                         "n_positive": int((v > 0).sum()), "n_better_than_control": better}
        print(f"{name:>20} {v.mean():>13.5f} {v.median():>10.5f} "
              f"{int((v > 0).sum()):>4}/{len(v):<3} {better:>7}/{len(v):<4}")

    with open(os.path.join(out, f"{args.surface}_ablation.json"), "w") as f:
        json.dump({"surface": args.surface, "n_failures": len(df), "summary": summary}, f, indent=2)

    print("\ninterpretation: an intervention is worth an end-to-end run only if it")
    print("raises n>0 above the control's. A higher mean margin with the same n>0")
    print("moves pairs that were already winnable and is not evidence of a fix.")


if __name__ == "__main__":
    main()
