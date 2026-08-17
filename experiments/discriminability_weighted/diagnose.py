"""Oracle separation diagnostic for P3.

The dev sweep showed stage 2 firing on most pairs while almost never
changing the winner. That is either (a) the weights are degenerate - close
to uniform, so ZNCC_w is just ZNCC - or (b) the weights are real but
weighted ZNCC still cannot separate the true location from the decoy. Those
have completely different consequences, so this measures both directly
instead of sweeping more parameters at the symptom.

For every pair it reports, using ground truth ONLY to score an already-made
measurement (never to make one):

  reachable        is the true location within 5px of any pooled candidate?
                   If not, no re-scoring stage of any kind can fix the pair,
                   and it is out of P3's scope by construction.
  plain_margin     ZNCC(truth) - ZNCC(chosen), at the same hypothesis.
                   Negative on failures by definition.
  weighted_margin  the same margin under ZNCC_w.
  weight_gini      concentration of the weight map. 0 = uniform (the whole
                   idea is a no-op), 1 = all mass on one pixel.
  weight_on_edges  fraction of weight mass in the top-10% gradient-magnitude
                   pixels of the template. Uniform weights give 0.10 by
                   definition, so anything near 0.10 means the weighting is
                   not finding structure.

If weighted_margin > plain_margin on failures, the mechanism is real and the
problem is deployment. If it is not, P3 is refuted as a re-scorer.

    python -m experiments.discriminability_weighted.diagnose --surface development
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching, ranking

from .harness import _build_pool
from .weighted_zncc import extract_patch_at, uniform_weights, weighted_zncc_point
from .weights import confuser_variance_weights, estimate_lattice_vectors, lattice_shift_weights

ROOT = os.path.dirname(os.path.abspath(__file__))
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(ROOT, "data", "tune_degraded"),
    "validate_fresh": os.path.join(ROOT, "data", "validate_fresh"),
}


def gini(w: np.ndarray) -> float:
    v = np.sort(w.ravel().astype(np.float64))
    n = v.size
    if v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def edge_mass(w: np.ndarray, template: np.ndarray) -> float:
    gx = cv2.Sobel(template, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(template, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, 90.0)
    return float(w[mag >= thresh].sum())


def analyse(data_root: str, split: str, alpha: float, smooth: float) -> pd.DataFrame:
    manifest = load_manifest(data_root, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        psf_sigma, _, candidates = _build_pool(
            ref, search, candidate_generation.DEFAULT_SCALE_HYPOTHESES,
            candidate_generation.DEFAULT_ROTATION_HYPOTHESES, True)
        ranked = ranking.rank_classical(candidates)
        chosen = ranked[0]
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

        dists = [np.hypot(c.x - gt_x, c.y - gt_y) for c in candidates]
        nearest_i = int(np.argmin(dists))
        reachable = bool(dists[nearest_i] <= 5.0)
        truth_cand = candidates[nearest_i]
        chosen_err = float(np.hypot(chosen.x - gt_x, chosen.y - gt_y))

        # Score each location under ITS OWN best hypothesis, exactly as the
        # re-scoring stage does. Scoring the true location under the chosen
        # candidate's scale/rotation would understate it whenever the two
        # sit at different hypotheses, which is precisely the rotation/scale
        # failure mode - that confound would fake a negative result.
        def _templ_and_patch(cand):
            t = matching.build_template(ref, cand.scale, cand.rotation_deg, psf_sigma)
            th_, tw_ = t.shape
            p = extract_patch_at(search, int(round(cand.x - tw_ / 2.0)),
                                 int(round(cand.y - th_ / 2.0)), t.shape)
            return t, p

        t_chosen, p_chosen = _templ_and_patch(chosen)
        t_truth, p_truth = _templ_and_patch(truth_cand)
        if p_chosen is None or p_truth is None:
            continue

        # Plain margin comes straight from the pool: post-deduplication each
        # candidate's .score is already the best over all hypotheses at that
        # location, which is the strongest form of the comparison.
        plain_chosen = float(chosen.score)
        plain_truth = float(truth_cand.score)

        w_lat_c, diag_lat = lattice_shift_weights(t_chosen, alpha=alpha, smooth_sigma=smooth)
        w_lat_t, _ = lattice_shift_weights(t_truth, alpha=alpha, smooth_sigma=smooth)
        lat_chosen = weighted_zncc_point(t_chosen, p_chosen, w_lat_c)
        lat_truth = weighted_zncc_point(t_truth, p_truth, w_lat_t)

        # Confuser weights are built from the rival set, then applied in each
        # candidate's own footprint (rivals resampled into it), mirroring the
        # harness rather than inventing a second convention here.
        def _cvar(templ, patch, rivals):
            tgt = (templ.shape[1], templ.shape[0])
            rs = [cv2.resize(r, tgt, interpolation=cv2.INTER_AREA) for r in rivals]
            w, _d = confuser_variance_weights(rs, alpha=alpha, smooth_sigma=smooth)
            return weighted_zncc_point(templ, patch, w)

        rivals = [p_chosen, p_truth]
        cv_chosen = _cvar(t_chosen, p_chosen, rivals)
        cv_truth = _cvar(t_truth, p_truth, rivals)
        tmpl, uni = t_chosen, uniform_weights(t_chosen.shape)
        w_lat, w_cv = w_lat_c, uniform_weights(t_chosen.shape)
        w_cv, _ = confuser_variance_weights(
            [cv2.resize(r, (t_chosen.shape[1], t_chosen.shape[0]), interpolation=cv2.INTER_AREA)
             for r in rivals], alpha=alpha, smooth_sigma=smooth)

        rows.append({
            "pair_id": row["pair_id"], "structural_family": row["structural_family"],
            "chosen_err_px": chosen_err, "correct": chosen_err <= 5.0,
            "reachable": reachable, "nearest_cand_dist_px": float(dists[nearest_i]),
            "pool_size": len(candidates), "psf_sigma": psf_sigma,
            "plain_margin": plain_truth - plain_chosen,
            "lattice_margin": lat_truth - lat_chosen,
            "confuser_margin": cv_truth - cv_chosen,
            "lattice_vectors": diag_lat.get("n_vectors", 0),
            "lattice_fellback": diag_lat.get("fell_back", False),
            "lattice_gini": gini(w_lat), "confuser_gini": gini(w_cv),
            "lattice_edge_mass": edge_mass(w_lat, tmpl),
            "confuser_edge_mass": edge_mass(w_cv, tmpl),
            "uniform_edge_mass": edge_mass(uni, tmpl),
        })
        print(f"  {row['pair_id']:32s} err={chosen_err:8.2f} reach={reachable} "
              f"plain={rows[-1]['plain_margin']:+.4f} lat={rows[-1]['lattice_margin']:+.4f} "
              f"cvar={rows[-1]['confuser_margin']:+.4f} gini={rows[-1]['lattice_gini']:.3f}",
              flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="development", choices=sorted(SURFACES))
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--smooth", type=float, default=1.0)
    args = p.parse_args()

    df = analyse(SURFACES[args.surface], "development", args.alpha, args.smooth)
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{args.surface}_diagnostic.csv"), index=False)

    fails = df[~df.correct]
    reach_fails = fails[fails.reachable]
    print("\n================ SUMMARY ================")
    print(f"pairs                       : {len(df)}")
    print(f"correct                     : {int(df.correct.sum())}")
    print(f"failures                    : {len(fails)}")
    print(f"  of which REACHABLE        : {len(reach_fails)}  <- P3's addressable set")
    print(f"  of which unreachable      : {len(fails) - len(reach_fails)}  (candidate generation)")
    print()
    print("weight-map health (alpha=%.2f)" % args.alpha)
    print(f"  lattice gini              : {df.lattice_gini.mean():.3f}  (0 = uniform)")
    print(f"  confuser gini             : {df.confuser_gini.mean():.3f}")
    print(f"  uniform edge mass         : {df.uniform_edge_mass.mean():.3f}  (by definition ~0.10)")
    print(f"  lattice edge mass         : {df.lattice_edge_mass.mean():.3f}")
    print(f"  confuser edge mass        : {df.confuser_edge_mass.mean():.3f}")
    print(f"  lattice fell back to unif : {int(df.lattice_fellback.sum())}/{len(df)}")
    print(f"  mean lattice vectors      : {df.lattice_vectors.mean():.2f}")
    if len(reach_fails):
        print("\nmargins on REACHABLE failures (need > 0 to flip; higher is better)")
        for col in ("plain_margin", "lattice_margin", "confuser_margin"):
            v = reach_fails[col]
            print(f"  {col:17s} mean={v.mean():+.5f} median={v.median():+.5f} "
                  f"n>0={int((v > 0).sum())}/{len(v)}")
        print("\n  per-pair (reachable failures)")
        print(reach_fails[["pair_id", "chosen_err_px", "plain_margin",
                            "lattice_margin", "confuser_margin"]].to_string(index=False))
    corr = df[df.correct]
    if len(corr):
        print("\nmargins on CORRECT pairs (must stay > 0 or the change breaks them)")
        for col in ("plain_margin", "lattice_margin", "confuser_margin"):
            v = corr[col]
            print(f"  {col:17s} mean={v.mean():+.5f} n<=0={int((v <= 0).sum())}/{len(v)}")


if __name__ == "__main__":
    main()
