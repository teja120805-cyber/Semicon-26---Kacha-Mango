"""Are the remaining failures a 1-D problem?

Motivated by US8073242B2 (SEM addressing-point recipes), which
`reports/RESEARCH_SURVEY_SCORING.md` §2 cites and flags — "our two axes may be
separately solvable" — but never turned into a proposal. The patent's own words:
*"ignoring the uniqueness of any APs in the X-direction and attaching importance
only to the uniqueness of APs in the Y-direction."*

The patent's mechanism itself does not transfer: it is a **recipe-design**
strategy for choosing where to place an addressing point, and here the crop is
given rather than chosen. What does transfer is the underlying question, which
is cheap to answer and has never been asked of this data:

  **When the pipeline picks the wrong location, is the error along a lattice
  axis, or in an arbitrary direction?**

If failures are predominantly *along-lattice*, the residual ambiguity is
effectively **one-dimensional** — the pipeline already knows the cross-lattice
coordinate and is only confused about which lattice period it is on. That is a
far easier problem than 2-D search, and would justify a 1-D disambiguator
(which nothing in this project has tried). If the errors are isotropic, the
framing is wrong and the direction is closed.

Measured on the 22 reachable failures identified by
`experiments/reachability_verification/`, decomposing the error vector
(chosen − truth) onto the lattice basis estimated from the template itself.

    python -m experiments.axis_decomposition.diagnose
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, matching, ranking
from pipeline.localize import PSF_MATCH_SIGMA, _decisiveness

from experiments.discriminability_weighted.weights import estimate_lattice_vectors

ROOT = os.path.dirname(os.path.abspath(__file__))
REACH = os.path.join(os.path.dirname(ROOT), "reachability_verification",
                      "outputs", "frozen_reachability.csv")


def main() -> None:
    reach = pd.read_csv(REACH)
    targets = reach[(~reach.correct) & (reach.reachable)]
    manifests = {s: load_manifest("data", s).set_index("pair_id") for s in targets.split.unique()}
    print(f"decomposing {len(targets)} reachable failures onto their lattice basis\n")

    rows = []
    for _, tr in targets.iterrows():
        row = manifests[tr.split].loc[tr.pair_id]
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = np.array([float(row["gt_x"]), float(row["gt_y"])])

        best = None
        for sigma in (0.0, PSF_MATCH_SIGMA):
            pool = candidate_generation.deduplicate_by_location(
                candidate_generation.build_candidate_pool(ref, srch, psf_sigma=sigma))
            gap = _decisiveness(pool)
            if best is None or gap > best[0]:
                best = (gap, sigma, pool)
        _, sigma, pool = best
        chosen = ranking.rank_classical(pool)[0]
        err = np.array([chosen.x, chosen.y]) - gt
        err_norm = float(np.linalg.norm(err))

        tmpl = matching.build_template(ref, chosen.scale, chosen.rotation_deg, sigma)
        vecs = estimate_lattice_vectors(tmpl)
        if not vecs:
            rows.append({"pair_id": tr.pair_id, "err_px": err_norm, "has_lattice": False})
            continue

        # Lattice vectors are measured in TEMPLATE pixels; the Search image is
        # in the same pixel scale as the template (the template is the
        # downscaled Reference), so they transfer directly.
        best_align, best_v = None, None
        for dy, dx in vecs:
            v = np.array([dx, dy], dtype=float)
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            u = v / n
            along = abs(float(err @ u))
            cross = float(np.linalg.norm(err - (err @ u) * u))
            ratio = along / max(err_norm, 1e-9)
            if best_align is None or ratio > best_align[0]:
                best_align, best_v = (ratio, along, cross, n), (dx, dy)
        ratio, along, cross, pitch = best_align
        # How close is the along-component to an integer multiple of the pitch?
        periods = along / pitch
        phase_err = abs(periods - round(periods))
        rows.append({"pair_id": tr.pair_id, "split": tr.split, "family": tr.structural_family,
                      "err_px": err_norm, "has_lattice": True,
                      "along_px": along, "cross_px": cross, "along_fraction": ratio,
                      "pitch_px": pitch, "periods": periods, "phase_err": phase_err,
                      "lattice_vec": str(best_v)})
        print(f"  {tr.pair_id:30s} err={err_norm:8.2f}  along={along:8.2f} cross={cross:7.2f}  "
              f"along_frac={ratio:.3f}  periods={periods:6.2f} (phase err {phase_err:.2f})",
              flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "axis_decomposition.csv"), index=False)

    ok = df[df.has_lattice]
    print("\n" + "=" * 60)
    print(f"AXIS DECOMPOSITION OF {len(ok)} REACHABLE FAILURES")
    print("=" * 60)
    if len(ok):
        print(f"  median along-axis fraction of the error : {ok.along_fraction.median():.3f}")
        print(f"  failures >80% along a lattice axis      : {int((ok.along_fraction > 0.8).sum())}/{len(ok)}")
        print(f"  failures >90% along a lattice axis      : {int((ok.along_fraction > 0.9).sum())}/{len(ok)}")
        print(f"  median cross-axis residual              : {ok.cross_px.median():.2f} px")
        print(f"  within a quarter-period of an integer   : "
              f"{int((ok.phase_err < 0.25).sum())}/{len(ok)}")
        print("\n  A 1-D framing is supported only if the along-axis fraction is high AND the")
        print("  cross-axis residual is small (the pipeline already has that coordinate right).")
    with open(os.path.join(out, "axis_decomposition.json"), "w") as f:
        json.dump({"n": int(len(ok)),
                   "median_along_fraction": float(ok.along_fraction.median()) if len(ok) else None,
                   "n_above_0.8": int((ok.along_fraction > 0.8).sum()) if len(ok) else 0,
                   "median_cross_px": float(ok.cross_px.median()) if len(ok) else None}, f, indent=2)


if __name__ == "__main__":
    main()
