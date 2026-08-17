"""Can the information ZNCC THROWS AWAY resolve the 1-D lattice ambiguity?

Two results from this session combine into a specific, previously-untested
question.

  * `experiments/axis_decomposition/` — the residual failures are
    **one-dimensional**: the error is 100% along a lattice axis at the median,
    with a cross-axis residual of only 1.47px. The pipeline is confused about
    *which lattice period* it sits on, nothing else.
  * `experiments/template_fidelity_ablation/` — the content that discriminates
    one location from another lives substantially at **LOW spatial frequency**;
    high-pass filtering destroys it, with a monotone dose-response.

Now the observation that ties them together. **ZNCC is zero-mean and
variance-normalised by construction** (`cv2.TM_CCOEFF_NORMED`): it subtracts the
window mean and divides by the window standard deviation. So the absolute
brightness level and the absolute contrast of a candidate window are
*discarded before scoring ever happens*. On a Search image carrying vignette,
gamma or illumination falloff, those quantities vary slowly and smoothly across
the field — which is exactly the kind of signal that could distinguish "period
n" from "period n+1" along an axis.

Every one of the sixteen prior attempts modified how the mid/high-frequency
structure is scored. **None used the low-frequency content that normalisation
removes.** That is the gap.

Signals tested at the true vs the chosen location (both under the winner's own
hypothesis):

  mean_agree   -|mean(T) - mean(W)|          absolute brightness match
  std_agree    -|log(std(T)/std(W))|         absolute contrast match
  envelope     corr of heavily-blurred T, W  slowly-varying illumination shape

**Honest caveat, recorded before running.** The Reference and Search travel
different degradation paths (`image_reference` vs `image_search`), so absolute
brightness may simply not transfer between them. If so these signals are
uninformative for a reason that has nothing to do with the lattice, and that
would be the finding.

**Bar (pre-registered, same as DDIS and OTSDF):** >70% preference for truth on
the 22 reachable failures to justify building anything. Chance is 50%.

    python -m experiments.lowfreq_1d.diagnose
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

ROOT = os.path.dirname(os.path.abspath(__file__))
REACH = os.path.join(os.path.dirname(ROOT), "reachability_verification",
                      "outputs", "frozen_reachability.csv")


def envelope(a: np.ndarray, frac: float = 0.25) -> np.ndarray:
    s = max(2.0, frac * min(a.shape))
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), s,
                             borderType=cv2.BORDER_REPLICATE)


def signals(t: np.ndarray, w: np.ndarray) -> dict:
    t = t.astype(np.float64)
    w = w.astype(np.float64)
    mean_agree = -abs(float(t.mean()) - float(w.mean()))
    st, sw = float(t.std()), float(w.std())
    std_agree = -abs(np.log(max(st, 1e-6) / max(sw, 1e-6)))
    et, ew = envelope(t).ravel(), envelope(w).ravel()
    et = et - et.mean()
    ew = ew - ew.mean()
    den = float(np.linalg.norm(et) * np.linalg.norm(ew))
    env = float(et @ ew / den) if den > 1e-9 else 0.0
    return {"mean_agree": mean_agree, "std_agree": std_agree, "envelope": env}


def main() -> None:
    reach = pd.read_csv(REACH)
    targets = reach[(~reach.correct) & (reach.reachable)]
    manifests = {s: load_manifest("data", s).set_index("pair_id") for s in targets.split.unique()}
    print(f"{len(targets)} reachable failures; bar = >70% preference for truth\n")

    rows = []
    for _, tr in targets.iterrows():
        row = manifests[tr.split].loc[tr.pair_id]
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        srch = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = (float(row["gt_x"]), float(row["gt_y"]))

        best = None
        for sigma in (0.0, PSF_MATCH_SIGMA):
            pool = candidate_generation.deduplicate_by_location(
                candidate_generation.build_candidate_pool(ref, srch, psf_sigma=sigma))
            gap = _decisiveness(pool)
            if best is None or gap > best[0]:
                best = (gap, sigma, pool)
        _, sigma, pool = best
        chosen = ranking.rank_classical(pool)[0]
        truth = min(pool, key=lambda c: np.hypot(c.x - gt[0], c.y - gt[1]))

        def win(c):
            t = matching.build_template(ref, c.scale, c.rotation_deg, sigma)
            th, tw = t.shape
            x0, y0 = int(round(c.x - tw / 2.0)), int(round(c.y - th / 2.0))
            if x0 < 0 or y0 < 0 or y0 + th > srch.shape[0] or x0 + tw > srch.shape[1]:
                return None, None
            return t, srch[y0:y0 + th, x0:x0 + tw]

        tc, wc = win(chosen)
        tt, wt = win(truth)
        if wc is None or wt is None:
            continue
        sc, st_ = signals(tc, wc), signals(tt, wt)
        rec = {"pair_id": tr.pair_id, "split": tr.split, "family": tr.structural_family}
        for k in sc:
            rec[f"{k}_margin"] = st_[k] - sc[k]
        rows.append(rec)
        print(f"  {tr.pair_id:30s} mean={rec['mean_agree_margin']:+8.3f} "
              f"std={rec['std_agree_margin']:+7.4f} env={rec['envelope_margin']:+7.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "lowfreq_signals.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"LOW-FREQUENCY SIGNALS ON {len(df)} REACHABLE FAILURES")
    print("=" * 60)
    print(f"  {'signal':>14} {'median margin':>15} {'prefers truth':>16}")
    summary = {}
    for k in ("mean_agree", "std_agree", "envelope"):
        v = df[f"{k}_margin"]
        n = int((v > 0).sum())
        summary[k] = {"median": float(v.median()), "prefers_truth": n, "n": len(df)}
        flag = "  <- PASSES BAR" if n / max(len(df), 1) > 0.70 else ""
        print(f"  {k:>14} {v.median():>15.4f} {n:>9}/{len(df):<4} "
              f"({n/max(len(df),1):.0%}){flag}")
    # A combined vote, since the three are not independent but may be complementary
    votes = sum((df[f"{k}_margin"] > 0).astype(int) for k in ("mean_agree", "std_agree", "envelope"))
    maj = int((votes >= 2).sum())
    print(f"  {'majority of 3':>14} {'-':>15} {maj:>9}/{len(df):<4} "
          f"({maj/max(len(df),1):.0%})")
    print("\n  If all sit near chance, the Reference/Search degradation paths differ")
    print("  enough that absolute brightness does not transfer — which closes the")
    print("  low-frequency route regardless of the 1-D framing being correct.")
    with open(os.path.join(out, "lowfreq_signals.json"), "w") as f:
        json.dump({"n": len(df), "summary": summary,
                   "majority_prefers_truth": maj}, f, indent=2)


if __name__ == "__main__":
    main()
