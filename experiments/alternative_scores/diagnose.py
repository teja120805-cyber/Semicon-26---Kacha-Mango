"""Two alternative similarity measures, screened against the same bar as DDIS.

Both are gaps identified by searching current literature and patents against
`reports/RESEARCH_SURVEY_SCORING.md`:

**OTSDF / MACE correlation filters.** MOSSE, MACE and UMACE appear in the
survey's source list but never became a P1–P6 proposal. A MACE filter is
designed to produce a sharp peak at the target while *minimising average
correlation energy everywhere else* — i.e. explicit sidelobe suppression, which
is precisely the "many near-equal peaks" problem here. The Optimal Trade-off
SDF (OTSDF) adds a noise-tolerance parameter:

    H(f) = X*(f) / ( alpha*|X(f)|^2 + (1-alpha)*mean|X|^2 )

  * `alpha = 0` -> plain matched filter, i.e. production correlation (null control)
  * `alpha = 1` -> MACE / inverse filter, maximum sidelobe suppression

**Phase-based matching** (the idea behind MS-POFT, 2025, and phase congruency
generally). Local phase is invariant to contrast and illumination in a way
gradient magnitude is not. Implemented here as a log-Gabor quadrature pair
giving a local-phase map, which is then correlated instead of intensity. This
is an approximation of the published method, not a reimplementation of it, and
is labelled as such.

**Honest prior, recorded before running.** Both are expected to fail.
`experiments/template_fidelity_ablation/` established that the discriminating
aperiodic content lives substantially at LOW spatial frequency, and both of
these suppress or discard low-frequency amplitude — OTSDF by dividing it out,
phase by discarding amplitude entirely. That is the same mechanism that sank
P1. They are run because they are cheap, and because "we expect it to fail" is
not a measurement.

**The bar (pre-registered, same as DDIS):** separate the median 0.029 ZNCC
deficit on the 22 reachable failures. >70% preference for truth to justify a
harness. Chance is 50%.

    python -m experiments.alternative_scores.diagnose
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
ALPHAS = (0.0, 0.3, 0.6, 0.9, 1.0)


def _zm(a):
    a = a.astype(np.float64)
    return a - a.mean()


def otsdf_score(template: np.ndarray, window: np.ndarray, alpha: float) -> float:
    """Cosine-normalised response of the OTSDF filter built from `template`,
    evaluated against `window` at zero shift. alpha=0 reduces to the matched
    filter, which is plain correlation."""
    t, w = _zm(template), _zm(window)
    X = np.fft.rfft2(t)
    Y = np.fft.rfft2(w)
    p = np.abs(X) ** 2
    denom = alpha * p + (1.0 - alpha) * float(p.mean()) + 1e-12
    H = np.conj(X) / denom
    num = float(np.real(np.sum(H * Y)))
    norm = float(np.sqrt(np.sum(np.abs(H) ** 2) * np.sum(np.abs(Y) ** 2))) + 1e-12
    return num / norm


def _log_gabor_phase(img: np.ndarray, wavelength: float = 8.0, sigma_ratio: float = 0.65):
    """Local phase via a log-Gabor quadrature pair (monogenic-style, isotropic).
    Returns the phase angle map. Approximation of phase-congruency front ends."""
    a = _zm(img)
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    r[0, 0] = 1.0
    f0 = 1.0 / wavelength
    lg = np.exp(-(np.log(r / f0) ** 2) / (2 * np.log(sigma_ratio) ** 2))
    lg[0, 0] = 0.0
    F = np.fft.rfft2(a) * lg
    even = np.fft.irfft2(F, s=a.shape)
    # Riesz transform -> odd component
    denom = np.where(r == 0, 1.0, r)
    odd = np.fft.irfft2(F * (-1j * (fx + 1j * fy) / denom), s=a.shape)
    return np.arctan2(np.real(odd), even)


def phase_score(template: np.ndarray, window: np.ndarray) -> float:
    """Agreement between local-phase maps: mean cos(phase difference), which is
    +1 for perfect phase alignment and 0 for unrelated structure."""
    pt = _log_gabor_phase(template)
    pw = _log_gabor_phase(window)
    return float(np.cos(pt - pw).mean())


def main() -> None:
    reach = pd.read_csv(REACH)
    targets = reach[(~reach.correct) & (reach.reachable)]
    manifests = {s: load_manifest("data", s).set_index("pair_id") for s in targets.split.unique()}
    print(f"screening {len(targets)} reachable failures; bar = >70% preference for truth\n")

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
        rec = {"pair_id": tr.pair_id, "split": tr.split,
               "zncc_margin": float(truth.score - chosen.score)}
        for a in ALPHAS:
            rec[f"otsdf{a}_margin"] = otsdf_score(tt, wt, a) - otsdf_score(tc, wc, a)
        rec["phase_margin"] = phase_score(tt, wt) - phase_score(tc, wc)
        rows.append(rec)
        print(f"  {tr.pair_id:30s} zncc={rec['zncc_margin']:+.4f}  "
              f"otsdf0={rec['otsdf0.0_margin']:+.4f} otsdf0.9={rec['otsdf0.9_margin']:+.4f} "
              f"otsdf1={rec['otsdf1.0_margin']:+.4f} phase={rec['phase_margin']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "alternative_scores.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"ALTERNATIVE SCORES ON {len(df)} REACHABLE FAILURES")
    print("=" * 60)
    print(f"  {'measure':>18} {'median margin':>15} {'prefers truth':>15}")
    summary = {}
    cols = [("ZNCC (reference)", "zncc_margin")] + \
           [(f"OTSDF alpha={a}", f"otsdf{a}_margin") for a in ALPHAS] + \
           [("phase (log-Gabor)", "phase_margin")]
    for name, col in cols:
        v = df[col]
        n = int((v > 0).sum())
        summary[name] = {"median": float(v.median()), "prefers_truth": n, "n": len(df)}
        flag = "  <- PASSES BAR" if n / max(len(df), 1) > 0.70 else ""
        print(f"  {name:>18} {v.median():>15.5f} {n:>8}/{len(df):<5} "
              f"({n/max(len(df),1):.0%}){flag}")
    print("\n  alpha=0 is the null control: it is the matched filter, so its sign pattern")
    print("  should track ZNCC's. Any measure not clearly above 50% is at chance.")
    with open(os.path.join(out, "alternative_scores.json"), "w") as f:
        json.dump({"n": len(df), "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
