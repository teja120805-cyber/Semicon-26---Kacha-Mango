#!/usr/bin/env python
"""Scaled adaptive sigma: applied = k * estimate_sigma(pair).

The unscaled estimator (run_experiment.py) protected every degraded family
but forfeited the gains - its absolute scale is dragged down by the Search
image's noise floor, while its RELATIVE ordering across families is
correct. This variant keeps the ordering and fixes the scale with a single
global constant derived on `development` ONLY:

    k = (dev-optimal global sigma) / (dev median estimated sigma)
      = 1.6 / 1.029 = 1.555

Both inputs come from the development split alone - 1.6 from
psf_matched_template's dev sweep, 1.029 from this experiment's dev run.
No frozen-benchmark result is used to set k.
"""
import json, os, sys, time
import numpy as np, pandas as pd, cv2
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.evaluate import load_manifest
from evaluation import benchmark, metrics
from harness import localize_adaptive
from spectral_sigma import estimate_sigma

K = 1.6 / 1.029
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
SPLITS = ["development", "validation", "held_out", "challenge", "cross_generator"]

def run_split(split):
    man = load_manifest(DATA_ROOT, split); rows = []
    for _, row in man.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        se = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        s = float(np.clip(K * estimate_sigma(ref.astype(np.float32), se.astype(np.float32)), 0.0, 2.5))
        res, _ = localize_adaptive(ref, se, sigma_override=s)
        rows.append({**row.to_dict(), "pred_x": res.x, "pred_y": res.y,
                     "error_px": float(np.hypot(res.x - row["gt_x"], res.y - row["gt_y"])),
                     "confidence": res.confidence, "ambiguity_ratio": res.ambiguity_ratio,
                     "ambiguous": res.ambiguous, "runtime_s": res.runtime_s,
                     "ranking_mode": "psf_matched_adaptive_scaled", "sigma_est": s})
    return pd.DataFrame(rows)

def main():
    print(f"=== scaled adaptive, k={K:.4f} (dev-derived) ===", flush=True)
    dfs = []
    for s in SPLITS:
        t0 = time.perf_counter(); df = run_split(s); dfs.append(df)
        print(f"  {s}: n={len(df)} acc@5px={(df.error_px<=5).mean():.3f} "
              f"sigma_med={df.sigma_est.median():.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    cand = pd.concat(dfs, ignore_index=True)
    cand.to_csv(os.path.join(OUT_DIR, "per_pair_results_adaptive_scaled.csv"), index=False)
    base = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(base, cand, seeds_agree=True)
    json.dump(gate, open(os.path.join(OUT_DIR, "integration_gate_scaled.json"), "w"), indent=2)
    json.dump(metrics.full_report(cand), open(os.path.join(OUT_DIR, "metrics_scaled.json"), "w"), indent=2)
    print("\n=== gate ==="); print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    ba, ca = float((base.error_px<=5).mean()), float((cand.error_px<=5).mean())
    print(f"\npooled {ba:.4f} -> {ca:.4f}  ({ca-ba:+.4f})")
    print("\n=== per split ===")
    for s in SPLITS:
        print(f"  {s:<16} {(base[base.split==s].error_px<=5).mean():.3f} -> {(cand[cand.split==s].error_px<=5).mean():.3f}")
    base = base.assign(fam=base.pair_id.str.rsplit("_", n=1).str[0])
    cand = cand.assign(fam=cand.pair_id.str.rsplit("_", n=1).str[0])
    bf = base.assign(ok=base.error_px<=5).groupby("fam").ok.mean()
    cf = cand.assign(ok=cand.error_px<=5).groupby("fam").ok.mean()
    sg = cand.groupby("fam").sigma_est.median()
    print("\n=== per family ===")
    for f in sorted(bf.index):
        d = cf[f]-bf[f]; mk = "  <-- REGRESSION" if d<-1e-9 else ("  <-- gain" if d>1e-9 else "")
        print(f"  {f:<28} sigma={sg[f]:.2f}  {bf[f]:.3f} -> {cf[f]:.3f} ({d:+.3f}){mk}")
    b = base.set_index("pair_id").error_px; c = cand.set_index("pair_id").error_px
    cm = b.index.intersection(c.index)
    r = int(((b[cm]>5)&(c[cm]<=5)).sum()); k2 = int(((b[cm]<=5)&(c[cm]>5)).sum())
    print(f"\nRescued {r}  Broken {k2}  net {r-k2:+d}")
    print(f"runtime {cand.runtime_s.sum()/pd.read_csv(BASELINE_CSV).runtime_s.sum():.2f}x")

main()
