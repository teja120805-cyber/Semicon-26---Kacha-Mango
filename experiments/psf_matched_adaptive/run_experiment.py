#!/usr/bin/env python
"""Frozen-benchmark evaluation of per-pair adaptive PSF matching.

There is NO hyperparameter sweep here, and that is deliberate. The sigma is
fully determined by spectral_sigma.estimate_sigma from the image pair; the
fit band and clamp are fixed a priori from sampling geometry. Since
`development` contains no degraded-acquisition family
(psf_matched_template/REPORT.md §3), anything tuned on it would inherit that
blind spot - so nothing is tuned. The method is run exactly once over the
full frozen benchmark and handed to the unmodified integration gate.

Never touches pipeline/, generator/, model/, or data/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = "/tmp/driftsense"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluate import load_manifest  # noqa: E402
from evaluation import benchmark, metrics  # noqa: E402

from harness import localize_adaptive  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_pair_results.csv")
SPLITS = ["development", "validation", "held_out", "challenge", "cross_generator"]


def run_split(split: str) -> pd.DataFrame:
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        if ref is None or search is None:
            raise FileNotFoundError(row["pair_id"])
        res, sigma = localize_adaptive(ref, search)
        rows.append({**row.to_dict(), "pred_x": res.x, "pred_y": res.y,
                     "error_px": float(np.hypot(res.x - row["gt_x"], res.y - row["gt_y"])),
                     "confidence": res.confidence, "ambiguity_ratio": res.ambiguity_ratio,
                     "ambiguous": res.ambiguous, "runtime_s": res.runtime_s,
                     "ranking_mode": "psf_matched_adaptive", "sigma_est": sigma})
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Full frozen benchmark, per-pair adaptive sigma (no tuning) ===")
    dfs = []
    for split in SPLITS:
        t0 = time.perf_counter()
        df = run_split(split)
        dfs.append(df)
        print(f"  {split}: n={len(df)} acc@5px={(df['error_px'] <= 5).mean():.3f} "
              f"sigma_med={df.sigma_est.median():.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    cand = pd.concat(dfs, ignore_index=True)
    cand.to_csv(os.path.join(OUT_DIR, "per_pair_results_adaptive.csv"), index=False)

    base = pd.read_csv(BASELINE_CSV)
    gate = benchmark.run_integration_gate(base, cand, seeds_agree=True)
    with open(os.path.join(OUT_DIR, "integration_gate_result.json"), "w") as f:
        json.dump(gate, f, indent=2)
    with open(os.path.join(OUT_DIR, "adaptive_metrics.json"), "w") as f:
        json.dump(metrics.full_report(cand), f, indent=2)

    print("\n=== Integration gate ===")
    print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
    b_acc, c_acc = float((base.error_px <= 5).mean()), float((cand.error_px <= 5).mean())
    print(f"\npooled {b_acc:.4f} -> {c_acc:.4f}  ({c_acc - b_acc:+.4f})")

    print("\n=== per split ===")
    for s in SPLITS:
        print(f"  {s:<16} {(base[base.split == s].error_px <= 5).mean():.3f} -> "
              f"{(cand[cand.split == s].error_px <= 5).mean():.3f}")

    base = base.assign(fam=base.pair_id.str.rsplit("_", n=1).str[0])
    cand = cand.assign(fam=cand.pair_id.str.rsplit("_", n=1).str[0])
    bf = base.assign(ok=base.error_px <= 5).groupby("fam").ok.mean()
    cf = cand.assign(ok=cand.error_px <= 5).groupby("fam").ok.mean()
    sg = cand.groupby("fam").sigma_est.median()
    print("\n=== per family (sigma = median estimated) ===")
    for fam in sorted(bf.index):
        d = cf[fam] - bf[fam]
        mark = "  <-- REGRESSION" if d < -1e-9 else ("  <-- gain" if d > 1e-9 else "")
        print(f"  {fam:<28} sigma={sg[fam]:.2f}  {bf[fam]:.3f} -> {cf[fam]:.3f} ({d:+.3f}){mark}")

    b = base.set_index("pair_id").error_px
    c = cand.set_index("pair_id").error_px
    common = b.index.intersection(c.index)
    r = int(((b[common] > 5) & (c[common] <= 5)).sum())
    k = int(((b[common] <= 5) & (c[common] > 5)).sum())
    print(f"\nRescued {r}  Broken {k}  net {r - k:+d}")
    print(f"runtime multiplier {cand.runtime_s.sum() / pd.read_csv(BASELINE_CSV).runtime_s.sum():.2f}x")


if __name__ == "__main__":
    main()
