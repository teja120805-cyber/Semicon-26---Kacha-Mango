"""P4 - peak-to-sidelobe ratio, in the two places it could matter.

    PSR = (peak - mu_sidelobe) / sigma_sidelobe

with a small exclusion window around the peak. `reports/RESEARCH_SURVEY_SCORING.md`
§P4 proposes it as (a) a better-founded replacement for the pool-internal gap
statistic and (b) the principled replacement for `AMBIGUITY_THRESHOLD = 0.92`,
which `reports/PROJECT_STATUS.md` records as miscalibrated - firing on 128/156
pairs at 31% precision.

Two distinct questions, measured separately because they have different stakes:

  A. SELECTOR. `pipeline/localize.py` picks between the two PSF arms by
     `_decisiveness` - the score gap to the best candidate >10px away. Would
     PSR pick better? This one moves ACCURACY, so it is a real candidate
     change and is measured as rescued/broken against production.

  B. CONFIDENCE. Which statistic best separates correct from wrong pairs, and
     what coverage/accuracy curve does each buy? This changes no prediction -
     it is a calibration deliverable, and it is what `AMBIGUITY_THRESHOLD`
     should be derived from.

Null control for A: `--selector decisiveness` reproduces production exactly.

    python -m experiments.psr_confidence.run --surface tune_degraded
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from evaluation.evaluate import load_manifest
from pipeline import candidate_generation, feature_extraction, matching, ranking, refinement
from pipeline.localize import PSF_GAP_DISTINCT_PX, PSF_MATCH_SIGMA, _decisiveness

ROOT = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(ROOT), "discriminability_weighted")
SURFACES = {
    "development": os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data"),
    "tune_degraded": os.path.join(DW, "data", "tune_degraded"),
    "validate_fresh": os.path.join(DW, "data", "validate_fresh"),
}

PSR_EXCLUDE_PX = 11  # half-width of the excluded window around the peak


def psr(score_map: np.ndarray, peak_xy: tuple[int, int], exclude: int = PSR_EXCLUDE_PX) -> float:
    """Peak-to-sidelobe ratio. The sidelobe region is everything outside an
    `exclude`-radius box around the peak; a degenerate sidelobe (too few
    pixels, or zero variance) returns nan so callers cannot silently treat
    an undefined value as a confident one."""
    x, y = peak_xy
    h, w = score_map.shape
    mask = np.ones((h, w), dtype=bool)
    y0, y1 = max(0, y - exclude), min(h, y + exclude + 1)
    x0, x1 = max(0, x - exclude), min(w, x + exclude + 1)
    mask[y0:y1, x0:x1] = False
    side = score_map[mask]
    if side.size < 100:
        return float("nan")
    mu, sd = float(side.mean()), float(side.std())
    if sd <= 1e-9:
        return float("nan")
    return (float(score_map[y, x]) - mu) / sd


def arm_stats(reference, search, sigma):
    """Build one PSF arm's pool and both candidate selector statistics."""
    raw = candidate_generation.build_candidate_pool(
        reference, search, scale_hypotheses=candidate_generation.DEFAULT_SCALE_HYPOTHESES,
        rotation_hypotheses=candidate_generation.DEFAULT_ROTATION_HYPOTHESES, psf_sigma=sigma)
    pool = candidate_generation.deduplicate_by_location(raw)
    top = ranking.rank_classical(pool)[0]
    tmpl = matching.build_template(reference, top.scale, top.rotation_deg, sigma)
    smap = matching.correlate(search, tmpl)
    px = int(np.clip(round(top.x - tmpl.shape[1] / 2.0), 0, smap.shape[1] - 1))
    py = int(np.clip(round(top.y - tmpl.shape[0] / 2.0), 0, smap.shape[0] - 1))
    return {"sigma": sigma, "pool": pool, "decisiveness": _decisiveness(pool),
            "psr": psr(smap, (px, py))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="tune_degraded", choices=sorted(SURFACES))
    args = p.parse_args()

    manifest = load_manifest(SURFACES[args.surface], "development")
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = (float(row["gt_x"]), float(row["gt_y"]))

        arms = [arm_stats(ref, search, s) for s in (0.0, PSF_MATCH_SIGMA)]

        rec = {"pair_id": row["pair_id"], "structural_family": row["structural_family"]}
        for name, key in (("decisiveness", "decisiveness"), ("psr", "psr")):
            # Ties go to sigma=0.0, matching production's tuple order.
            vals = [a[key] for a in arms]
            best = 0 if not (vals[1] > vals[0]) else 1
            arm = arms[best]
            ranked = ranking.apply_center_tiebreak(
                ranking.rank_classical(arm["pool"]), search.shape)
            x, y = refinement.refine(ref, search, ranked[0], arm["sigma"])
            err = float(np.hypot(x - gt[0], y - gt[1]))
            rec[f"{name}_sigma"] = arm["sigma"]
            rec[f"{name}_err_px"] = err
            rec[f"{name}_correct"] = err <= 5.0
            if name == "decisiveness":
                scores = sorted((c.score for c in arm["pool"]), reverse=True)
                rec["ambiguity_ratio"] = feature_extraction.ambiguity_ratio(scores)
                rec["winner_score"] = float(ranked[0].score)
                rec["gap"] = arm["decisiveness"]
                rec["psr_value"] = arm["psr"]
        rows.append(rec)
        print(f"  {row['pair_id']:32s} dec_err={rec['decisiveness_err_px']:8.2f} "
              f"psr_err={rec['psr_err_px']:8.2f} "
              f"armswap={rec['decisiveness_sigma'] != rec['psr_sigma']}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{args.surface}_psr.csv"), index=False)

    # ---- A. selector -------------------------------------------------------
    dec_acc = float(df.decisiveness_correct.mean())
    psr_acc = float(df.psr_correct.mean())
    rescued = int((~df.decisiveness_correct & df.psr_correct).sum())
    broken = int((df.decisiveness_correct & ~df.psr_correct).sum())
    swaps = int((df.decisiveness_sigma != df.psr_sigma).sum())
    print("\n=========== A. PSR AS THE DUAL-ARM SELECTOR ===========")
    print(f"surface {args.surface}  n={len(df)}")
    print(f"  production (decisiveness gap) acc@5px = {dec_acc:.4f}   <- null control")
    print(f"  PSR selector                  acc@5px = {psr_acc:.4f}")
    print(f"  arm choice differs on {swaps}/{len(df)} pairs; rescued={rescued} broken={broken} "
          f"net={rescued - broken:+d}")

    # ---- B. confidence -----------------------------------------------------
    correct = df.decisiveness_correct.values
    print("\n=========== B. CONFIDENCE STATISTICS ===========")
    print(f"{'statistic':>18} {'AUC':>7} {'median correct':>16} {'median wrong':>14} {'ratio':>8}")
    stats = {}
    for name, col, higher_is_better in (("gap", "gap", True), ("PSR", "psr_value", True),
                                         ("winner_score", "winner_score", True),
                                         ("ambiguity_ratio", "ambiguity_ratio", False)):
        v = df[col].values.astype(float)
        ok = np.isfinite(v)
        s = v if higher_is_better else -v
        pos, neg = s[ok & correct], s[ok & ~correct]
        if len(pos) == 0 or len(neg) == 0:
            continue
        auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        mc, mw = float(np.median(v[ok & correct])), float(np.median(v[ok & ~correct]))
        stats[name] = {"auc": auc, "median_correct": mc, "median_wrong": mw}
        print(f"{name:>18} {auc:>7.3f} {mc:>16.4f} {mw:>14.4f} "
              f"{(mc / mw if mw not in (0.0,) else float('nan')):>8.2f}")

    print("\n  coverage / accuracy if we answer only the most confident fraction")
    print(f"{'statistic':>18} {'50% cov':>9} {'70% cov':>9} {'90% cov':>9}")
    for name, col, higher_is_better in (("gap", "gap", True), ("PSR", "psr_value", True),
                                         ("ambiguity_ratio", "ambiguity_ratio", False)):
        v = df[col].values.astype(float)
        ok = np.isfinite(v)
        s = (v if higher_is_better else -v)[ok]
        c = correct[ok]
        order = np.argsort(-s)
        line = f"{name:>18}"
        for cov in (0.5, 0.7, 0.9):
            n = max(1, int(round(cov * len(s))))
            line += f" {c[order[:n]].mean():>9.3f}"
        print(line)

    with open(os.path.join(out, f"{args.surface}_psr_summary.json"), "w") as f:
        json.dump({"surface": args.surface, "n": len(df),
                   "selector": {"decisiveness_acc": dec_acc, "psr_acc": psr_acc,
                                 "rescued": rescued, "broken": broken, "arm_swaps": swaps},
                   "confidence": stats}, f, indent=2)


if __name__ == "__main__":
    main()
