"""Accuracy/error metrics and structural/condition breakdowns.

All functions operate on a per-pair results DataFrame (produced by
evaluate.py) with at least: `error_px`, `runtime_s`, and whichever grouping
columns a given breakdown needs. Nothing here reads or writes ground truth
into the pipeline - this module only ever consumes already-computed errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCES_PX = (1, 2, 3, 4, 5)


def summarize(df: pd.DataFrame) -> dict:
    """The full metric set required by the brief: accuracy@1-5px, median,
    mean, P90, P95, max, >10px and >50px failure rates, runtime per pair."""
    if len(df) == 0:
        return {"n": 0}
    err = df["error_px"].to_numpy()
    out = {"n": int(len(df))}
    for k in TOLERANCES_PX:
        out[f"accuracy_at_{k}px"] = float(np.mean(err <= k))
    out["median_error_px"] = float(np.median(err))
    out["mean_error_px"] = float(np.mean(err))
    out["p90_error_px"] = float(np.percentile(err, 90))
    out["p95_error_px"] = float(np.percentile(err, 95))
    out["max_error_px"] = float(np.max(err))
    out["failure_rate_gt_10px"] = float(np.mean(err > 10))
    out["failure_rate_gt_50px"] = float(np.mean(err > 50))
    if "runtime_s" in df.columns:
        out["mean_runtime_s"] = float(df["runtime_s"].mean())
        out["median_runtime_s"] = float(df["runtime_s"].median())
    return out


def _bucket_noise(dose_search: float) -> str:
    if dose_search >= 150:
        return "low_noise"
    if dose_search >= 60:
        return "medium_noise"
    return "high_noise"


def _bucket_score(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def add_breakdown_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the categorical columns every breakdown in the brief needs,
    from raw per-pair metadata already present in the results DataFrame."""
    df = df.copy()
    df["noise_level"] = df["dose_search"].apply(_bucket_noise)
    df["scale_condition"] = np.where(df["extra_scale"] != 1.0, "scale_drift", "no_scale_drift")
    df["rotation_condition"] = np.where(df["rotation_deg"] != 0.0, "rotation_drift", "no_rotation_drift")
    df["boundary_condition"] = np.where(
        df["crosses_mat_boundary"] | df["crosses_strip_boundary"], "boundary", "non_boundary"
    )
    df["periodicity_level"] = df["periodicity_score"].apply(_bucket_score)
    df["uniqueness_level"] = df["uniqueness_score"].apply(_bucket_score)
    return df


def breakdown(df: pd.DataFrame, by: str) -> dict:
    """Per-group summary(), keyed by the distinct values of column `by`."""
    return {str(group): summarize(sub) for group, sub in df.groupby(by)}


def full_report(df: pd.DataFrame) -> dict:
    """Overall summary plus every breakdown required by the brief
    (section 6): family, noise, scale, rotation, boundary, periodicity,
    uniqueness."""
    df = add_breakdown_columns(df)
    return {
        "overall": summarize(df),
        "by_structural_family": breakdown(df, "structural_family"),
        "by_noise_level": breakdown(df, "noise_level"),
        "by_scale_condition": breakdown(df, "scale_condition"),
        "by_rotation_condition": breakdown(df, "rotation_condition"),
        "by_boundary_condition": breakdown(df, "boundary_condition"),
        "by_periodicity_level": breakdown(df, "periodicity_level"),
        "by_uniqueness_level": breakdown(df, "uniqueness_level"),
    }


def _case_rationale(row: pd.Series) -> str:
    boundary_bits = []
    if bool(row.get("crosses_mat_boundary", False)):
        boundary_bits.append("crosses a mat boundary")
    if bool(row.get("crosses_strip_boundary", False)):
        boundary_bits.append("crosses a strip")
    boundary_desc = " and ".join(boundary_bits) if boundary_bits else "stays deep inside one mat (no boundary)"
    return (
        f"Structural family `{row['structural_family']}` (split: `{row['split']}`); this crop "
        f"{boundary_desc}. Predicted error: {row['error_px']:.2f}px."
    )


def select_failure_cases(df: pd.DataFrame) -> dict:
    """Deterministically pick one representative successful, difficult, and
    catastrophic case from a results DataFrame - selected once, here, by
    this function, not re-randomized by the Streamlit app on every page
    load (app/app.py reads the file this writes,
    outputs/reports/failure_analysis_cases.json)."""
    ordered = df.sort_values(["error_px", "pair_id"]).reset_index(drop=True)
    successful = ordered.iloc[0]
    catastrophic = ordered.iloc[-1]
    mid_band = ordered[(ordered["error_px"] >= 10) & (ordered["error_px"] <= 60)]
    difficult = mid_band.iloc[0] if len(mid_band) else ordered.iloc[len(ordered) // 2]

    def _case(row: pd.Series) -> dict:
        return {
            "pair_id": row["pair_id"], "split": row["split"], "structural_family": row["structural_family"],
            "error_px": float(row["error_px"]), "rationale": _case_rationale(row),
        }

    return {"successful": _case(successful), "difficult": _case(difficult), "catastrophic": _case(catastrophic)}
