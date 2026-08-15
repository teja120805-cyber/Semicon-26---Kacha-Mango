"""Orchestrates the Phase 3 accuracy forensics sweeps: for every factor in
sweep_config.SINGLE_FACTOR_SWEEPS and every combination in
sweep_config.INTERACTION_SWEEPS, generates controlled pairs and runs the
instrumented classical pipeline over them, writing per-pair diagnostics and
per-factor/per-level summaries under experiments/accuracy_forensics/outputs/.

One variable changes at a time within a single-factor sweep (see
sweep_config module docstring for why this is a paired design); interaction
sweeps vary exactly the two (or three) named factors together in a small
factorial grid, never more.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import pandas as pd

from evaluation import metrics as eval_metrics
from experiments.accuracy_forensics import sweep_config as cfg
from experiments.accuracy_forensics.harness import generate_and_localize, make_family

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

PRIMARY_FACTORS = [
    "rotation_deg", "extra_scale", "periodicity_preset", "boundary_condition",
    "search_dose", "raster_drift_shear", "row_jitter",
]
SECONDARY_FACTORS = [k for k in cfg.SINGLE_FACTOR_SWEEPS if k not in PRIMARY_FACTORS]


def run_single_factor(factor: str, verbose: bool = True) -> list[dict]:
    spec = cfg.SINGLE_FACTOR_SWEEPS[factor]
    n = spec["n"]
    records = []
    for level in spec["levels"]:
        family = make_family(f"{factor}__{level['label']}", level["crop_mode"], level["overrides"])
        for pair_index in range(n):
            rec = generate_and_localize(pair_index, cfg.FORENSICS_SEED, family)
            rec["factor"] = factor
            rec["level"] = level["label"]
            records.append(rec)
        if verbose:
            print(f"  [{factor}] level={level['label']:24s} n={n} done")
    return records


def run_interaction(name: str, verbose: bool = True) -> list[dict]:
    spec = cfg.INTERACTION_SWEEPS[name]
    n = spec["n"]
    grid_keys = list(spec["grid"].keys())
    grid_values = list(spec["grid"].values())
    records = []
    for crop_mode in spec["crop_modes"]:
        for combo in itertools.product(*grid_values):
            overrides = dict(zip(grid_keys, combo))
            level_label = "crop=" + crop_mode + "," + ",".join(f"{k}={v}" for k, v in overrides.items())
            family = make_family(f"{name}__{level_label}", crop_mode, overrides)
            for pair_index in range(n):
                rec = generate_and_localize(pair_index, cfg.FORENSICS_SEED, family)
                rec["interaction"] = name
                rec["cell"] = level_label
                for k, v in overrides.items():
                    rec[f"grid_{k}"] = v
                rec["grid_crop_mode"] = crop_mode
                records.append(rec)
            if verbose:
                print(f"  [{name}] cell={level_label:48s} n={n} done")
    return records


def summarize_records(records: list[dict], group_cols: list[str]) -> dict:
    df = pd.DataFrame(records)
    out = {}
    for group_vals, sub in df.groupby(group_cols):
        key = group_vals if isinstance(group_vals, str) else "|".join(str(v) for v in group_vals)
        s = eval_metrics.summarize(sub)
        s["failure_location_counts"] = sub["failure_location"].value_counts().to_dict()
        out[key] = s
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 accuracy forensics sweeps.")
    parser.add_argument("--group", choices=["primary", "secondary", "interactions", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    all_single_records: list[dict] = []
    all_interaction_records: list[dict] = []

    factors_to_run = []
    if args.group in ("primary", "all"):
        factors_to_run += PRIMARY_FACTORS
    if args.group in ("secondary", "all"):
        factors_to_run += SECONDARY_FACTORS

    for factor in factors_to_run:
        print(f"=== single-factor sweep: {factor} ===")
        recs = run_single_factor(factor)
        all_single_records.extend(recs)
        with open(os.path.join(OUT_DIR, f"single_factor_{factor}.json"), "w") as f:
            json.dump(recs, f, indent=2, default=str)

    if args.group in ("interactions", "all"):
        for name in cfg.INTERACTION_SWEEPS:
            print(f"=== interaction sweep: {name} ===")
            recs = run_interaction(name)
            all_interaction_records.extend(recs)
            with open(os.path.join(OUT_DIR, f"interaction_{name}.json"), "w") as f:
                json.dump(recs, f, indent=2, default=str)

    if all_single_records:
        df = pd.DataFrame(all_single_records)
        df.to_csv(os.path.join(OUT_DIR, f"single_factor_per_pair_{args.group}.csv"), index=False)
        summary = summarize_records(all_single_records, ["factor", "level"])
        with open(os.path.join(OUT_DIR, f"single_factor_summary_{args.group}.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    if all_interaction_records:
        df = pd.DataFrame(all_interaction_records)
        df.to_csv(os.path.join(OUT_DIR, "interaction_per_pair.csv"), index=False)
        summary = summarize_records(all_interaction_records, ["interaction", "cell"])
        with open(os.path.join(OUT_DIR, "interaction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    elapsed = time.perf_counter() - t0
    total_pairs = len(all_single_records) + len(all_interaction_records)
    print(f"\nDone. {total_pairs} pairs evaluated in {elapsed:.1f}s ({elapsed / max(total_pairs, 1):.3f}s/pair).")


if __name__ == "__main__":
    main()
