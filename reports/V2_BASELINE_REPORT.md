# DriftSense V2 — Baseline Report

> **Superseded 2026-08-15**: `pipeline/candidate_generation.py`'s default scale/rotation hypothesis
> grid was integrated into production (5x5 -> 9x9, same span, half the step size) after
> `reports/ACCURACY_IMPROVEMENT_PHASE.md`'s validation showed a robust, reproducible improvement.
> **Current production pooled accuracy is 77.6%@5px (n=156), as of 2026-08-16** (71.2% when
> this banner was first written; raised by the A2/A6 integrations and then by PSF-matched
> dual-arm candidate generation — `reports/GATE_EXCEPTIONS.md`) — this report's numbers
> below (68.6%@5px, the pre-integration 25-hypothesis classical baseline) are the historical analysis
> that motivated the change, kept for reference, not the current configuration. See
> `reports/ACCURACY_IMPROVEMENT_PHASE.md` for the integration evidence and current per-condition
> results.
>
> **The runtime and hypothesis-count figures below are superseded too, not just accuracy**
> (clarified 2026-08-17). This report's "0.51 s/pair, 25 scale x rotation hypotheses/pair" describes
> the pre-integration configuration. Production now evaluates an **11 scales x 9 rotations = 99**
> hypothesis grid per pair, in two template passbands, at **3.72 s/pair mean / 3.62 s/pair median**
> (machine-dependent — see README.md's "Runtime, hardware and timing method").
>
> **Regenerated 2026-08-15** (prior to the above) after the Phase 2/4 dataset audit found a real
> cross-split instance-leakage bug in the RNG seed derivation (see `reports/DATASET_AUDIT.md` section
> 2). The seeding was fixed (`generator/dataset_generator.py`, now `driftsensev2.1.0`) and the full
> dataset was regenerated from the same family/split definitions under the corrected scheme, then
> re-evaluated. The qualitative conclusions were unchanged (old pooled 67.3%@5px vs. corrected
> 68.6%@5px) — the leakage inflated confidence in specific held-out numbers, not the general shape of
> the failure pattern.

**Pipeline**: `pipeline/localize.py`, classical mode (multi-scale x multi-rotation ZNCC candidate
generation, spatial deduplication, subpixel parabolic refinement — no learned component).
**Dataset**: V2 generator, 136 pairs across `development`/`validation`/`held_out`/`challenge`, plus
20 external `cross_generator` pairs (see `reports/V2_ARCHITECTURE_PLAN.md` section 9). **Generator
version**: `driftsensev2.1.0`. **Seed**: `777001` (main splits), external seed for cross_generator.
Full per-pair results: `outputs/reports/per_pair_results.csv`. Plots: `outputs/plots/*.png`.

This report exists to answer one question honestly, per the brief: **does V2's harder, macro-
structured dataset actually make the localization problem more solvable than it looks on paper, or
does it reveal a weakness a simpler dataset was hiding?** The answer, below, is closer to the second:
a working classical baseline exists, but it is dominated by two failure modes — periodic-repeat
ambiguity and rotation/scale drift — and controlled forensics (`reports/ACCURACY_FORENSICS.md`) is
what actually establishes which of the two dominates, rather than asserting it from this correlational
breakdown alone.

## Pooled accuracy (all 156 pairs, all splits)

| Metric | Value |
|---|---|
| n | 156 |
| Accuracy@1px | 66.0% |
| Accuracy@2px | 68.6% |
| Accuracy@3px | 68.6% |
| Accuracy@4px | 68.6% |
| Accuracy@5px | 68.6% |
| Median error | 0.34 px |
| Mean error | 65.3 px |
| P90 error | 226.7 px |
| P95 error | 498.8 px |
| Max error | 900.8 px |
| Failure rate (>10px) | 29.5% |
| Failure rate (>50px) | 18.6% |
| Mean runtime/pair | 0.51 s (CPU, 25 scale x rotation hypotheses/pair) — *superseded: production is now 3.72 s/pair over 11 x 9 = 99 hypotheses* |

**Read this distribution as bimodal, not "69% good, 31% a little off"**: the huge gap between the
median (0.34px, essentially perfect) and the mean (65.3px) means most pairs are found almost exactly
right and a substantial minority are found *catastrophically* wrong (a wrong periodic repeat, often
hundreds of pixels away) — not mildly off.

## The dominant correlational finding: structure beats periodicity, drift beats noise

| Condition | n | Accuracy@5px | Median error | P90 error |
|---|---:|---:|---:|---:|
| Boundary-crossing (mat or strip) | 88 | **86.4%** | 0.25px | 26.5px |
| Non-boundary (deep in one mat) | 68 | **45.6%** | 12.37px | 484.3px |
| High uniqueness score | 74 | **85.1%** | 0.27px | 35.9px |
| Low uniqueness score | 68 | **45.6%** | 12.37px | 484.3px |
| No rotation drift | 130 | **73.8%** | 0.33px | 212.2px |
| Rotation drift present | 26 | **42.3%** | 14.98px | 245.4px |
| No scale drift | 130 | **73.1%** | 0.33px | 108.1px |
| Scale drift present | 26 | **46.2%** | 20.93px | 453.6px |

**Whether a crop touches real macro structure predicts success far better than any single noise or
acquisition condition does.** `uniqueness_score` (a pure function of boundary/mat-count geometry,
computed at generation time — see `generator/metadata.py`) tracks this almost exactly, because it's
measuring the same thing.

Periodicity score is a much weaker and *non-monotonic* predictor on this correlational breakdown:

| periodicity_level | n | Accuracy@5px |
|---|---:|---:|
| low | 41 | 75.6% |
| medium | 38 | 68.4% |
| high | 77 | 64.9% |
| **noise_level** | | |
| low_noise | 130 | 70.8% |
| medium_noise | 8 | 50.0% |
| high_noise | 18 | 61.1% |

`medium_noise` looks worse than `high_noise` above purely because it's dominated by the small,
deliberately-hard `ch_combined_acquisition` family (noise *and* rotation *and* scale together, n=8) —
these single-axis breakdowns can be confounded when a family combines several hard axes at once.
**This confound is exactly why `reports/ACCURACY_FORENSICS.md` exists**: it isolates one factor at a
time with dedicated controlled sweeps instead of relying on the family mix here.

## By structural family

| Family | Split | n | Accuracy@5px | Median error |
|---|---|---:|---:|---:|
| dev_strip_anchor | development | 8 | 100.0% | 0.20px |
| dev_single_mat | development | 8 | 50.0% | 6.16px |
| dev_dense_periodic | development | 8 | **25.0%** | 249.24px |
| val_mat_boundary | validation | 10 | 90.0% | 0.24px |
| val_same_preset_boundary | validation | 10 | 80.0% | 0.28px |
| val_multi_mat | validation | 10 | 100.0% | 0.22px |
| val_linewidth_bias | validation | 10 | 90.0% | 0.25px |
| ho_heavy_noise | held_out | 10 | 70.0% | 0.34px |
| ho_rotation_drift | held_out | 10 | **30.0%** | 21.39px |
| ho_scale_drift | held_out | 10 | 40.0% | 32.97px |
| ho_vignette_gamma | held_out | 10 | 80.0% | 0.27px |
| ch_combined_acquisition | challenge | 8 | 50.0% | 21.62px |
| ch_barrel_charging | challenge | 8 | 87.5% | 0.63px |
| ch_speckle_saltpepper | challenge | 8 | 87.5% | 0.23px |
| ch_worst_case | challenge | 8 | 50.0% | 5.29px |
| cross_generator_external | (external) | 20 | 65.0% | 0.94px |

**The single worst family in the entire benchmark is `dev_dense_periodic` (25.0%), not a
rotation/scale family** — and it has *zero* rotation or scale drift (`dev_single_mat`/`dev_strip_anchor`/
`dev_dense_periodic` are all rotation=0, scale=1.0 by family definition). It is deep inside the
densest-pitch preset (`mat_dense`) with no boundary in view: pure periodic-repeat ambiguity, no
acquisition drift involved at all. `ho_rotation_drift` (30.0%) is the second-worst. This is the
central reason this report does **not** assert "rotation/scale drift is the dominant failure mode" —
that claim would be premature on this evidence alone, and is exactly what
`reports/ACCURACY_FORENSICS.md`'s controlled, single-factor sweeps were built to test properly rather
than infer from a handful of 8-10-pair families.

## What this report does NOT claim

- It does not root-cause the rotation/scale-drift or periodicity findings to a specific pipeline
  mechanism — see `reports/ACCURACY_FORENSICS.md` for the controlled experiment that does.
- It does not tune any pipeline parameter (hypothesis grids, thresholds) based on these results before
  reporting them — this is the frozen classical baseline; changes after this point are a new,
  separately-reported experiment (`experiments/`), never a silent rewrite of this baseline.
- It does not treat `cross_generator_external`'s 65.0%@5px as a strong generalization claim — 20
  pairs, no rotation/scale-drift mechanism in that external generator, reported separately from V2's
  own splits by design (`reports/V2_ARCHITECTURE_PLAN.md` section 9).
