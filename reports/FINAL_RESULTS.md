# Final Results

> **Current production accuracy: 77.6%@5px pooled (n=156), as of 2026-08-16.** Reached via
> PSF-matched dual-arm candidate generation (`reports/GATE_EXCEPTIONS.md` exception 3,
> `experiments/psf_gated_selection/REPORT.md`), which corrected a ~16x sharpness mismatch between
> the correlation template and the Search image. Any pooled-accuracy figure below that predates
> that change is historical.
>
> **This applies to the whole error distribution, not just pooled accuracy** (clarified
> 2026-08-17): the median, mean, P90, P95 and >10px/>50px failure rates quoted in "Frozen benchmark"
> below are also pre-change figures. Current production values are median **0.3222px**, mean
> **47.729px**, P90 **69.85px**, P95 **433.76px**, >10px **21.15%**, >50px **14.10%** (22/156).

Summary of the full audit-through-experiment pass over DriftSense V2. Detail lives in the other 12
of the 13 `reports/` documents and each `experiments/<name>/REPORT.md`; this ties them together.

## Frozen benchmark

`data/` (`generator_version=driftsensev2.1.0`, seed `777001`; development=24, validation=40,
held_out=40, challenge=32, cross_generator=20, n=156). Classical baseline as measured before the
2026-08-16 PSF-matched change: **68.6%@5px pooled**, median error 0.34px, mean 65.3px (bimodal —
usually near-exact, sometimes a full wrong-location miss), P90 226.7px, P95 498.8px, failure rate
>10px = 29.5%, >50px = 18.6%. *(Historical — every figure in this paragraph is superseded. Current
production: **77.6%@5px**, median **0.3222px**, mean **47.729px**, P90 **69.85px**, P95
**433.76px**, >10px **21.15%**, >50px **14.10%**.)* Full breakdown:
`reports/V2_BASELINE_REPORT.md`.

This number is a *correction* of an earlier run (67.3%@5px) that carried a real cross-split
instance-leakage bug in the dataset's RNG seeding — found in the Phase 2 audit
(`reports/DATASET_AUDIT.md`), fixed in `generator/dataset_generator.py`, and verified directly (0
duplicate canvas/crop/GT signatures across all 136 internal pairs, vs. significant overlap before).
The correction changed the numbers only slightly and changed no qualitative conclusion — expected,
since the bug inflated confidence in specific held-out numbers, not the shape of the result.

## What actually drives failure (`reports/ACCURACY_FORENSICS.md`)

Controlled single-factor and interaction sweeps (~3,100 pairs, independent of the frozen benchmark),
ranked by real, measured impact rather than assumption:

1. **Boundary presence is the strongest lever found anywhere.** It substantially rescues both other
   failure modes even in combination — the worst tested condition (misaligned rotation *and* scale)
   scores 40%@5px with a mat boundary in view, 0%@5px without one.
2. **Periodicity/aliasing is a stronger standalone bottleneck than rotation/scale drift**, and it
   fails at candidate generation — the classical search never proposes the true location — not
   ranking. 62-85% of these failures land within a quarter-pitch of an exact integer multiple of the
   mat's own word pitch, scaling with pitch density.
3. **Rotation/scale damage tracks distance to the nearest tested hypothesis** in
   `pipeline/candidate_generation.py`'s fixed grid, not drift magnitude — a clean sawtooth (grid-hit
   values recover to 65-70%@5px, grid-midpoint values collapse to 40-45%). Scale misalignment is
   consistently more damaging than rotation misalignment.
4. **Noise, raster drift, row jitter, and all 12 secondary degradation factors are minor** — none
   approaches the damage of the three findings above, alone or in every combination tested. The one
   partial exception, barrel/pincushion distortion, shows a real but much smaller magnitude-dependent
   effect, consistent with its already-documented uncorrected-ground-truth mechanism.

This directly overturned the working assumption (stated in the original mission brief) that
rotation/scale drift was the presumptive dominant failure mode — the evidence instead points to
periodicity and hypothesis-grid alignment as at-least-comparable, more mechanistically-understood
causes, with boundary presence dominating all of them. Applying the same instrumented diagnostics
directly to the frozen benchmark itself (not just the synthetic sweeps —
`scripts/decompose_baseline_failures.py`) confirms this on real data: across the current **35**
failures, **37% are discovery failures** (the true location is not within 5px of any pooled
candidate, so no re-scoring or re-ranking stage can reach them) and the remaining **63% are
selection failures** (the true location *is* in the pool but loses a near-tie). Failure
concentrates specifically where no boundary is present — 41.2% failure rate without one, **7.95%**
with one. *(Corrected 2026-08-17: this passage previously read "31 candidate generation / 11
ranking / 7 genuine ambiguity", which sums to 49 failures and belongs to the superseded 74.36% run;
and it quoted 3.4% as the boundary failure rate, which is in fact the boundary **catastrophic**
>50px rate, not failure@5px. The 41.2% non-boundary figure is correct.)* Visual analysis of the
10 worst failures (`outputs/visualizations/catastrophic_failures/` — not in the repository;
produced by `scripts/visualize_catastrophic_failures.py`, which needs
`outputs/reports/baseline_failure_decomposition.csv` from `scripts/decompose_baseline_failures.py`)
sharpens this further: **6 of 10
have high periodicity (`periodicity_score` >= 0.64, 5 of 10 at the maximum 1.0), and 4 of 10 have
zero rotation *and* scale drift at all** *(recomputed 2026-08-17; previously 9 of 10 and 7 of 10)* — the catastrophic tail
specifically is a periodicity story, not primarily a rotation/scale one, because a periodicity
failure can jump to a wrong repeat arbitrarily far away while a grid-misalignment failure is
typically a wrong-but-nearby location.

## Experiments run against the integration gate

| Experiment | Verdict | Why |
|---|---|---|
| `embedding_reranker_v1` (CNN embedding re-ranker, pre-existing) | **REJECT** | Failed every criterion, all 3 seeds — 72 triplets from 2 of 13+ families, textbook overfitting. |
| `wider_candidate_pool` (more peaks/hypothesis, tighter NMS) | **REJECT** — structural no-op | Bit-identical predictions to baseline on all 132 gate pairs; `rank_classical`'s pure arg-max selection can never be affected by pool width alone. Clarifies that fixing periodicity needs a re-ranking stage, not more candidates. |
| `finer_hypothesis_grid` (81 vs. 25 scale×rotation hypotheses) | **INTEGRATED (2026-08-15)** | A follow-up targeted validation campaign (`experiments/finer_grid_validation/`) confirmed net rescue +16/+6 on two independent datasets, 0-1 breaks, with every rotation/scale-affected family improving and every other family showing exactly zero change on both. Now production (`pipeline/candidate_generation.py`). See `reports/ACCURACY_IMPROVEMENT_PHASE.md`. |
| `periodicity` — gradient-domain scoring | **REJECT** | Net rescue -9 (1 rescued, 10 broken) — gradient/edge representation is *more* ambiguous between periodic repeats than raw intensity, not less. |
| `periodicity` — intensity+gradient ensemble scoring | **REJECT** | Net rescue -4, and fails the runtime budget (roughly doubles per-hypothesis cost). |
| `rotation_scale` — coarse-to-fine local refinement | **REJECT** | Net rescue +2, weaker than `finer_hypothesis_grid`'s +5, at similar (not meaningfully lower) runtime cost. No advantage over the already-tested alternative. |

**Update (2026-08-15): `finer_hypothesis_grid` has since been integrated** after a dedicated
follow-up validation campaign — see `reports/ACCURACY_IMPROVEMENT_PHASE.md` for the full evidence and
exact diff. `pipeline/ranking.py`'s default remains `rank_classical` (unchanged); only
`candidate_generation.py`'s default hypothesis grid changed (5x5 -> 9x9, same span).
`refinement.py`/`feature_extraction.py` remain completely unmodified. Every other experiment in the
table above is still correctly not integrated, per the mission's integration rule (every criterion,
not most). Two classical approaches targeting
periodicity directly (gradient/ensemble scoring) were tested and both made things worse, which is
itself a meaningful result: it rules out "try a different classical correlation representation" and
sharpens the case that periodicity needs either more image context or a properly-data-scaled learned
representation, not a cheaper scoring tweak.

## Generator

DRAM-only, by explicit decision (no FinFET). One real completeness gap found relative to the
reference/demo generator — a continuous "feature size scale" multiplier — implemented as
`feature_size_scale` (`generator/mat_generator.py`), verified to change rendered geometry and to
correctly update `periodicity_score`. Every other acquisition/noise/distortion mechanism the
reference generator has was already present in V2 (`reports/DEGRADATION_COVERAGE.md`).

## Streamlit application

Sidebar went from a single page-navigation control to a full parameter panel (Structure, SEM imaging
physics, Acquisition noise & drift, Distortion & polygon scaling, Noise, Die layout) — every control
a real generator parameter, each with a tooltip. Two new sections: **Generate Sample** (regenerates
live from the sidebar, runs the real pipeline, shows predicted-vs-ground-truth) and **Experiment
Results** (forensics findings + every experiment's gate verdict, in-app). Verified via Streamlit's
`AppTest` framework (no browser-automation tooling available in this environment): all 8 sections
load without exceptions, and both a slider and a dropdown were confirmed to change the generated
image bytes, not just re-render cosmetically. Real server boot also confirmed.

## Repository hygiene

Terminology cleanup (no prior-project comparison framing in README/app — design rationale presented
on its own terms in `reports/` and code comments). Root-level design
docs moved into `reports/`. Rejected-experiment checkpoints moved out of `model/checkpoints/`
(production) into `experiments/embedding_reranker_v1/checkpoints/`, with all path references and
script defaults updated to match. `experiments/*/data/` and `experiments/*/outputs/` gitignored
alongside the existing `data/`/`outputs/` conventions, so none of the ~3,100 forensics pairs or
experiment result CSVs bloat the repository. Test suite: 24 tests (`generator/test_gt_safety.py`,
10; `pipeline/test_ranking.py`, 14), all passing, covering GT correctness/safety, reproducibility,
metadata completeness, preset differentiation, finite-coordinate sanity and tie-break behaviour.
`generator/test_dataset_validation.py` defines no `test_*` functions — it exposes
`validate_split`/`validate_dataset` helpers invoked as a CLI step — but two latent edge-clamping
bugs were found and fixed in it along the way (in those helpers, not the generator).

## Dataset sufficiency (`reports/DATASET_AUDIT.md` section 5)

Sufficient for classical-pipeline evaluation (post RNG-fix). **Not sufficient for training a
learned component** — unchanged by the RNG fix, since that fixed scene diversity per pair, not pair
count. `development` (24 pairs, 3 of 15 structural families) is the direct, diagnosed cause of
`embedding_reranker_v1`'s 72-triplet overfitting. If a learned-model experiment is attempted, the
audit specifies a bounded expansion (~150-180 development pairs across ~12-15 families, ~450-540
triplets — 6-7x more, not "huge amounts of data") — not yet executed, since no learned-model
experiment has been attempted this round.

## What would be worth doing next, if this continues

1. `finer_hypothesis_grid` has since been revisited on a larger validation surface
   (`experiments/finer_grid_validation/`) and **integrated on 2026-08-15** — it was the most
   promising result of the experiments tested at the time this list was written, and it is now
   production. What remains open is the same question one level down: whether an even finer or
   adaptively-placed hypothesis grid buys anything beyond the current 11 x 9 = 99.
2. A properly-scoped, adequately-sized experiment targeting periodicity specifically: a re-ranking
   stage over a wider shortlist (the architecture shape `wider_candidate_pool` showed is necessary,
   combined with the training-data-scale fix `embedding_reranker_v1`'s rejection already diagnosed,
   and now further motivated by `periodicity`'s finding that classical scoring alternatives don't
   help). This should use the bounded `development` expansion from `reports/DATASET_AUDIT.md` section
   5 before training anything, not the current 24-pair split.
3. Neither is required — production remains a defensible, fully-understood classical baseline.
   Of the 42 experiment directories on disk, **5 reached production** (4 of them as documented gate
   exceptions, `reports/GATE_EXCEPTIONS.md`); the large majority were rejected against the gate and
   are recorded as rejections. A high rejection rate against a gate that was actually enforced — and
   4 exceptions that are logged rather than buried — is the honest, disciplined outcome here, not a
   claim that nothing shipped.
