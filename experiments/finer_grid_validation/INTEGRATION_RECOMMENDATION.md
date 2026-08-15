# Integration Recommendation: finer_hypothesis_grid

# PASS — READY FOR PRODUCTION INTEGRATION

**Production has NOT been modified. This is a recommendation only, awaiting explicit approval
before any change is made.**

## Summary

A validation set deliberately constructed to contain the exact conditions `finer_hypothesis_grid`
targets (11 categories, A-K, 12 pairs each, 132 pairs), generated on two independent seeds, shows a
dramatically larger and cleaner effect than the general-purpose benchmarks used previously — and,
critically, the improvement is **mechanistically confined to exactly the families it should affect**,
with **zero effect, positive or negative, on every family it shouldn't touch**.

## Results

| | Dataset A (seed 651900) | Dataset B (seed 782411) |
|---|---:|---:|
| n | 132 | 132 |
| Acc@5px: baseline -> fine | 52.3% -> **64.4%** (+12.1pp) | 54.5% -> **59.1%** (+4.6pp) |
| Mean error: baseline -> fine | 151.8px -> **112.1px** | 128.1px -> **101.2px** |
| Max error | 898.6px (unchanged) | 1194.4px (unchanged) |
| >50px: baseline -> fine | 37.1% -> **26.5%** | 31.8% -> **26.5%** |
| Candidate recall@5px: baseline -> fine | 71.2% -> **78.8%** | 72.7% -> **78.8%** |
| Rescue / Break / Net | 16 / 0 / **+16** | 7 / 1 / **+6** |
| Runtime ratio | 3.16x (clean, interleaved) | ~3.1x (consistent) |

Full metric set (@1/2/3/5/10px, median, P90, P95, candidate recall @1/2/5/10/20px) in
`outputs/validation_results.json`.

## Per-family mechanistic confirmation (the key evidence)

| Family | Involves rotation/scale? | Dataset A delta | Dataset B delta |
|---|---|---:|---:|
| A_no_boundary | No | +0.0pp | +0.0pp |
| B_high_periodicity | No | +0.0pp | +0.0pp |
| C_low_periodicity | No | +0.0pp | +0.0pp |
| G_boundary_crossing | No | +0.0pp (100%->100%) | +0.0pp (100%->100%) |
| H_noise_degradation | No | +0.0pp | +0.0pp |
| D_rotation | **Yes** | +8.3pp | +16.7pp |
| E_scale | **Yes** | +16.7pp | +0.0pp |
| F_rotation_scale | **Yes** | +41.7pp | +0.0pp |
| I_boundary_rotation_scale | **Yes** | +41.7pp | +16.7pp |
| J_periodicity_rotation_scale | **Yes** | +8.3pp | +0.0pp |
| K_difficult_interaction | **Yes** | +16.7pp | +16.7pp |

**Every single family without rotation/scale drift shows exactly zero change, in both independently-
seeded datasets — including `B_high_periodicity`, the purest test of "does this fix periodicity
alone" (it does not, confirmed, exactly as `FORWARD_HYPOTHESIS.md` predicted before running
anything).** Every family with rotation/scale drift improves or stays flat — never regresses. This
is not a statistical coincidence; it is the predicted mechanism operating exactly as diagnosed.

Failure-level classification (`outputs/failure_level_changes_dataset_{A,B}.csv`) confirms the
mechanism directly: 20 of 23 total rescues across both datasets are labeled
`hypothesis_grid_misalignment_corrected` (GT was in the candidate pool under the coarse grid but
lost to a better-aligned wrong hypothesis; the finer grid gives GT's own hypothesis a closer match
and it wins instead). 3 rescues are `candidate_generation_recovered` (GT entered the pool only under
the finer grid). The single break (dataset B) is `new_hypothesis_created_a_competing_wrong_peak` — a
new, finer hypothesis found a spuriously better-scoring wrong location, the known double-edged-sword
risk already documented in the original `finer_hypothesis_grid/REPORT.md`.

## Gate evaluation (all 8 criteria)

| # | Criterion | Result |
|---|---|---|
| 1 | Accuracy improves meaningfully | **PASS** — +12.1pp (A), +4.6pp (B) |
| 2 | Reproduces on independent seed | **PASS** — both datasets improve, same mechanism, same per-family pattern |
| 3 | Net rescue positive | **PASS** — +16 (A), +6 (B) |
| 4 | Break count acceptably low | **PASS** — 0 (A), 1 (B), against 16-23 rescues |
| 5 | Mean error improves | **PASS** — both datasets, ~26-40px reduction |
| 6 | Catastrophic (>50px) rate does not worsen | **PASS** — improves in both (37.1%->26.5%, 31.8%->26.5%); max error unchanged in both, not worsened |
| 7 | No ground-truth/data leakage | **PASS** — confirmed in `FORWARD_HYPOTHESIS.md` (GT never enters the candidate-generation/ranking/refinement decision path) and by explicit signature checks (0 overlap between dataset A, dataset B, and every production/experiment dataset) |
| 8 | Runtime acceptable | **PASS** — ~3.1-3.2x, well under this project's established 5x integration-gate budget |

**All 8 criteria pass.**

## Important honesty note on "the gate"

This passes the **8-criterion targeted-validation gate specified for this campaign** — a
deliberately more diagnostic standard than the original project-wide integration gate
(`evaluation/benchmark.py`), which required literal improvement on the *general-purpose* `validation`
split. That split was independently diagnosed (`reports/ACCURACY_FORENSICS.md`,
`experiments/finer_hypothesis_grid/REPORT.md`) as ceiling-limited — ties, does not regress, on two
independent datasets — because its family composition doesn't contain much of the condition this fix
targets. This campaign's gate was designed *by explicit instruction* to test the hypothesis fairly
on data that does contain those conditions; it is not a case of loosening the bar to force a pass.
The original project-wide gate's numbers are unchanged and still on record.

## What this does NOT show

- Does not fix pure periodicity (`B_high_periodicity` unaffected in both datasets) — this remains
  open, per `reports/ACCURACY_IMPROVEMENT_PHASE.md`'s finding that classical scoring alternatives and
  a modest learned candidate generator both failed to move it.
- Does not reduce max error (900px-1194px range, unchanged) — the single worst pair in each dataset
  is not a grid-misalignment case.
- Introduces a small, real, non-zero break risk (1/132 on dataset B) from new hypotheses
  occasionally creating a competing wrong peak — bounded and outweighed by rescues in every test run
  so far, but not literally zero.

**Recommendation: integrate**, on the strength of this evidence, contingent on your explicit approval
(see the proposed diff below — Phase 7).
