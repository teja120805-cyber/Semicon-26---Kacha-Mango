# experiments/pitch_aware_prominence — REJECT (near-miss on production seed, FAILED second-seed generalization check)

## Summary

**Verdict: REJECT.** On the production-seed frozen benchmark, this looked like the strongest
result of the entire campaign - a genuine, honestly-traced net-positive change (pooled
accuracy@5px 0.7436 -> 0.7500, `held_out` 0.650 -> 0.675 strictly improving, 6 of 7 gate criteria
passing). Per this report's own recommendation, and mirroring the project's
`multiway_tiebreak_v1` precedent of validating a near-miss on a second independently-seeded
dataset before recommending integration, that validation was run (section 6 below) - **and it
failed decisively**: on a fresh dataset generated from the same family table with an independent
seed (618234), the mechanism produced **zero rescues and four real regressions**
(0.13-0.37px correct matches breaking to 42-812px errors), dropping pooled accuracy@5px from
0.6618 to 0.6324 (-2.94pp) on that draw. The production-seed improvement does not generalize -
it was the mechanism getting lucky on which specific pairs its side effects landed on, not a
reliable fix. This is exactly the failure mode the project's second-seed validation discipline
exists to catch, and it caught it. Production is untouched; this is a full REJECT, not a
follow-up item for the user's review queue.

## 1. The idea - a surgical follow-up to `prominence_rerank`'s own failure analysis

`prominence_rerank/REPORT.md` found a real signal (peak prominence relative to a generic local
annulus) that was net HARMFUL: it broke 14-15 pairs for every 4 it rescued at full benchmark
scale, because a fixed 12px search radius penalized any candidate with *any* nearby competing
peak - including ordinary correct matches on non-periodic families that happened to have some
incidental nearby structure. Its own "what this doesn't rule out" section proposed the fix tested
here: require the competing peak to be at a spacing consistent with a *specific, measured*
periodic pitch for that pair, not any nearby peak at all.

`pitch_aware.py::detect_pitch` self-correlates each candidate's own template (at its own winning
scale/rotation hypothesis) and looks for a genuine trough-then-peak oscillation along the row and
column axes independently (DRAM word/bit lines are axis-aligned by construction -
`generator/pattern_renderer.py`). **A first attempt using only an absolute peak-height threshold
(0.3) fired on every family tested, including ones the dataset's own family descriptions call
non-periodic** (`dev_strip_anchor`) - a decay-tail ripple in the autocorrelation profile crossed
the naive threshold. Direct side-by-side inspection of the actual radial profiles showed the
real discriminator is AMPLITUDE (peak height minus the preceding trough), not absolute height:
genuinely periodic pairs showed trough~0.22-0.24, peak~0.92-0.96 (amplitude ~0.70); the
non-periodic pair's tail ripple was trough~0.25, "peak"~0.42 (amplitude ~0.17). Requiring both
`peak >= 0.5` AND `amplitude >= 0.25` correctly separates them (verified directly across all 24
development pairs before running any sweep - see `pitch_aware.py`'s docstring). If NO axis shows
genuine periodicity, prominence is exactly 0.0 and the ranking is untouched - this is what
prevents `prominence_rerank`'s false-positive mechanism from recurring here.

For candidates where periodicity IS detected, the candidate is only checked for a competing peak
at exact integer multiples (1x, 2x) of the *measured* pitch, within a small registration
tolerance (2px) - a much narrower, better-targeted check than a generic radius scan.

## 2. Implementation

`pitch_aware.py::compute_pitch_aware_prominence` - builds the candidate's own template, detects
its pitch, and (only if periodicity was detected) recomputes the full correlation surface once to
sample scores at +/-1x and +/-2x the measured pitch. `rank_pitch_aware_prominence` re-scores the
top-K classical candidates by `score + gamma * prominence` (gamma=0.0 provably identical to
`rank_classical`, confirmed directly). `harness.py::localize_pitch_aware` is structurally
identical to `pipeline.localize.localize`, swapping only the ranking step.

## 3. Evaluation

### Dev sweep (n=24, gamma ∈ {0, 1, 3, 5, 8, 12} × top_k ∈ {4, 8})

| gamma | top_k | acc@5px | mean_error_px |
|---:|---:|---:|---:|
| 0.0 | 4 | 0.583 | 111.50 |
| 1.0 | 4 / 8 | 0.583 | **110.98** |
| 3.0 | 4 / 8 | 0.583 | 110.98 |
| 5.0 | 4 | 0.583 | 142.48 |
| 5.0 | 8 | 0.542 | 144.11 |
| 8.0 | 4 | 0.583 | 143.65 |
| 8.0 | 8 | 0.500 | 178.23 |
| 12.0 | 4 | 0.583 | 130.62 |
| 12.0 | 8 | 0.500 | 174.96 |

Unlike `prominence_rerank`, this version never drops development accuracy at `top_k=4` for any
tested gamma - only `top_k=8` configs degrade (more candidates eligible for re-scoring, more
opportunity to disturb an already-correct top-4 pick that a stricter pitch-specific check would
otherwise leave alone). The dev-only selection procedure picked `gamma=1.0, top_k=4` (tied
accuracy, best - lowest - mean error among the tied group).

### Full frozen benchmark at the dev-selected config

| Split | Baseline acc@5px | Candidate acc@5px | Gate criterion |
|---|---:|---:|---|
| development | 0.583 | 0.583 | (not gated) |
| validation | 0.900 | 0.900 | tie - **fails criterion 1** (must strictly improve) |
| held_out | 0.650 | **0.675** | **passes criterion 2** |
| challenge | 0.750 | 0.750 | tie (not required to improve) |
| cross_generator | 0.800 | 0.800 | **passes criterion 3** (improve-or-tie) |
| **Pooled (n=156)** | **0.7436** | **0.7500** | |

```
"criteria": {
  "1_improves_validation": false,
  "2_improves_held_out": true,
  "3_improves_or_ties_cross_generator": true,
  "4_no_catastrophic_increase": true,
  "5_no_per_family_regression": true,
  "6_acceptable_runtime": true,
  "7_stable_across_seeds": true
}
```
6 of 7 criteria pass; only criterion 1 (`validation` must strictly improve, not tie) blocks a
full gate pass. Runtime: 3.27s/pair mean vs. baseline 3.18s/pair (1.03x, negligible). Note on
criterion 4: the aggregate catastrophic-rate check passing masks one real per-pair regression -
see the full per-pair accounting in section 4 below.

## 4. Full honest accounting - every single pair that changed, not just the net number

Only **6/156 pairs** had any coordinate change at all (diff > 0.01px) - this is a narrow,
low-blast-radius change, not a broad re-ranking overhaul:

| Pair | Split | Baseline error | Candidate error | Effect |
|---|---|---:|---:|---|
| `ho_heavy_noise_004` | held_out | 95.58px | **0.09px** | Clean rescue (crosses 5px line) |
| `ch_combined_acquisition_000` | challenge | 79.09px | **0.12px** | Clean rescue (crosses 5px line) |
| `ch_combined_acquisition_005` | challenge | 0.11px | 76.89px | Clean break (crosses 5px line) |
| `ho_vignette_gamma_005` | held_out | 25.41px | 522.41px | Raw-error regression, no threshold cross (already failing before and after) |
| `dev_dense_periodic_002` | development | 8.40px | 50.26px | Raw-error regression, no threshold cross |
| `dev_dense_periodic_005` | development | 573.56px | 519.24px | Raw-error improvement, no threshold cross |

**Net accuracy@5px movement is +2 rescued / -1 broken = +1 pair pooled**, but the full picture is
more mixed than that headline number alone suggests:

- `held_out`'s catastrophic rate stayed EXACTLY unchanged (8/40 both ways) only because
  `ho_heavy_noise_004`'s rescue (catastrophic -> fine) happened to cancel out
  `ho_vignette_gamma_005`'s independent new catastrophic regression (25px, already failing, but
  now 522px - a 20x worse raw error, on a pair the gate's accuracy@5px metric was already
  scoring as a failure either way, so this real quality regression is invisible to the headline
  accuracy number and only shows up by reading the per-pair CSV directly).
- `challenge`'s net accuracy is an exact wash within the SAME structural family
  (`ch_combined_acquisition`: one pair rescued, a different pair in the same 8-pair family
  broken) - the family-level accuracy check correctly reports `regressed: false` (4/8 -> 4/8),
  but that aggregate stability is masking two individually significant, opposite-direction
  per-pair flips.
- The two `dev_dense_periodic` changes are both already-failing pairs getting a different
  (better or worse) large error - directionally inconsistent (one better, one worse) and neither
  crosses any threshold, so they don't affect any gate criterion, but they show the mechanism is
  not narrowly, surgically improving periodicity failures - it's perturbing several of them,
  sometimes for the better and sometimes for the worse.

## 5. Why the production-seed result looked promising but wasn't trustworthy

Before running any second-seed check, three things already argued for caution rather than
treating the production-seed result as ready to integrate:

1. **It fails on the same structural reason `finer_hypothesis_grid` did** (`validation` already
   near a 90% ceiling on only 40 pairs - one pair is 2.5pp).
2. **The effect size was tiny in absolute pair count** (6/156 touched, net +1) - well within the
   range a single dataset draw's noise could produce.
3. **The mechanism had a real, demonstrated capacity to make specific pairs substantially worse**
   even when the aggregate metric didn't move (the `ho_vignette_gamma_005` case) - exactly the
   kind of side effect this project's own `center_tiebreak_v2` history warns against accepting
   without independently-seeded confirmation.

All three turned out to be the right instincts.

## 6. Second-seed validation - the result that changes the verdict

Ran the disciplined, dev-selected config (`gamma=1.0, top_k=4`, untouched from the production-
seed tuning - no re-tuning against the new data) against a **freshly-generated dataset**: same
`generator.dataset_generator.FAMILIES` table (identical structural coverage:
`development`/`validation`/`held_out`/`challenge`, 136 pairs), but **seed 618234** instead of
production's 777001 - by construction (the generator's per-pair
`default_rng([seed, family_salt(split, family), pair_index])` scheme), zero RNG overlap with
production or any other experiment in this campaign. `cross_generator` isn't part of the
FAMILIES table and has no independent second generator available, so it's excluded from this
check - a real, honestly-flagged limitation, not a hidden one.

| Split | n | Baseline acc@5px | Candidate acc@5px |
|---|---:|---:|---:|
| development | 24 | 0.542 | **0.458** |
| validation | 40 | 0.875 | 0.875 |
| held_out | 40 | 0.625 | **0.575** |
| challenge | 32 | 0.531 | 0.531 |
| **Pooled (n=136)** | | **0.6618** | **0.6324** |

**Zero pairs rescued. Four pairs broken**, all clean, unambiguous regressions from
near-perfect matches to large errors:

| Pair | Split | Baseline error | Candidate error |
|---|---|---:|---:|
| `dev_single_mat_006` | development | 0.37px | 42.77px |
| `dev_dense_periodic_001` | development | 0.13px | 811.81px |
| `ho_rotation_drift_004` | held_out | 0.19px | 55.17px |
| `ho_rotation_drift_006` | held_out | 0.15px | 60.20px |

Pooled accuracy@5px on this fresh draw: **-2.94pp** (0.6618 -> 0.6324) - the opposite direction
of the production-seed result, and by a larger margin. Note the baseline numbers themselves also
differ from production (0.6618 vs. production's equivalent-splits ~0.77 pooled) simply because
this is a different random draw from the same family table - a useful reminder of how much
natural variance exists between seeds on a dataset this size, which is exactly why a single-seed
near-miss (even a gate-criteria-passing one) isn't trustworthy evidence on its own.

## 7. Conclusion

The production-seed result was real (not a bug, not fabricated - the mechanism genuinely does
what it's designed to do) but was fitting to which specific pairs happened to have a
periodic-pitch-competitor structure that this heuristic handles well vs. poorly, on that one
draw. On an independent draw, the same mechanism, same disciplined dev-selected hyperparameters,
produced a **clean net negative with zero offsetting rescues**. This is a full REJECT: the
mechanism should not be integrated, and its earlier gate-passing performance should not be read
as evidence in its favor without this second-seed result also being considered.

## Reproduce

```
cd experiments/pitch_aware_prominence
python run_experiment.py            # production-seed result (looked promising, 6/7 gate criteria)
python validate_second_seed.py      # second-seed generalization check (failed - the real verdict)
```

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_pitch_aware.csv`,
`outputs/pitch_aware_metrics.json`, `outputs/integration_gate_result.json`,
`outputs/second_seed_baseline.csv`, `outputs/second_seed_pitch_aware.csv`,
`outputs/second_seed_summary.json`.
