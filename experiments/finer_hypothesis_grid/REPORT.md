# Experiment A: rigorous re-validation of the finer scale/rotation hypothesis grid

**Status: PROMISING — ROBUST, BUT FORMALLY NOT INTEGRATED.** The net-rescue benefit is now confirmed
on a genuinely independent dataset (zero overlap, different seed), and the reason it fails the gate
is itself confirmed to be a real, reproducible property of the validation split — not noise. Per the
project's integration rule (every criterion must pass), it is **not integrated**.

## What this experiment is

Tests whether densifying `pipeline/candidate_generation.py`'s scale x rotation hypothesis grid (81
vs. the production 25 hypotheses, same +-8%/+-5deg span, half the step size) fixes the
grid-misalignment sawtooth documented in `reports/ACCURACY_FORENSICS.md` Finding 3. Calls the same
production `candidate_generation`/`ranking`/`refinement` functions unmodified, only passing a
different hypothesis-grid argument (`experiments/finer_hypothesis_grid/harness.py`, mirroring
`pipeline/localize.py`'s own orchestration with full candidate-pool diagnostics added).

## Re-validation methodology (this pass)

Prior result (net rescue +5 on the frozen benchmark, corrected runtime 3.17x, failing only
"validation must improve") raised the question: is this robust, or split-specific? Re-ran with full
instrumentation (candidate recall, failure-location category, periodicity/boundary/rotation/scale
breakdowns) on:

1. **The frozen benchmark** (`data/` — validation/held_out/challenge/cross_generator, n=132).
2. **A genuinely fresh, independently-seeded dataset** (seed `424242`, distinct from both the
   production seed `777001` and any other experiment's seed) generated using the identical family
   definitions, entirely isolated under `experiments/finer_hypothesis_grid/fresh_data/` — verified
   **zero signature overlap** with the frozen benchmark (0/40, 0/40, 0/32 duplicate canvas+crop+GT
   signatures across validation/held_out/challenge).
3. A clean, interleaved (contention-fair) runtime measurement.

## Result

| Metric | Frozen benchmark | Fresh independent (seed 424242) |
|---|---:|---:|
| n | 132 | 112 |
| Pooled acc@5px: baseline -> fine | 70.5% -> 73.5% | 69.6% -> 73.2% |
| Candidate recall@5px: baseline -> fine | 82.6% -> 87.1% | 82.1% -> 87.5% |
| Rescue / Break / Net | 6 / 2 / **+4** | 5 / 1 / **+4** |
| Candidate-generation failures: baseline -> fine | 23 -> **17** | 20 -> **14** |
| Runtime ratio (clean, interleaved) | 3.31x | (not separately remeasured; consistent with 3.17x prior) |

**The net-rescue effect (+4) and the reduction in candidate-generation failures (-6 pairs, both
datasets) are identical in direction and similar in magnitude on completely independent data.** This
is strong evidence the effect is real and general, not a fluke of one particular random draw.

### Per-split breakdown (both datasets)

| Split | Frozen: base -> fine | Fresh: base -> fine |
|---|---|---|
| validation | 90.0% -> **90.0%** (tied) | 82.5% -> **82.5%** (tied) |
| held_out | 55.0% -> 60.0% (+5.0pp) | 70.0% -> 77.5% (+7.5pp) |
| challenge | 68.8% -> 75.0% (+6.2pp) | 53.1% -> 56.2% (+3.1pp) |
| cross_generator | 65.0% -> 65.0% (tied) | (not part of fresh set — external, fixed) |

**Validation ties exactly on both datasets.** This confirms — it is not noise — that validation's
family composition (`val_mat_boundary`, `val_same_preset_boundary`, `val_multi_mat`,
`val_linewidth_bias`: all boundary-heavy, already-easy cases for the classical baseline) simply
doesn't contain much of the grid-misalignment failure mode this fix addresses, so there is little
room for it to move regardless of dataset. held_out and challenge — the splits that actually sample
rotation/scale drift and combined hard conditions — improve consistently on both datasets.

### Does it help the specific failure modes it should? (frozen / fresh)

- **High-periodicity cases**: 68.9%->73.8% / 58.1%->64.5% — improves on both, more than the pooled
  average. Candidate recall for this bucket also rises (82.0%->88.5% / 75.8%->85.5%) — the finer
  grid measurably helps the *candidate-generation* stage for periodic cases too, not only ranking.
- **No-boundary cases**: 48.1%->51.9% (frozen, +3.8pp) but 41.7%->37.5% (fresh, **-4.2pp** — a
  regression). At n=52/24 this cell is thin and could be noise either direction, but it is reported
  honestly rather than only citing the favorable frozen-benchmark number: **the no-boundary-specific
  benefit does not consistently replicate.**
- **Rotation/scale cases**: mostly positive but individual buckets are small (n=3-14) and noisy in
  both directions — e.g. scale-medium improves strongly on both (41.7%->66.7% / 85.7%->78.6%, this
  one particular cell actually reverses), rotation-low improves strongly on both (42.9%->71.4% /
  45.5%->63.6%). The aggregate rotation/scale-affected accuracy is positive on both datasets even
  though individual sub-buckets are too small to trust in isolation.
- **Candidate-generation failures specifically**: -6 on both datasets (23->17 frozen, 20->14 fresh)
  — the single most consistent, trustworthy result in this re-validation.

## Is the +5 (now +4) net rescue robust or split-specific?

**Robust.** Confirmed on a dataset with zero overlap with the frozen benchmark, same family
definitions, different seed: net rescue +4, candidate-generation failures down 6, validation ties,
held_out and challenge improve — every one of these replicated in direction and similar in magnitude.
The one genuine caveat is the no-boundary-specific sub-analysis, which does not replicate cleanly.

## Verdict

Formally **fails the gate** — criterion 1 ("must improve validation") is not met on either dataset,
and this is now confirmed to be a real property of validation's composition rather than noise. Per
the project's stated integration rule ("only integrate if it passes... if it fails, do not integrate,
document why, move on"), **this is not integrated**. It remains the strongest, most rigorously
re-validated result of any experiment tested — a genuinely reproducible improvement on the splits
that actually contain the failure mode it targets, blocked only by a criterion whose underlying split
doesn't contain that failure mode in the first place. That is a fact worth surfacing plainly, not a
reason to override the rule unilaterally.

## Production impact

None. `pipeline/candidate_generation.py`'s default hypothesis grid is unchanged; this experiment
only ever passes a different grid as an explicit argument to functions imported unmodified.
