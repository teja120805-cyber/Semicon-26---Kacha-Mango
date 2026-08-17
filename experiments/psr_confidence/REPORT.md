# experiments/psr_confidence — P4: PSR rejected twice; the ambiguity threshold recalibrated and validated

**Date:** 2026-08-16. **Frozen benchmark: 0 runs.** Accuracy is unchanged by everything here —
§3 is a calibration result, and every prediction is bit-identical to production.

## Summary

Three results, two negative and one positive:

1. **PSR as the dual-arm PSF selector — REJECT.** It changes the arm on 15/24 and 23/40 pairs and
   picks *worse*: 0.708 → 0.583 on `development` (net −3), 0.700 → 0.650 on `tune_degraded`
   (net −2). The integrated decisiveness-gap rule is better on both surfaces.
2. **PSR as the confidence statistic — REJECT, by a wide margin.**
   `reports/RESEARCH_SURVEY_SCORING.md` §P4 predicted PSR "should do the same job better" than the
   existing pool gap. It does it far worse: **AUC 0.577 and 0.765 against the gap's 0.964 and
   0.941.** This is a clean refutation of a specific survey prediction.
3. **`AMBIGUITY_THRESHOLD` recalibrated — POSITIVE, and validated on held-back data.** The
   documented miscalibration is real and reproduces independently, but the diagnosis in
   `PROJECT_STATUS.md` is incomplete: the *statistic* is good (AUC 0.949/0.933), it is the
   *constant* that is wrong. Moving 0.92 → **0.990** more than doubles flag precision,
   0.324 → 0.750, on a surface never used to fit it.

## 1. PSR as the selector (A)

`pipeline/localize.py` chooses between `psf_sigma = 0.0` and `1.6` by `_decisiveness` — the score
gap to the best candidate more than 10px away. PSR is the textbook alternative.

| surface | n | production (gap) | PSR selector | arm differs | rescued | broken | net |
|---|---:|---:|---:|---:|---:|---:|---:|
| `development` | 24 | **0.7083** | 0.5833 | 15 | 0 | 3 | **−3** |
| `tune_degraded` | 40 | **0.7000** | 0.6500 | 23 | 1 | 3 | **−2** |

Null control: the `decisiveness` arm of this harness reproduces production's 0.7083 on
`development` exactly — the same figure the production pipeline produces — so the comparison is
against real production behaviour, not a reimplementation of it.

Consistent in sign and magnitude on two independent surfaces. Not integrated, and not worth
revisiting: PSR disagrees with the gap on more than half of all pairs and is worse each time.

## 2. PSR as the confidence statistic (B)

Separating correct from wrong pairs:

| statistic | AUC (dev) | AUC (degraded) | median correct | median wrong | separation |
|---|---:|---:|---:|---:|---:|
| pool gap | **0.941** | **0.964** | 0.0146–0.0160 | 0.0029–0.0030 | **5.0–5.4×** |
| ambiguity_ratio | 0.933 | 0.949 | 0.981 | 0.996 | — |
| winner_score | 0.739 | 0.777 | 0.831–0.866 | 0.724–0.790 | 1.1–1.2× |
| **PSR** | 0.765 | **0.577** | 6.37–7.25 | 3.05–6.17 | 1.0–2.4× |

PSR is the *worst* usable statistic tested on the degraded surface — 0.577 is very close to chance.
The likely mechanism: PSR's sidelobe statistics are computed over a correlation surface that, on a
periodic DRAM array, is *itself* periodic. The "sidelobe" population is full of lattice repeats of
the peak, so σ_sidelobe is inflated exactly where discrimination is hardest. The measure assumes a
noise-like background, which is the same assumption §1 of the survey identifies as the reason ZNCC
is the wrong matched filter here. P4 inherits the flaw it was meant to fix.

## 3. Recalibrating the ambiguity threshold — the positive result

**The defect reproduces.** On `tune_degraded`, `AMBIGUITY_THRESHOLD = 0.92` fires on 37/40 pairs at
precision 0.324 — an independent reproduction of the documented "128/156 at 31%".

**The cause is the constant, not the statistic.** `ambiguity_ratio` spans 0.816–0.999 with a median
of 0.985, so a 0.92 cut sits far below the bulk of the distribution and flags nearly everything.
Meanwhile the statistic's AUC is 0.933–0.949, i.e. it ranks pairs well.

**Protocol.** Fit on `development` + `tune_degraded` pooled (n=64, 19 failures); test on
`validate_fresh` (n=40, 11 failures, seed 271828), read **once**, after the rule was fixed in code.
Selection rule, fixed before any test data was read: *maximise precision subject to failure recall
≥ 0.80* — recall is the safety property, precision the usability one.

**Fit curve (n=64).**

| threshold | flag rate | precision | failure recall | answered | acc on answered |
|---:|---:|---:|---:|---:|---:|
| 0.920 *(production)* | 0.906 | 0.328 | 1.000 | 0.094 | 1.000 |
| 0.980 | 0.688 | 0.432 | 1.000 | 0.312 | 1.000 |
| **0.984** | 0.562 | 0.528 | **1.000** | 0.438 | **1.000** |
| 0.988 | 0.406 | 0.692 | 0.947 | 0.594 | 0.974 |
| **0.990** *(chosen)* | 0.344 | **0.773** | 0.895 | 0.656 | 0.952 |
| 0.994 | 0.250 | 0.875 | 0.737 | 0.750 | 0.896 |
| 0.996 | 0.156 | 0.900 | 0.474 | 0.844 | 0.815 |

Smooth at 0.002 granularity — the chosen point is a knee, not a spike, so it is not balanced on a
cliff.

**Test on held-back `validate_fresh`, read once.**

| threshold | flag rate | precision | failure recall | answered | acc on answered |
|---|---:|---:|---:|---:|---:|
| 0.920 (production) | 0.850 | 0.324 | 1.000 | 0.150 | 1.000 |
| **0.990 (recalibrated)** | 0.300 | **0.750** | 0.818 | **0.700** | **0.929** |

Precision more than doubles and holds within 0.023 of its fitted value on data never used to fit
it. The cost is disclosed: failure recall falls from 1.000 to 0.818 — roughly one failure in five is
no longer flagged. At 0.92 the flag catches every failure but is nearly meaningless, since it fires
on 85–91% of all pairs.

**Consistency check.** `experiments/crop_uniqueness_ceiling/` §4 reported the pipeline can answer
69% of pairs at 92.5% accuracy. The recalibrated threshold reaches 70.0% at 92.9% on held-back
data — independently reproducing that operating point as a **single constant on a statistic already
in production**, rather than as a separate gating mechanism. Two different routes agreeing is
meaningful evidence the operating point is real.

**Two operating points worth offering, not one:**

- `0.990` — answers 70% at 92.9%. Best general-purpose setting.
- `0.984` — answers 43.8% at **100%** accuracy with **zero** missed failures on the fit surfaces.
  For a use case where a wrong coordinate is costlier than a deferral, this is the safer choice.
  Note it was *not* validated on the test surface, because only the 0.990 rule was fixed before the
  test read; quoting its fit numbers as validated would be exactly the error this protocol exists
  to prevent.

## 4. Recommendation

Change `pipeline/localize.py::AMBIGUITY_THRESHOLD` from `0.92` to `0.990`.

This is a **reporting-only** change: `ambiguous` is a returned flag, never consulted by
`localize()` to make a decision, so accuracy, coordinates and runtime are all bit-identical. That
makes the integration gate largely inapplicable — criteria 1–6 all measure prediction quality,
which by construction cannot move. It should be reviewed as a calibration fix on the evidence
above, not run through a gate designed for algorithm changes, and if adopted it should be recorded
in `reports/GATE_EXCEPTIONS.md` as an exception of a fourth, distinct kind: *not evaluable by the
gate* rather than *failing it*.

Not applied — `pipeline/` is not modified without approval.

## 5. Run-count disclosure

- Frozen 156-pair benchmark: **0 runs.**
- `development` (24), `tune_degraded` (40): 1 run each, both statistics computed in the same pass.
- `validate_fresh` (40): **1 run**, read once after the threshold rule was fixed in code.
- Threshold sweep: 51 thresholds, but all on the fit surfaces only, and the selection rule was
  committed before the test read.

## Reproduce

```
python -m experiments.psr_confidence.run --surface development
python -m experiments.psr_confidence.run --surface tune_degraded
python -m experiments.psr_confidence.run --surface validate_fresh
python -m experiments.psr_confidence.calibrate
```
