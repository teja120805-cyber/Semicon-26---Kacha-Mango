# experiments/reachability_verification — the campaign's headline number was wrong, and is now corrected

**Date:** 2026-08-17. **No production change.** This is a diagnostic: it selects nothing, tunes
nothing, and alters no configuration.

## Summary

`experiments/REACHABILITY_CAMPAIGN.md` claimed **74% of failures are unreachable** — ground truth
not within 5px of any pooled candidate — and that claim was published in the project README's Known
Limitations. It rested on **19 failures** across two tuning surfaces.

Re-measured on the frozen 156-pair benchmark, which carries **35 failures**:

| | claimed | measured |
|---|---:|---:|
| failures unreachable | 74% | **37.1%** (13 of 35) |
| failures reachable | 26% | **62.9%** (22 of 35) |
| pool recall | 0.750 | **0.917** |
| selector efficiency | ~0.93 | **0.846** |

**The claim was roughly double the truth.** It has been corrected in the README,
`REACHABILITY_CAMPAIGN.md` and `reports/PROJECT_STATUS.md`.

## Why the original was wrong

The two tuning surfaces were unrepresentative, and in a direction that was predictable in
hindsight:

| surface | unreachable share |
|---|---:|
| `development` (24 pairs) | 4/7 = 57% |
| `tune_degraded` (40 pairs, deliberately degraded) | 10/12 = 83% |
| **frozen benchmark (156 pairs)** | **13/35 = 37%** |

`tune_degraded` was built specifically to over-represent heavy noise, speckle, vignette and
combined drift — the families where candidate generation fails most. That is exactly what it was
designed to do, and it made it a poor basis for a population-level claim. `development` reproduces
at 57% here, so the original measurement was not a computational error; it was a sampling error.

The frozen figure is close to the project's **pre-existing 45% candidate-generation share** from
`experiments/oracle_ceiling_diagnostic/`. **That measurement was right and the campaign's was the
outlier.** The campaign explicitly noted its number was "sharper than" the 45% figure and treated
the discrepancy as a difference in surfaces rather than a warning sign. It should have been treated
as a warning sign.

## What the frozen benchmark actually shows

Of 35 failures:

- **13 (37%) unreachable** — true location never proposed. Out of scope for any re-scoring stage.
- **22 (63%) reachable** — the true location *is* in the pool, at **median rank 3** (min 2, max 45),
  losing to the winner by a **median of 0.029 ZNCC** (max 0.611).

Discovery is therefore largely solved: pool recall is **0.917**. The headroom is in **selection
among near-ties**, which is consistent with `oracle_ceiling_diagnostic`'s finding that every failure
is a near-tie within 0.05.

Per split, the unreachable share varies widely, which is itself a caution against small-sample
claims:

| split | n | failures | unreachable | share |
|---|---:|---:|---:|---:|
| `validation` | 40 | 3 | 0 | 0% |
| `challenge` | 32 | 9 | 2 | 22% |
| `cross_generator` | 20 | 4 | 1 | 25% |
| `held_out` | 40 | 12 | 6 | 50% |
| `development` | 24 | 7 | 4 | 57% |

## What changes, and what does not

**Unchanged:** every rejection in the campaign. Those are empirical results — six experiments,
hundreds of configurations, zero rescues — and none of them depended on this number.

**Changed:** the *explanation* offered for them, and the recommendation drawn from it. "Stop
building re-scorers, because 74% of failures are out of scope" was wrong on both the premise and
the conclusion. The corrected position is narrower and less comfortable:

> The near-tie is real, reachable, and small — median rank 3, median deficit 0.029 ZNCC. **Twelve
> independent attempts to break it have failed** (nine in `ACCURACY_90_CAMPAIGN.md`, three in this
> campaign). That is strong evidence the problem is hard, not evidence it is out of scope.

A thirteenth attempt is not obviously futile. It needs a genuinely different discriminative signal
rather than another reweighting of ZNCC, and a reasonable bar before any integration work: **show it
can separate a 0.029 ZNCC gap on the 22 reachable failures.**

## Method note for future work

The generated tuning surfaces (`tune_degraded`, `validate_fresh`) remain useful for *tuning* — they
close the documented blind spot that `development` has no degraded-acquisition family, and they
correctly caught `anisotropic_psf/` as a dev-only mirage. But they are **not** a valid basis for
population-level claims about failure composition, because their family mix is deliberately skewed.
Claims of that kind belong on the frozen benchmark, which is what it exists for.

## Reproduce

```
python -m experiments.reachability_verification.run
```

Outputs `outputs/frozen_reachability.csv` (per pair) and `frozen_reachability.json` (summary).
