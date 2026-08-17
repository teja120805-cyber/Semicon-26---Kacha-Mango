# experiments/hypothesis_ensemble — the winner's-curse hypothesis is REFUTED, and the inversion is the finding

**Date:** 2026-08-17. Diagnostic only; no production change. **Nothing here is a candidate for
integration** — including the inverted signal in §3, for reasons stated there.

## 1. The hypothesis, stated before running

`candidate_generation.deduplicate_by_location` keeps the **highest-scoring** detection at each
location and discards the rest, so the score *distribution* across the ~160 hypotheses that detect a
location is thrown away before ranking.

The prediction: the arg-max over many correlated noisy scores suffers **winner's curse**, biased
toward whichever location drew a lucky hypothesis. A decoy aligned with one particular
(scale, rotation) wins on its single best score; the true location should be high across *many*
hypotheses. So a **robust aggregate** — median or trimmed mean — should be the less biased estimator
and should favour truth.

Distinct from prior work: `cross_hypothesis_consensus_rerank/` re-ranked by *how many* hypotheses
found a location (a **count**, measured as a clean no-op); this uses the **magnitude distribution**.

## 2. Result — the prediction is refuted, and inverted

On the 22 reachable failures:

| aggregate | median margin | prefers truth |
|---|---:|---:|
| `max` *(production's estimator)* | −0.0293 | 0/22 (0%) |
| `mean` | −0.0835 | **5/22 (23%)** |
| `median` | −0.0806 | **5/22 (23%)** |
| `trimmed` (drop weakest 20%) | −0.1087 | **3/22 (14%)** |
| `top5` (size-independent) | −0.0478 | **6/22 (27%)** |
| `count` | −1.000 | 8/22 (36%) |

**Every robust aggregate is worse than the max**, and `trimmed` — the most robust — is the worst of
all. The prediction was not merely unsupported; it is backwards.

## 3. Why, and what it closes

The cluster sizes give it away. **The decoy is detected by more hypotheses than the truth** (median 4
vs 3; decoy has more on 12/22 pairs), and its score distribution is higher throughout, not just at
the max.

The mechanism is clear in hindsight and matters beyond this experiment:

> **A lattice-aligned decoy is self-similar under small scale and rotation perturbations, so it
> matches well across a whole neighbourhood of hypotheses. The true location depends on aperiodic
> boundary content, which only matches at very nearly the correct hypothesis.** On periodic
> structure, robustness across hypotheses is therefore a **decoy signature, not a truth signature.**

That closes the entire consensus / voting / robust-aggregation family with a mechanism rather than an
empirical null, and it retroactively explains two existing results:
`cross_hypothesis_consensus_rerank`'s no-op, and `prominence_rerank`'s net harm — both were, in
effect, rewarding robustness.

It also explains the one thing that *does* help. `finer_hypothesis_grid` (integrated) and
`gated_escalation` both work by **adding hypotheses**: if the true location only matches at nearly
the exact hypothesis, a denser grid is simply more likely to contain it. Same mechanism seen from
the other side, and consistent with `axis_decomposition/`'s finding that a 2% scale-quantisation
error accumulates past one full lattice period.

## 4. The inverted signal — reported, and explicitly NOT recommended

Inverting the sign clears the pre-registered bar:

| aggregate | inverted preference | sign-test p |
|---|---:|---:|
| `trimmed` | **19/22 (86%)** | 0.0004 |
| `mean` | 17/22 (77%) | 0.0085 |
| `median` | 17/22 (77%) | 0.0085 |
| `top5` | 16/22 (73%) | 0.026 |
| `count` | 14/22 (64%) | 0.14 |

**This must not be treated as a result, for three specific reasons:**

1. **It is a post-hoc sign flip.** The direction was pre-registered and came out reversed. Reversing
   it after the fact and quoting p = 0.0004 is exactly the error the pre-registration existed to
   prevent. The honest reading of this table is "the forward hypothesis is refuted", not "the
   inverse is confirmed".
2. **The closest prior experiment to it was net-harmful.** `prominence_rerank/` penalised candidates
   with nearby competing peaks — closely related to penalising a robust score distribution — and was
   *"real signal, net HARMFUL when active"* (0.7436 → 0.6795 in its exploratory configuration).
3. **It is measured on failures only.** All 22 pairs here are failures. It says nothing about the
   121 currently-correct pairs, and `prominence_rerank`'s harm came precisely from false positives
   on ordinary correct matches. A signal that separates failures can still destroy successes.

This is also the exact shape of `pitch_aware_prominence` (#8 in `ACCURACY_90_CAMPAIGN.md`): a
symmetric statistic that looked strong on one draw and failed second-seed validation, with #9 then
showing the effect was an artifact of the bonus direction.

**If anyone does pursue it**, the minimum bar is: a penalty-only formulation (never a bonus, per #9's
lesson), screened against *correct* pairs as well as failures, and validated on at least two
production-family seeds before any frozen run.

## 5. Status

- **Forward hypothesis: REFUTED.** Robust aggregation is worse than the max at every setting.
- **Mechanistic finding: robustness across hypotheses marks the decoy, not the truth.** This is the
  transferable output, and it closes the consensus/voting family with a reason.
- **Nothing integrable**, including the inversion.

## Reproduce

```
python -m experiments.hypothesis_ensemble.diagnose
```
