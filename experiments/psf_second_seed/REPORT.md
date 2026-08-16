# experiments/psf_second_seed — the +2.56pp gain REPLICATES, but do not integrate as-is

## Summary

Independent second-seed validation of `psf_matched_template`'s fixed `sigma_extra=1.6`, using
seed **618234** — the same second seed `pitch_aware_prominence` was validated against, so it
cannot have been chosen to flatter this candidate.

**The gain replicates, and is larger on the new seed:**

| | n | baseline | candidate | delta |
|---|---:|---:|---:|---:|
| production seed 777001 | 156 | 0.7436 | 0.7692 | **+0.0256** |
| **second seed 618234** | 136 | 0.6618 | 0.6985 | **+0.0368** |
| pooled over both | 292 | 0.7055 | 0.7363 | **+0.0308** |

On the second seed: rescued 8, broken 3, **every split improves or ties**, and **no family
regresses**. That is a materially better picture than the production seed produced.

This is the first result in this project to survive an independent-seed check. It is a real
effect, not a single-draw artifact.

**But it should still not be integrated as it stands**, for two reasons that only become visible
when both seeds are analysed together. See §3.

## 1. Method

The generator's per-pair RNG folds `(seed, split, family, pair_index)` together, so there is zero
cross-seed leakage from the production set. Both the unmodified production `pipeline.localize`
and the `sigma=1.6` candidate were run over the same regenerated 136 pairs.

Split composition note: `FAMILIES` covers development/validation/held_out/challenge = 136 pairs.
`cross_generator`'s 20 pairs come from a separate routine and are not regenerated, so this seed's
pooled figure is over n=136 and is **not** directly comparable to the 156-pair production number.
Only the baseline-vs-candidate delta within each seed is meaningful.

## 2. Second-seed results

| split | baseline | candidate | delta |
|---|---:|---:|---:|
| development | 0.542 | 0.542 | +0.000 |
| validation | 0.875 | 0.900 | +0.025 |
| held_out | 0.625 | 0.675 | +0.050 |
| challenge | 0.531 | 0.594 | +0.062 |

Notably, the four families that regressed on the production seed are all flat or better here:
`ch_worst_case` +0.125 (was −0.250), `ho_vignette_gamma` +0.000 (was −0.200),
`ch_barrel_charging` +0.000 (was −0.125), `ch_speckle_saltpepper` +0.000 (was −0.125).

Taken alone, that reads as "the production-seed regressions were small-sample noise." Combining
the seeds shows that reading is **wrong**.

## 3. Why this still should not be integrated

### 3a. The regressions are systematic, not noise

Rescued/broken counts summed across **both** seeds, per family:

| family | rescued | broken | net |
|---|---:|---:|---:|
| ho_vignette_gamma | 0 | 2 | **−2** |
| ch_barrel_charging | 1 | 2 | **−1** |
| ch_worst_case | 2 | 3 | **−1** |
| ch_speckle_saltpepper | 0 | 1 | **−1** |
| dev_strip_anchor / val_multi_mat / cross_generator | — | — | 0 |
| val_linewidth_bias / val_mat_boundary / val_same_preset_boundary / ho_scale_drift | 1 each | 0 | +1 each |
| ch_combined_acquisition / dev_single_mat / dev_dense_periodic / ho_heavy_noise / ho_rotation_drift | 2–3 each | 0–1 | +2 each |
| **total** | **21** | **12** | **+9** |

**Every net-negative family is a degraded-acquisition family, and not one of them is net-positive
on either seed.** The direction is perfectly consistent across two independent draws. The
production seed did not get unlucky — PSF-matching the template genuinely helps clean and
periodic-ambiguity cases and genuinely hurts vignette/gamma, barrel/charging, speckle/salt-pepper
and worst-case combinations. Averaging over more seeds will not make that go away.

### 3b. The net gain is not statistically significant

A two-sided sign test on the 33 discordant pairs across both seeds (21 rescued vs. 12 broken)
gives **p = 0.16**. With 292 pairs total this is the honest read: the *direction* replicates
convincingly, the *magnitude* does not yet clear conventional significance.

### 3c. The consequence

The net benefit depends on the benchmark's family mix. This benchmark is roughly 70% clean-optics
and 30% degraded-acquisition families. A production workload with a heavier degraded fraction
could see a net *negative*. Integrating a change whose sign depends on the workload composition,
on p = 0.16 evidence, is not justified — even under the documented gate-exception mechanism that
`scale_range_v1` used (`reports/GATE_EXCEPTIONS.md`).

## 4. What to do instead — and the evidence says it should work

Do not choose between "always blur 1.6" and "never blur". **Gate it.**

> **CORRECTION (added after this report was first written).** The threshold-gate proposal that
> stood here was **wrong**, and checking it before spending a benchmark run showed why. Tabulating
> each family's estimated sigma against its measured delta at σ=1.6:
>
> | | estimated sigma range |
> |---|---|
> | families that GAIN | 0.36 – 1.06 |
> | families that LOSE | 0.34 – 1.03 |
>
> The ranges overlap almost completely. `ch_combined_acquisition` (0.36) and `ho_heavy_noise`
> (0.37) are *gainers* with low estimates; `ch_barrel_charging` (0.71) and `ch_worst_case` (0.55)
> are *losers* with mid estimates. **No threshold on the spectral estimator separates them.** The
> claim that its ordering tracks benefit came from reading only the four harmed families and not
> checking the gainers against the same scale.
>
> What distinguishes the harmed families is not blur at all: they are the ones carrying
> *non-stationary, non-Gaussian* corruption — barrel warp, spatially-varying vignette/gamma,
> impulse (salt-and-pepper) noise. Pure stationary noise (`ho_heavy_noise`) gains normally.
>
> The replacement approach — per-pair self-selection between the two templates using the
> pool-internal gap statistic — is in `experiments/psf_gated_selection/`.

**Caveat to carry forward:** this would be the fifth frozen-benchmark configuration in this line.
Any threshold must be chosen from the *estimator distribution* (which is ground-truth-free and can
be computed on any data) rather than by sweeping accuracy on the benchmark, and the result must be
confirmed on both seeds before it is proposed for integration.

## Reproduce

```
cd experiments/psf_second_seed
python validate_second_seed.py   # regenerates 136 pairs at seed 618234, ~20 min
```

Outputs: `outputs/second_seed_baseline.csv`, `outputs/second_seed_psf.csv`,
`outputs/second_seed_summary.json`. The regenerated `validation_data/` (~200MB) is reproducible
from the seed and can be deleted.
