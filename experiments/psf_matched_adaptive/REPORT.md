# experiments/psf_matched_adaptive — REJECT (adaptive sigma does not beat the fixed sigma)

## Summary

**The adaptive idea failed.** Estimating the PSF-matching blur per pair from the image spectra
does *not* outperform the single global sigma from `psf_matched_template`, in either variant
tested. The unscaled estimator is a complete wash (pooled unchanged at 0.7436); the dev-scaled
variant reaches 0.7628, still below the fixed sigma's 0.7692, and introduces a **−0.400
catastrophic regression** on `ho_vignette_gamma`.

The failure has a specific, identifiable cause: **a Gaussian-blur model of the spectral difference
is confounded by both noise and gamma**, and it is wrong in opposite directions on the two family
groups that matter most.

Nothing here is integrated. Production remains at 0.7436.

## 1. The estimator

Convolving with a Gaussian multiplies the power spectrum by `exp(-4π²σ²f²)`, so

```
log P_template(f) − log P_search(f) = 4π²σ² f² + c
```

is a straight line in `f²`, and `σ = sqrt(slope / 4π²)` in closed form. Fitted over
f ∈ [0.06, 0.30] cycles/px, template spectrum vs. the average over a 5×5 grid of Search patches.

Deliberately **parameter-free** — band and clamp fixed a priori from sampling geometry — because
`development` contains no degraded-acquisition family, so any dev-tuned constant inherits that
blind spot (see `psf_matched_template/REPORT.md` §3).

Worth noting: no structural family overrides the blur parameters, so the true *optical* mismatch
really is ~1.0px for every pair. The estimator's actual job is therefore matching **usable
bandwidth**, not PSF — applying less blur where the Search image's high frequencies are
noise-dominated rather than signal.

## 2. Variant A — unscaled: protective, but forfeits the gains

| split | baseline | adaptive |
|---|---:|---:|
| development | 0.583 | 0.625 |
| validation | 0.900 | 0.925 |
| held_out | 0.650 | 0.600 |
| challenge | 0.750 | 0.750 |
| cross_generator | 0.800 | 0.800 |
| **pooled** | **0.7436** | **0.7436 (+0.0000)** |

Rescued 3, broken 3. Runtime 0.99x.

It **did** do the protective half of its job — every family that regressed under the global
σ=1.6 (`ch_worst_case`, `ch_barrel_charging`, `ch_speckle_saltpepper`, `ch_combined_acquisition`)
is now exactly +0.000. But it forfeited nearly all the gains: `dev_single_mat` fell from +0.250 to
+0.000, `dev_dense_periodic` from +0.250 to +0.125.

Cause: the Search image's noise floor raises its high-frequency power, flattening the fitted slope
and dragging the estimate down to a median of **0.645** — below even the physically-derived 1.0,
and far below the 1.6 that produced the gains. The estimator's *relative ordering* across families
is right; its *absolute scale* is systematically too low.

## 3. Variant B — dev-scaled: better, but breaks one family badly

Applied `σ = k · σ̂` with `k = 1.6 / 1.029 = 1.555`, both terms taken from `development` only
(1.6 from `psf_matched_template`'s dev sweep, 1.029 from variant A's dev median). No
frozen-benchmark result was used to set `k`.

| split | baseline | scaled adaptive |
|---|---:|---:|
| development | 0.583 | 0.708 |
| validation | 0.900 | **0.950** |
| held_out | 0.650 | **0.575** |
| challenge | 0.750 | 0.781 |
| cross_generator | 0.800 | 0.800 |
| **pooled** | **0.7436** | **0.7628 (+1.92pp)** |

Rescued 10, broken 7. Runtime 1.02x.

**The decisive failure:** `ho_vignette_gamma` collapses 0.800 → **0.400**. The estimator assigns
that family the *highest* sigma of any (1.60) — because a gamma nonlinearity reshapes the power
spectrum in a way a Gaussian-blur model reads as blur. So the estimator over-blurs precisely the
family that can least afford it. `held_out` also drops further than under variant A (0.575 vs
0.600), since `ho_vignette_gamma` lives there.

This is a clean falsification of the estimator's core assumption: the spectral difference between
template and Search is **not** purely a Gaussian blur. Noise inflates high-frequency power (biasing
σ down); gamma reshapes the whole curve (biasing σ up). The two confounds pull in opposite
directions on different families, which is why no single correction to the estimator fixes it.

## 4. Where this line of work now stands

| configuration | pooled | held_out | vignette_gamma | gate |
|---|---:|---:|---:|---|
| baseline | 0.7436 | 0.650 | 0.800 | — |
| fixed σ=1.0 (physical) | 0.7564 | 0.650 | 0.800 | fail 2,3,4,5 |
| **fixed σ=1.6 (dev-selected)** | **0.7692** | **0.675** | 0.600 | fail 4,5 |
| adaptive (unscaled) | 0.7436 | 0.600 | 0.700 | fail 2,4,5 |
| adaptive × 1.555 | 0.7628 | 0.575 | 0.400 | fail 2,4,5 |

**The fixed σ=1.6 remains the best configuration, and adaptivity did not improve on it.**

### A methodological warning that must be recorded

**Four full frozen-benchmark runs have now been made in this line of work.** Each individual
choice was defensible — dev-only selection, a pre-registered physical value, a parameter-free
estimator, a dev-derived scale — but four runs against a 156-pair benchmark is enough to select on
benchmark noise regardless of the discipline applied to each step. The ±2pp differences between
the top configurations are within what that many looks can manufacture.

**No configuration in this table should be integrated on the strength of these numbers alone.**
Before any integration decision, the leading candidate (fixed σ=1.6) must be re-validated on an
independently-seeded dataset, exactly as `pitch_aware_prominence` was — that check is what caught
a 6/7-gate-passing result there as a single-seed artifact.

## 5. What would actually be worth trying next

The estimator's failure is informative rather than fatal to the idea. The confounds are both
addressable:

- **Gamma:** estimate and invert the photometric nonlinearity before the spectral fit, or fit on a
  rank-normalized (histogram-equalized) version of both images, which is gamma-invariant.
- **Noise:** fit the Search spectrum as `signal · exp(-4π²σ²f²) + noise_floor` with the floor as a
  free parameter, instead of assuming pure blur. Estimating the floor from the highest-frequency
  bins and subtracting it before the linear fit would remove the downward bias directly.

Both are small changes to `spectral_sigma.py` and neither requires new tuning. But given the
benchmark-mining risk above, the honest sequencing is: **second-seed-validate σ=1.6 first**, and
only then revisit adaptivity.

## Reproduce

```
cd experiments/psf_matched_adaptive
python run_experiment.py   # variant A, unscaled, parameter-free
python run_scaled.py       # variant B, dev-derived k = 1.555
```

Outputs: `outputs/per_pair_results_adaptive.csv`, `outputs/integration_gate_result.json`,
`outputs/adaptive_metrics.json`, `outputs/per_pair_results_adaptive_scaled.csv`,
`outputs/integration_gate_scaled.json`, `outputs/metrics_scaled.json`.
