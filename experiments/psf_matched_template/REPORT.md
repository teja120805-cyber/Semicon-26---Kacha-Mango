# experiments/psf_matched_template — NOT INTEGRATED (fails gate), but the strongest result this project has produced

## Summary

**Pooled accuracy 0.7436 → 0.7692 (+2.56pp), and `validation` improved 0.900 → 0.950 — the first
time any experiment in this project has moved that split rather than tying its 90% ceiling.**
Runtime is 1.00x. The integration gate nevertheless **fails** on criteria 4 and 5, so per project
rules this does **not** merge. It is reported here as a validated mechanism with a clear, specific
follow-up, not as a candidate for integration.

The change is one line of physics. The Reference and Search images travel different optical and
resampling paths, so the template the pipeline correlates is roughly **16x sharper** than the
image it is correlated against:

| | blur applied | in Search-pixel terms |
|---|---|---:|
| Reference → template | `blur_sigma_ref_px = 0.6` at 1000px scale, then shrunk 10x | **0.06** |
| Search | `blur_search_effective_px = 1.0` at fine scale, then area-averaged 10x | **1.00** |

Convolving the template with `sigma_extra = sqrt(1.0² − 0.06²) ≈ 1.0` puts both sides of the
correlation in the same passband. Nothing else changes: same hypothesis grid, same peak count,
same `rank_classical`, same dedup, same tie-break.

## 1. Why this was tried

`crop_uniqueness_ceiling` §3 measured the mechanism: the Search content at the true location and
at the decoy differ at ZNCC **0.732**, yet the template separates them by **0.0098**. The template
was discarding the information that distinguishes them. Template fidelity — not ranking, not
candidate discovery, not periodicity — was the bottleneck.

A pre-check confirmed the prediction before the experiment was built. Sweeping `sigma_extra` and
measuring template fidelity at ground truth on 20 failing pairs:

| `sigma_extra` | fidelity @ ground truth | true location beats the decoy |
|---:|---:|---:|
| 0.0 (production) | 0.7785 | 0 / 20 |
| 0.7 | 0.8290 | 1 / 20 |
| **1.0** | **0.8507** | 4 / 20 |
| 1.3 | 0.8361 | 8 / 20 |
| 1.6 | 0.8082 | 10 / 20 |

Fidelity peaks at **sigma = 1.0**, exactly the value the blur parameters predict. That is a
genuine confirmation of the mechanism, not a fitted result.

## 2. Two configurations, both reported

`sigma_extra = 0.0` is included in the sweep and reproduces the production pipeline **exactly**
(development acc 0.583, mean error 111.50px, identical to baseline to the decimal) — the sweep
contains its own null control.

**(a) Dev-selected, `sigma_extra = 1.6`.** The `development`-only sweep chose this. Full frozen
benchmark, run once:

| split | baseline | candidate |
|---|---:|---:|
| development | 0.583 | **0.750** |
| validation | 0.900 | **0.950** |
| held_out | 0.650 | **0.675** |
| challenge | 0.750 | 0.656 |
| cross_generator | 0.800 | 0.800 |
| **pooled (n=156)** | **0.7436** | **0.7692 (+2.56pp)** |

Rescued 13, broken 9, net +4. Runtime 1.00x.

**(b) Physically-derived, `sigma_extra = 1.0`.** Pre-registered in `psf_match.py`'s docstring from
the blur parameters *before* any benchmark ran, so this is not a value read off the frozen set.

| pooled | 0.7436 → **0.7564 (+1.28pp)** |
|---|---|

Rescued 5, broken 3. Runtime 0.97x.

Both full runs are reported. Neither is presented as the best of N — (a) is what dev-only
selection produced, (b) is what the physics predicted, and (a) is the better of the two.

## 3. Why the gate fails, and what it reveals

| criterion | σ=1.6 | σ=1.0 |
|---|---|---|
| 1 improves validation | **pass** | **pass** |
| 2 improves held_out | **pass** | fail |
| 3 improves/ties cross_generator | **pass** | fail |
| 4 no catastrophic increase | fail | fail |
| 5 no per-family regression | fail | fail |
| 6 acceptable runtime | **pass** (1.00x) | **pass** |
| 7 stable across seeds | pass | pass |

Per-family at σ=1.6 shows a clean split:

| gains | | regressions | |
|---|---:|---|---:|
| dev_dense_periodic | +0.250 | ch_worst_case | −0.250 |
| dev_single_mat | +0.250 | ho_vignette_gamma | −0.200 |
| ch_combined_acquisition | +0.125 | ch_barrel_charging | −0.125 |
| ho_heavy_noise | +0.100 | ch_speckle_saltpepper | −0.125 |
| ho_rotation_drift | +0.100 | | |
| ho_scale_drift | +0.100 | | |
| val_mat_boundary | +0.100 | | |
| val_same_preset_boundary | +0.100 | | |

Every gain is a clean-optics or periodic-ambiguity family. Every regression is a family with an
*additional* degradation — barrel distortion, vignette/gamma, speckle/salt-pepper, or worst-case
combinations. Those images have already lost or corrupted high-frequency content; blurring the
template further destroys what discriminating signal remains.

**A methodological finding worth recording separately:** the `development` split contains only
three families (`dev_strip_anchor`, `dev_single_mat`, `dev_dense_periodic`), all clean-optics.
It contains **no degraded-acquisition family at all**, so a dev-only sweep is structurally blind
to over-blur damage and over-selected σ=1.6. Every hyperparameter this project has ever tuned
dev-only inherits that blind spot. This is a defect in the tuning protocol, not in this
experiment, and it affects prior conclusions too.

## 4. Verdict and follow-up

**Do not integrate.** A +2.56pp pooled gain does not justify overriding two gate criteria,
especially when the regressions concentrate on exactly the degraded-acquisition families a
production tool most needs to survive.

But the mechanism is validated, and the failure is specific and addressable: **a single global
sigma is a compromise between families that want more blur and families that want none.** The
obvious next step is a per-pair `sigma_extra` estimated blind from the image pair — for instance
by matching the radially-averaged power spectra of the downsampled Reference and the Search image,
which requires no ground truth and no generator parameters. Pairs whose Search image is already
degraded would then receive little or no additional blur automatically, which is precisely the
behaviour the per-family table asks for.

Secondary follow-up: fix the `development` split's family coverage before tuning anything else
dev-only.

## Reproduce

```
cd experiments/psf_matched_template
python run_experiment.py            # dev sweep + frozen benchmark at the dev-selected sigma
python run_fixed_sigma.py 1.0 physical   # frozen benchmark at the pre-registered physical sigma
```

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_psf.csv`,
`outputs/integration_gate_result.json`, `outputs/psf_metrics.json`,
`outputs/per_pair_results_physical.csv`, `outputs/integration_gate_physical.json`,
`outputs/metrics_physical.json`.
