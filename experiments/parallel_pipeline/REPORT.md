# experiments/parallel_pipeline — P1 spectral prewhitening: REJECT so far, with a specific reason and a specific next step

## Summary

**Not integrated. Production stays at 77.6%.** Spectral prewhitening (P1 from
`reports/RESEARCH_SURVEY_SCORING.md`) was implemented in three progressively safer forms. All three
are net-harmful, but they fail in an informative way: the whitened representation demonstrably
**contains real signal that ZNCC cannot see** — it rescues pairs by two orders of magnitude — while
being **too noise-dominated to arbitrate reliably**, breaking about as many pairs as it fixes.

The theory motivating P1 is not refuted. The specific implementation is, and the failure points at
a concrete fix (§5).

## 1. What was built

`experiments/parallel_pipeline/` is a sandbox for alternative *scoring* backends, kept separate from
the one-idea-per-folder convention because these are meant to be A/B'd on a shared harness.

- `spectral.py` — whitening filter `1/(|F|²+λ)^ρ`, targeted lattice-harmonic notching, and PSR.
- `harness.py` — production pipeline structure with the correlation run on filtered images.
- `arbitrate.py` — production pipeline unchanged, whitened score consulted **only** among near-ties.

Every backend has a null setting (`ρ=0`, `depth=0`, `tie_eps=0`) verified to reproduce production
**bit-for-bit per pair**, not assumed.

## 2. Form A — whitening as a replacement representation

Filter both images, run the unmodified correlation on the result. Tested on 10 currently-failing
pairs at ρ ∈ {0, 0.15, 0.3, 0.5}.

| pair | baseline | ρ=0 | ρ=0.15 | ρ=0.30 |
|---|---:|---:|---:|---:|
| `ho_rotation_drift_006` | 30.77 | 30.77 | **0.36** | **0.28** |
| `ch_combined_acquisition_001` | 59.82 | 59.82 | **0.33** | **0.29** |
| `dev_dense_periodic_002` | 8.40 | 8.40 | 856.60 | 355.97 |
| `ho_heavy_noise_009` | 16.45 | 16.45 | 514.03 | 514.07 |

Rescues of 30.77 → 0.36 and 59.82 → 0.33 are not marginal — the whitened score finds locations the
raw score misses entirely. But the breaks are just as large, and they concentrate on noisy families.

## 3. Form B — targeted lattice notching

Suppress only the strongest discrete harmonics, leaving the rest of the spectrum alone. Intended as
the surgical counterpart to broad whitening. **It is worse**, and the safety check is decisive:

| pair | baseline | depth=0 | depth=0.5 |
|---|---:|---:|---:|
| `ch_speckle_saltpepper_002` (currently CORRECT) | 0.42 | 0.42 | **274.82** |

Destroying a currently-correct pair by 274px rules this form out on its own.

## 4. Form C — bounded arbitration among near-ties

The safest possible use: candidate generation, ranking and refinement stay exactly as production;
the whitened score is consulted *only* to choose among candidates within `tie_eps` raw-ZNCC of the
top. Our own data says this costs no coverage — every failing pair measured is a near-tie within
0.05 (`experiments/oracle_ceiling_diagnostic/` §2) — and a clear winner cannot be disturbed.

Tested on 8 failing + 8 correct pairs, ρ=0.3, `tie_eps=0.02` (`tie_eps=0` verified identical to
production on all 16):

| outcome | pairs |
|---|---|
| **rescued** | `ch_worst_case_005` 630.88 → **0.62**; `ho_heavy_noise_003` 55.99 → **0.05**; `ho_rotation_drift_006` 30.77 → **0.44** |
| **broken** | `ch_worst_case_003` 21.16 → 73.21; `dev_dense_periodic_002` 8.40 → 50.26; `ch_speckle_saltpepper_002` 0.42 → 31.93; `ho_heavy_noise_006` 0.36 → 30.31 |

**3 rescued, 4 broken.** Two of the four breaks were previously correct. Even fully bounded, the
whitened score is close to a coin flip.

## 5. Why it fails, and what follows from that

The mechanism is now clear and it is not subtle. **The periodic array carries most of the image
energy.** Whitening removes it — which is the intent — but what remains is the aperiodic boundary
signal *and the sensor noise*, at comparable amplitude. `dose_search = 220` is genuinely noisy, and
whitening by `1/|F|^2ρ` amplifies exactly the high-frequency band where that noise dominates and
signal does not. The result is a score that sometimes locks onto real aperiodic structure
(spectacular rescues) and sometimes onto a noise realization (spectacular breaks).

That diagnosis names its own fix: **band-limit the whitening.** Whiten only the mid-band where the
lattice harmonics actually live, and leave the high band alone. Concretely, this composes with
something already in production — the PSF-matching blur (σ=1.6) suppresses high frequencies, while
whitening boosts them; applying both yields band-pass emphasis rather than broadband amplification.
That is a small change to `build_whitening_filter` (multiply by a low-pass envelope, or equivalently
whiten the PSF-matched images rather than the raw ones) and it is the obvious next experiment.

A second, independent option from the survey that this work did not reach: **P3,
discriminability-weighted ZNCC.** It attacks the same dilution problem in the *spatial* domain
rather than the frequency domain, and so does not amplify the noise floor at all — it reweights
existing pixels rather than reshaping the spectrum. Given how cleanly the frequency-domain route
failed on noise, the spatial route now looks like the better bet of the two.

## 6. Honest status

- P1 as implemented: **REJECT** in all three forms.
- The underlying claim from `reports/RESEARCH_SURVEY_SCORING.md` §1 — that ZNCC is the wrong measure
  for periodic structure — is **not** refuted, and the size of the rescues is evidence for it.
- Nothing here is a candidate for integration. Production is unchanged.
- PSR (P4) is implemented and computed but has not yet been evaluated as a replacement for the
  dual-arm selector or `AMBIGUITY_THRESHOLD`; that remains cheap and worth doing.

## Reproduce

```
cd experiments/parallel_pipeline
python -c "..."   # see harness.localize_spectral / arbitrate.localize_arbitrated
```

Sample sizes here are deliberately small (10–16 pairs) because these were mechanism checks, not
benchmark runs — no configuration reached the frozen benchmark, and none should on this evidence.
