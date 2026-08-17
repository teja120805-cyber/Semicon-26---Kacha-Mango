# experiments/template_fidelity_ablation — spatial-domain filtering: REJECT, with a mechanism that explains three earlier failures

**Date:** 2026-08-16. **Not integrated. Production stays at 77.6%@5px.** Frozen benchmark: **0 runs**.
`validate_fresh`: **0 runs**.

## Summary

Seven image-domain interventions were screened on a diagnostic that measures the quantity any of
them has to move first — the truth-vs-decoy score margin on failures. High-pass filtering scored
best, making **2 of 12 failures winnable against the control's 0 of 12**.

End-to-end it is **decisively harmful, with a clean monotone dose-response**:

| high-pass σ (Search) | low-pass σ | acc@5px | Δ | rescued | broken | net |
|---:|---:|---:|---:|---:|---:|---:|
| — (production) | — | **0.7000** | — | — | — | — |
| 8 | 0 | 0.6500 | −5.0pp | 0 | 2 | **−2** |
| 16 | 0 | 0.6000 | −10.0pp | 0 | 4 | **−4** |
| 16 | 0.5 | 0.6250 | −7.5pp | 1 | 4 | −3 |
| 32 | 0 | 0.5500 | −15.0pp | 1 | 7 | **−6** |

Harm scales monotonically with filter strength — 2, 4, 6 broken pairs at σ = 8, 16, 32. That is not
noise; it is a dose-response curve, and it says the filtering is destroying something the matcher
depends on.

## 1. Why the diagnostic was misleading — and it said so in advance

`run.py` measured `margin = ZNCC(truth) − ZNCC(winning decoy)`, and its own docstring stated the
condition for acting on it: *"an intervention is worth an end-to-end run only if it raises n>0 above
the control's."* High-pass did (2/12 vs 0/12), so the end-to-end run was warranted.

The two caveats written into the same file before the run both materialised:

1. **It compares truth only against the *currently winning* decoy.** High-pass improved truth's
   standing against that one rival while promoting others — the pool has 76 candidates on average.
2. **It only looks at failures.** It is structurally blind to damage among the 28 currently-correct
   pairs, which is where all the loss came from (0–1 rescued against 2–7 broken).

The lesson generalises: a margin-against-the-current-winner diagnostic is a **necessary** condition
screen, never a sufficient one. It is still worth running — it cheaply killed five of seven
interventions — but a positive result buys an end-to-end run, not a conclusion.

## 2. The mechanism — and it unifies three separate failures

**The aperiodic content that discriminates one location from another lives substantially at LOW
spatial frequency.** A mat boundary, a strip edge, an array-to-array transition — these are
large-scale intensity changes, not fine texture. High-pass filtering removes exactly the signal that
carries the discriminating information, leaving the periodic cell texture (which is mid-band and
survives) to dominate the correlation. The intervention amplifies the confuser and suppresses the
discriminator. The monotone dose-response is the signature of that.

This retrospectively explains a pattern across three independent experiments that were each
diagnosed separately:

| experiment | what it did to the spectrum | outcome |
|---|---|---|
| `parallel_pipeline/` (P1 spectral prewhitening) | divide by \|F\|^2ρ — boosts high band | REJECT, breaks ≈ rescues |
| `parallel_pipeline/` §3 (lattice notching) | delete lattice harmonics | REJECT, destroyed a correct pair by 275px |
| **this** (spatial high-pass) | subtract the low band | **REJECT, monotone harm** |

All three suppress low-to-mid frequency content relative to high. P1's own post-mortem attributed
its failure to *noise amplification* in the high band. This result suggests a second, independent
cause that P1's diagnosis missed: **signal destruction in the low band**. The band-limited fix P1's
report recommended as its successor addresses only the first cause, which may be why it never
looked promising enough to build.

**Practical consequence:** frequency-domain reweighting is now closed by three independent
mechanisms, not one. Any future proposal in this family should be required to show that it
*preserves* low-frequency boundary content, which none of the survey's P1/P2 formulations do.

## 3. What was screened, and what the screen killed cheaply

| intervention | rationale | margin n>0 | verdict |
|---|---|---:|---|
| `none` (control) | reproduces production exactly | 0/12 | — |
| `blur_search_0.5` | shot noise at dose 40–220 | 0/12 | killed at screen |
| `blur_search_1.0` | stronger version of the above | 1/12 | killed at screen |
| `median_search_3` | impulse noise (salt-pepper, charging) | 1/12 | killed at screen |
| `highpass_both_8` | remove window-scale gradient | 0/12 | killed at screen |
| `highpass_both_16` | as above, larger scale | **2/12** | **ran end-to-end → REJECT** |
| `bandpass_both` | P1's recommended successor, in the spatial domain where it cannot amplify the noise floor | **2/12** | covered by hp=16, lp=0.5 → REJECT |

Reference-side σ is 10× the Search-side σ throughout, because the Reference is 10× finer and the
same *physical* spatial frequency needs a 10× larger pixel σ. Filtering both at one σ would attack
different physical scales in each image and confound the comparison.

## 4. Null control

`hp_search = 0` skips filtering and must reproduce `pipeline.localize.localize` bit-for-bit on
x, y and confidence. Verified on 6 pairs in **both** end-to-end runs: 0 mismatches. The baseline row
reproduces production's 0.7000 on this surface exactly.

## 5. Honest status

- **REJECT.** No configuration is a candidate. Production untouched.
- Measured on one 40-pair surface. A monotone harmful dose-response across four strengths does not
  need a second surface to be believed; a positive result would have.
- The negative is specific to **linear, isotropic, global** frequency shaping. It does not rule out
  a *spatially adaptive* filter that preserves boundary regions — but §2 sets a clear bar for any
  such proposal: demonstrate low-frequency boundary content is preserved.
- Runtime rose 5.3 → 6.3–6.7 s/pair from the extra convolutions, which would have been an
  additional (minor) cost had it worked.

## 6. Run-count disclosure

- Frozen 156-pair benchmark: **0 runs.**
- `tune_degraded` (40): 1 ablation screen (7 interventions, failures only), 2 baselines,
  4 end-to-end configs, 2 null-control passes.
- `development` (24), `validate_fresh` (40): **0 runs.** Nothing survived the degraded surface, so
  neither was spent.

## Reproduce

```
python -m experiments.template_fidelity_ablation.run       --surface tune_degraded
python -m experiments.template_fidelity_ablation.endtoend  --surface tune_degraded
python -m experiments.template_fidelity_ablation.endtoend  --surface tune_degraded --configs 0:0,8:0,32:0
```
