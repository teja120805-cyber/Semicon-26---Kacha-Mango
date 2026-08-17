# experiments/discriminability_weighted — P3 discriminability-weighted ZNCC: REJECT as a re-scorer, but it located the real ceiling

**Date:** 2026-08-16. **Status:** complete on tuning surfaces. **Not integrated. Production stays
at 77.6%@5px.** The frozen 156-pair benchmark was **not** run for this experiment — nothing here
earned a frozen run, and the run budget is disclosed in §7.

## Summary

P3 from `reports/RESEARCH_SURVEY_SCORING.md` was implemented in both weighting schemes the survey
proposed, with two independent null controls verified bit-for-bit per pair. The mechanism is real
but **the wrong lever**:

- The weights are **not degenerate**. Lattice-shift weighting concentrates 44% more mass on the
  template's top-decile gradient pixels than uniform (0.144 vs 0.100), Gini 0.57–0.58, with zero
  fallbacks across 64 pairs. Confuser-variance weighting *is* near-degenerate (Gini 0.09, edge mass
  0.101 ≈ uniform) and is rejected on that basis alone.
- Re-scoring **fires often and changes almost nothing**: on `development` it engaged on up to 21 of
  24 pairs and changed the winner on at most 1, rescuing 0 and breaking 0 at every setting except
  one harmful corner (`confuser_variance`, α=1.0, which broke 1).
- The reason is not the score. **14 of 19 failures across both tuning surfaces are unreachable** —
  ground truth is not within 5px of *any* pooled candidate — so no re-scoring stage of any kind can
  fix them. P3's addressable set is 5 pairs of 19, and within it the weighted margin moves the
  *wrong way* on 4 of 5.

That last measurement is the useful output of this work. It is also what motivated
`experiments/wide_pool_rescoring/`, which tests the one combination the project has not tried.

## 1. What was built

```
weighted_zncc.py   ZNCC_w primitives: point form, and a dense form needing three
                   cross-correlations (not the survey's estimated four — sum(w*I) is reused)
weights.py         Scheme A lattice-shift dissimilarity; Scheme B confuser variance
harness.py         Production pipeline, unmodified, with weighted re-scoring among near-ties
verify_null.py     Two independent null checks
sweep.py           Cached-pool parameter sweep
diagnose.py        Oracle separation + weight-map health
pool_recall.py     Recall ceiling as a function of candidate-generation width
```

`pipeline/`, `generator/` and `model/` are untouched. The harness imports production candidate
generation, the dual-arm PSF selection, `rank_classical`, `apply_center_tiebreak` and `refine`
and calls them with production defaults.

**Design choice: bounded to near-ties, not dense.** `experiments/oracle_ceiling_diagnostic/`
established every remaining failure is a near-tie within 0.05 ZNCC, so a tie-bounded rule gives up
no measurable coverage while being unable to disturb a clear winner. It is also the one form of P1
(`experiments/parallel_pipeline/` §4) that stayed interpretable.

## 2. Null controls — verified, not asserted

Two checks, both run in `verify_null.py`:

**Check A — implementation correctness.** ZNCC_w at uniform weights *is* standard ZNCC
algebraically, so it must agree with the OpenCV `TM_CCOEFF_NORMED` that `pipeline/matching.py`
uses. Over 12 templates spanning three (scale, rotation, psf) combinations:

| quantity | result |
|---|---|
| dense max abs diff | 1.461e-05 |
| point max abs diff | 3.024e-06 |
| argmax mismatches | 0 / 12 |

**Check B — null control.** Both null settings reproduce `pipeline.localize.localize` bit-for-bit
on x, y, confidence and ambiguity_ratio — not merely on error@5px, which would hide sub-pixel drift.

| setting | pairs | mismatches |
|---|---|---|
| α = 0.0 (uniform weights, tie_eps 0.05) | 24 | **0** |
| tie_eps = 0.0 (α = 1.0) | 24 | **0** |

Note the honest distinction: check A's 1.5e-05 is *above* `ranking.TIE_SCORE_EPSILON` (1e-6), so
the dense weighted implementation is **not** a bit-exact substitute for production's correlator.
That is why α=0 short-circuits stage 2 outright rather than recomputing production's own numbers
through a second code path. Skipping is what makes the null exact; recomputing would not have.

## 3. A methodology fix this experiment adopted

`reports/PROJECT_STATUS.md` records that **`development` contains no degraded-acquisition family**,
so every dev-only sweep in this project is structurally blind to over-smoothing and
noise-amplification damage — the exact failure mode that sank P1. Tuning α on clean pairs alone
would repeat it.

So `make_datasets.py` generates two additional surfaces with the production generator, at fixed
seeds, spanning the axes `development` lacks (heavy noise, speckle + salt-and-pepper, vignette +
gamma, combined rotation/scale drift, worst-case):

| surface | seed | n | role |
|---|---|---|---|
| `development` (frozen) | 777001 | 24 | tuning |
| `tune_degraded` | 314159 | 40 | tuning — the degraded axes dev lacks |
| `validate_fresh` | 271828 | 40 | held back, never read during tuning |

**This is a deliberate deviation from "tune on the 24-pair development split only", and is flagged
rather than buried.** The justification: these are newly generated pairs, not frozen scoring
surfaces, so tuning on them is not benchmark mining — and tuning *without* a degraded family is a
known-defective procedure. Baseline on `tune_degraded` is 0.700@5px (28/40), close enough to
`development`'s 0.708 to be a comparable difficulty.

## 4. Result — the sweep

60 configurations on `development` (α ∈ {0.25, 0.5, 0.75, 1.0} × tie_eps ∈ {0.01, 0.02, 0.05} ×
2 schemes, plus null settings skipped as already verified):

| scheme | best acc@5px | Δ | rescued | broken | winner changed |
|---|---|---|---|---|---|
| `lattice_shift` | 0.7083 | +0.0pp | 0 | 0 | 0–1 of 24 |
| `confuser_variance` | 0.7083 | +0.0pp | 0 | 0–1 | 0 of 24 |

Baseline 0.7083. **Not one configuration rescued a single pair.** The stage fired on 9/24 pairs at
tie_eps=0.01 and 21/24 at 0.05, so the null result is not "it never ran" — it ran and agreed with
production almost everywhere.

## 5. Why — the diagnostic that matters

`diagnose.py` measures, per pair, whether the true location is reachable at all, and whether
weighting increases the margin `score(truth) − score(chosen)`. Both locations are scored under
**their own** best hypothesis; scoring truth under the chosen candidate's scale/rotation would
understate it exactly on the rotation/scale failures and would have faked a negative result.

**Reachability — the finding.**

| surface | failures | reachable (P3's scope) | unreachable (candidate generation) |
|---|---|---|---|
| `development` | 7 | 3 | 4 |
| `tune_degraded` | 12 | 2 | **10** |
| **total** | **19** | **5 (26%)** | **14 (74%)** |

This is a sharper and less favourable split than the project's existing 45% candidate-generation
figure (`experiments/oracle_ceiling_diagnostic/`). Both can be true — that decomposition was
measured on the frozen benchmark, this one on two different surfaces, one deliberately degraded —
but the direction is consistent and stronger here.

**Margins on the 5 reachable failures.** Positive means the weighting would flip the pair.

| pair | plain | lattice | confuser |
|---|---:|---:|---:|
| `dev_single_mat_003` | −0.0184 | −0.0323 | −0.0354 |
| `dev_single_mat_006` | −0.0124 | −0.0178 | −0.0113 |
| `dev_dense_periodic_004` | −0.0502 | −0.0497 | −0.0655 |
| `tune_speckle_saltpepper_003` | −0.0023 | −0.0002 | **+0.0117** |
| `tune_worst_case_002` | −0.0731 | −0.1509 | −0.1049 |

**Weighting usually makes the margin worse, not better** — 4 of 5 under both schemes. The single
exception (`tune_speckle_saltpepper_003`) is a genuine flip and the only evidence the mechanism can
ever work; it is one pair, and the same setting harms the other four.

**Weight-map health** rules out the "the weights were degenerate" explanation for scheme A:

| metric | uniform | lattice | confuser |
|---|---:|---:|---:|
| Gini | 0.00 | **0.57–0.58** | 0.09 |
| mass on top-decile gradient pixels | 0.100 | **0.140–0.144** | 0.101 |
| fell back to uniform | — | 0 / 64 | — |

Scheme A does what it claimed: it finds the aperiodic content and up-weights it. It just does not
help. Scheme B is near-uniform and is rejected as not meaningfully implementing its own idea —
z-normalizing each rival before taking the variance may be removing the very contrast differences
the weights were supposed to key on, which is a fixable defect but not worth fixing given §5's
reachability ceiling.

## 6. Honest status

- **P3 as a near-tie re-scorer: REJECT.** Zero rescues at every setting on 64 tuning pairs.
- **P3's premise is not refuted, but its scope is much smaller than the survey assumed.** The
  survey argued the discriminating information is present and ZNCC cannot see it. That may hold;
  what this measures is that for 74% of failures the true location is never *offered* to the score
  at all, so improving the score cannot reach them.
- **Scheme A is a working discriminability estimator** and is reusable. Scheme B as implemented is
  not, and its defect is named above.
- **Nothing here is a candidate for integration.** Production is unchanged and untouched.
- The reachability measurement is this experiment's real output, and it directly motivated
  `experiments/wide_pool_rescoring/`.

## 7. Run-count disclosure

Per the project convention on benchmark mining across a line of work:

- Frozen 156-pair benchmark: **0 runs.** Nothing here earned one.
- `development` (24 pairs, frozen but designated for tuning): 1 baseline + 60 swept configs +
  1 diagnostic + 1 null-control pass (48 checks).
- `tune_degraded` (40 pairs, freshly generated seed 314159): 1 baseline + 1 diagnostic +
  1 recall measurement.
- `validate_fresh` (40 pairs, seed 271828): **0 runs.** Never read.

The 60-config sweep is a large number for a 24-pair surface and would be a serious overfitting risk
had anything been selected from it. Nothing was: every configuration tied baseline, so no selection
occurred and there is nothing to have overfitted.

## Reproduce

```
python -m experiments.discriminability_weighted.verify_null
python -m experiments.discriminability_weighted.make_datasets
python -m experiments.discriminability_weighted.sweep     --surface development
python -m experiments.discriminability_weighted.diagnose  --surface tune_degraded
python -m experiments.discriminability_weighted.pool_recall --surface tune_degraded
```
