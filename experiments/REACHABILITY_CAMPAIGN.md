# Campaign — the reachability ceiling (2026-08-16)

Consolidated record of six independent experiments run in one session, in the same spirit as
`experiments/ACCURACY_90_CAMPAIGN.md`. **Production is unchanged at 77.6%@5px.** `pipeline/`,
`generator/` and `model/` were never modified.

**Headline: no accuracy improvement was found. One validated calibration improvement was.** The
campaign's real output is a measurement that reframes what is worth trying next.

## The unifying finding

> **74% of remaining failures are unreachable.** Ground truth is not within 5px of *any* candidate
> in the pool, so no re-scoring, re-ranking or tie-breaking stage — however good — can fix them.

Measured on 19 failures across two surfaces (`development` 4/7 unreachable, a fresh degraded
40-pair surface 10/12). This is sharper than the project's existing 45% candidate-generation figure
from `experiments/oracle_ceiling_diagnostic/`; both can be true, since they were measured on
different surfaces, but the direction is consistent and stronger here.

The corollary that makes it actionable:

> **Recall is not the binding constraint either.** Widening candidate generation lifts pool recall
> from 0.750 to 0.900, and converts **none** of it into accuracy — across 144 configurations,
> **zero** rescues, with breaks rising monotonically as recall rises.

Production accuracy decomposes as `recall × selector efficiency` = 0.750 × 0.93 on the degraded
surface. **The selector is already ~93% efficient on the pool it is given.** Widening the pool to
recall 0.900 drops selector efficiency to 0.78 for no net gain. So the lever is neither "score the
candidates better" nor "propose more candidates" — it is **make the true location score highly in
the first place**, i.e. template fidelity, which is exactly item 1 in `PROJECT_STATUS.md`'s
next-steps list.

## The six experiments

| # | experiment | verdict | evidence |
|---|---|---|---|
| 1 | `discriminability_weighted/` — P3 weighted ZNCC | **REJECT** | 0 rescues in 60 configs on 64 pairs |
| 2 | `wide_pool_rescoring/` — wider pool + re-scorer | **REJECT** | 0 rescues in 144 configs; breaks scale with recall |
| 3 | `psr_confidence/` — P4 PSR | **REJECT ×2**, **1 win** | PSR AUC 0.577 vs gap 0.964; threshold recalibration validated |
| 4 | `anisotropic_psf/` — horizontal-jitter-matched blur | **NOT SUPPORTED** | +8.3pp on dev, exactly 0 effect on 40 independent pairs |
| 5 | `template_fidelity_ablation/` — spatial high/band-pass | **REJECT** | monotone harm: −2, −4, −6 net at σ = 8, 16, 32 |
| 6 | `aperiodic_anchor/` — sub-model prior to *remove* decoys | **REJECT** | 0 rescues at every radius; −6 to −7 net |

### 1. P3 — discriminability-weighted ZNCC → REJECT

Both survey-proposed weighting schemes, two null controls verified bit-for-bit per pair. The
lattice-shift weights are genuinely non-degenerate (Gini 0.57, 44% more mass on top-decile gradient
pixels than uniform, 0 fallbacks in 64 pairs) — the idea was implemented properly and still does
nothing. Confuser-variance weighting is near-uniform (Gini 0.09) and is a weaker implementation of
its own idea. On the 5 reachable failures the weighted margin moves the **wrong way** on 4.

### 2. Wider pool + re-scoring → REJECT

The one combination the project had not tried: each half was individually useless for a reason the
other supplies. Together, still useless — and cleanly so. Every extra candidate widening reveals is
one the re-scorer prefers *over* the truth, so the weighted score has **negative** discriminative
value over this candidate set.

**Side finding:** `wider_candidate_pool/`'s "structural no-op" conclusion has **partly expired**.
Verified by direct attribution — with `ranking`'s multiway tier disabled, widening is a bit-exact
no-op at every k; with it enabled, it changes predictions. The original arg-max argument still
holds; the arg-max is simply no longer the last word, because the multiway centre tie-break
(integrated 2026-08-15) makes pool width observable. Impact is 1 pair and accuracy-neutral, so this
is a documentation-correctness issue — but `PROJECT_STATUS.md` presents the no-op as structural and
permanent, and should be qualified.

### 3. P4 — PSR → rejected twice, plus the campaign's one real win

**Rejected as the dual-arm selector**: 0.708 → 0.583 (dev), 0.700 → 0.650 (degraded).
**Rejected as a confidence statistic**: AUC 0.577 / 0.765 against the existing pool gap's
0.964 / 0.941. Likely mechanism — on a periodic array the correlation surface is *itself* periodic,
so PSR's "sidelobe" population is full of lattice repeats of the peak and σ_sidelobe is inflated
exactly where discrimination is hardest. P4 inherits the noise-like-background assumption that
makes ZNCC the wrong matched filter here in the first place.

**WIN — `AMBIGUITY_THRESHOLD` 0.92 → 0.990.** The documented miscalibration is real and reproduces
independently (37/40 pairs flagged at 0.324 precision). But the diagnosis was incomplete: the
*statistic* is good (AUC 0.933–0.949); the *constant* is wrong — `ambiguity_ratio` spans
0.816–0.999 with median 0.985, so a 0.92 cut sits far below the distribution and flags nearly
everything.

Fitted on `development` + degraded surface (n=64), tested **once** on held-back seed 271828 (n=40)
after the rule was fixed in code:

| | flag rate | precision | failure recall | answered | acc on answered |
|---|---:|---:|---:|---:|---:|
| production 0.92 | 0.850 | 0.324 | 1.000 | 0.150 | 1.000 |
| **recalibrated 0.990** | 0.300 | **0.750** | 0.818 | **0.700** | **0.929** |

Precision more than doubles and holds within 0.023 of its fitted value on unseen data. Cost
disclosed: failure recall 1.000 → 0.818. Independently reproduces
`crop_uniqueness_ceiling/` §4's "69% at 92.5%" operating point — via a single constant on a
statistic already in production rather than a separate gating mechanism.

Alternative operating point **0.984**: 43.8% answered at **100%** accuracy, zero missed failures
(fit surfaces only — not validated, and quoting it as validated would be the exact error the
protocol exists to prevent).

### 4. Anisotropic template PSF → NOT SUPPORTED

Derived from the generator, not guessed: `apply_raster_shear_drift` runs **after** the 10×
downsample, so every pair carries ~0.4px **horizontal-only** per-row jitter at Search resolution,
which the template's isotropic blur structurally cannot match.

Looked strong on `development` — +8.3pp, 2 rescued / 0 broken, single-peaked in σx, every σy
reduction harmful, i.e. exactly the predicted shape. On 40 independent pairs: **0 rescued, 0 broken
at all three settings.** Combined 2–0 across 64 pairs, sign test p = 0.25. The dev result was an
11-config sweep on 24 pairs and is reported as noise.

`validate_fresh` was deliberately **not** spent — the independent check already returned an exact
tie, and drawing again until a favourable number appears is the benchmark-mining this project's
conventions forbid.

**Two reusable outputs survive the negative verdict:**
- **Shear hypotheses are ruled out by arithmetic** — 1.0px across 1000 rows is 0.1px across a 100px
  template, an order of magnitude under tolerance. This is the obvious next move by analogy with
  the two integrated grid experiments, and it would have been wasted effort.
- **Template blur moves decisiveness far more than it moves peak location** — the anisotropic arm
  won the decisiveness comparison on 24/40 pairs and changed zero outcomes. Judge template work on
  peak location, never on score or decisiveness gains.

### 5. Spatial high-pass / band-pass filtering → REJECT, and it explains three earlier failures

Screened seven image-domain interventions on a truth-vs-decoy margin diagnostic. High-pass scored
best — 2 of 12 failures made winnable against the control's 0 of 12 — and end-to-end was
**decisively harmful with a monotone dose-response**: net −2, −4, −6 at σ = 8, 16, 32. Harm scaling
with filter strength is a mechanism signature, not noise.

**The mechanism, and it unifies three separate rejections.** The aperiodic content that
discriminates one location from another — mat boundaries, strip edges, array transitions — lives
substantially at **low spatial frequency**. It is a large-scale intensity change, not fine texture.
So high-pass filtering removes precisely the discriminating signal while leaving the periodic cell
texture (mid-band) to dominate. Three experiments now fail for a shared reason:

| experiment | effect on spectrum | outcome |
|---|---|---|
| `parallel_pipeline/` P1 prewhitening | boosts high band | REJECT — breaks ≈ rescues |
| `parallel_pipeline/` §3 lattice notching | deletes lattice harmonics | REJECT — destroyed a correct pair by 275px |
| `template_fidelity_ablation/` | subtracts low band | REJECT — monotone harm |

P1's own post-mortem blamed **noise amplification in the high band**. This result points to a
second, independent cause it missed: **signal destruction in the low band**. The band-limited
successor P1 recommended addresses only the first cause — which may be why it never looked
compelling enough to build. **Frequency-domain reweighting is now closed by three mechanisms, not
one.** Any future proposal in that family should be required to show it *preserves* low-frequency
boundary content; none of the survey's P1/P2 formulations do.

A methodological note worth keeping: the margin diagnostic **stated its own limits in advance** —
it compares truth only against the *currently winning* decoy, and looks only at failures, so it is
blind to newly promoted decoys and to damage among correct pairs. Both caveats materialised exactly.
A margin screen is a **necessary-condition filter** (it cheaply killed five of seven interventions)
and a positive result buys an end-to-end run, never a conclusion.

### 6. Aperiodic-anchor search restriction → REJECT

The one structurally untried direction: every other change re-scores the pool or enlarges it, and
`wide_pool_rescoring/` showed enlarging is strictly harmful. This **removes decoys** instead — match
the aperiodic sub-region only, use it as a spatial prior, discard candidates far from it. Backed by
two of the survey's own cited sources (KLA US9830421B2, Cognex US6324299B1) that never became P1–P6
proposals. Distinct from P3 because restricting the *support* removes the periodic content from the
ZNCC denominator entirely, which reweighting never did.

**Zero rescues at every radius; 6–7 pairs broken.**

**The diagnostic predicted it in advance.** GO criteria were written into the script before it ran:
prior nearer truth than the decoy on most failures, and covering nearly every correct pair. Measured:
nearer truth on **5/12** failures, covering **20/28** correct pairs. Neither met; the end-to-end run
confirmed the prediction.

**The mechanism, which closes the whole family.** On correct pairs the prior lands at median 0.8px
from truth — it is a *good* locator. But on failures it is nearer truth only 5/12 times, barely
better than chance. **Its errors are correlated with the full template's errors**, because both are
driven by the same aperiodic content in the same image: when that content is ambiguous or corrupted,
both go wrong together and the prior then confidently deletes the correct candidate. A spatial prior
is only useful as a filter if its errors are *independent* of what it filters. Any future anchor or
sub-model proposal needs a genuinely different measurement channel, not a different sub-region.

## Methodology note — a known defect, addressed

`PROJECT_STATUS.md` records that **`development` contains no degraded-acquisition family**, so every
dev-only sweep in this project has been blind to over-smoothing and noise-amplification damage — the
exact failure mode that sank P1. This campaign generated two additional surfaces with the production
generator at fixed seeds, spanning the axes `development` lacks:

| surface | seed | n | role | baseline |
|---|---:|---:|---|---:|
| `development` (frozen) | 777001 | 24 | tuning | 0.7083 |
| `tune_degraded` | 314159 | 40 | tuning — degraded axes | 0.7000 |
| `validate_fresh` | 271828 | 40 | held back | — |

This is a **deliberate deviation** from "tune on the 24-pair development split only", flagged rather
than buried: these are newly generated pairs, not frozen scoring surfaces, so tuning on them is not
benchmark mining — and tuning without a degraded family is a known-defective procedure. Experiment 4
is a direct vindication: it would have been reported as a +8.3pp win on `development` alone.

## Run-count disclosure

**Frozen 156-pair benchmark: 0 runs across the entire campaign.** Nothing earned one.

| surface | runs |
|---|---|
| `development` (24) | 4 baselines, ~75 swept configs, 2 diagnostics, 1 null-control pass |
| `tune_degraded` (40) | 9 baselines, ~157 swept configs, 3 diagnostics, 1 recall measurement |
| `validate_fresh` (40) | **1 run** — the calibration test read, after the rule was fixed |

The large config counts on experiments 1 and 2 would be a serious multiple-comparisons problem had
anything been selected from them. Nothing was — every configuration tied or lost, so there is no
selected result to correct for. Experiment 4's 11-config sweep on 24 pairs **did** produce a
selected-looking result, which is why it was independently checked and reported as noise.

## What this implies for the next attempt

1. **Stop building re-scorers.** Three now (nine in `ACCURACY_90_CAMPAIGN.md`, plus P3, plus the
   wide-pool combination). The reachability measurement explains why the whole class fails: 74% of
   failures are out of its scope, and the selector is already 93% efficient inside its scope.
2. **Stop reshaping the spectrum.** Closed by three independent mechanisms (experiment 5). Require
   any new proposal in this family to show it preserves low-frequency boundary content.
3. **Template fidelity remains the only lever the evidence supports**, but every obvious route is
   now closed: matching the jitter (exp. 4, no effect), filtering (exp. 5, harmful), and spatially
   restricting to the aperiodic region (exp. 6, harmful). Judge any successor on **peak location**,
   not on score or decisiveness, which move easily and mean little.
3b. **A useful spatial prior must come from an independent measurement channel.** Experiment 6's
   failure was not a tuning problem — the anchor's errors are correlated with the pipeline's, so it
   deletes the right answer exactly when it is needed. This rules out the sub-model family as
   currently framed.
4. **Do not add shear hypotheses.** Ruled out by arithmetic (experiment 4 §1).
5. **The frozen benchmark is intact** — 0 runs this campaign, so it retains full statistical value
   for whatever comes next.
6. **The calibration fix is ready to apply** and is independent of all of the above.

## Reports

- `experiments/discriminability_weighted/REPORT.md`
- `experiments/wide_pool_rescoring/REPORT.md`
- `experiments/psr_confidence/REPORT.md`
- `experiments/anisotropic_psf/REPORT.md`
- `experiments/template_fidelity_ablation/REPORT.md`
- `experiments/aperiodic_anchor/REPORT.md`
