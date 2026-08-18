# Batch report — four investigations, 2026-08-17

Requested: run everything proposed, then report once. **Production is unchanged at 77.6%@5px.**
`pipeline/`, `generator/` and `model/` untouched. This is a decision document — the recommendation
on each is stated up front.

## Verdicts

| # | Experiment | Source of the idea | Verdict |
|---|---|---|---|
| A | **Gated escalation** — denser hypothesis grid, only on flagged pairs | this session (industrial coarse→fine practice) | **NEAR-MISS — do not integrate as-is.** Best lead the project has; +1.92pp but fails 3 gate criteria |
| B | **Axis decomposition** — are failures 1-D? | US8073242B2, flagged in your survey §2, never tested | **DIAGNOSTIC — major positive finding.** No change to integrate, but it reframes the problem |
| C | **OTSDF / MACE correlation filters** | in your survey's sources, never a P1–P6 proposal | **REJECT** — at chance (11/22) |
| D | **Phase-based matching** | MS-POFT (2025), phase congruency | **REJECT** — 4/22, materially *worse* than chance |

---

## A. Gated escalation — the strongest lead, and still not integrable

**Idea.** Every one of the thirteen prior attempts changed the algorithm for *all* pairs, forcing a
global runtime budget. This leaves production untouched and runs a **denser scale/rotation grid only
on the 35.3% of pairs the pipeline already flags `ambiguous`**. Two things from earlier in this
session made it possible: the recalibrated flag (35.3% instead of 82%, so gating actually saves
something), and `reachability_verification/` identifying which pairs a second pass could help.

It escalates to something already proven: `finer_hypothesis_grid/` doubled grid density, measured
net rescue +16/+6, and **was integrated**. Density was never pushed further purely because of
global runtime. Gating removes that constraint.

**Null control:** factor 1 (production grid) reproduces production bit-for-bit on all three
surfaces. Verified, not assumed.

### It looked excellent on three independent surfaces

| surface | n | baseline | factor 3 | Δ | rescued | broken |
|---|---:|---:|---:|---:|---:|---:|
| `development` | 24 | 0.7083 | 0.7917 | +8.3pp | 2 | **0** |
| `tune_degraded` (seed 314159) | 40 | 0.7000 | 0.7750 | +7.5pp | 3 | **0** |
| `validate_fresh` (seed 271828, held back) | 40 | 0.7250 | 0.7750 | +5.0pp | 2 | **0** |
| **combined** | **104** | | | **+6.7pp** | **7** | **0** |

7–0 gives a sign test of **p = 0.008**, stronger than the p = 0.031 that gate exception 3 cleared.
On that evidence a frozen run was warranted.

### The frozen benchmark disagrees, and it is the number that counts

| | production | gated escalation |
|---|---:|---:|
| accuracy@5px | 0.7756 | **0.7949** (+1.92pp) |
| catastrophic (>50px) | 0.1410 | **0.1667** (worse) |
| rescued / broken | — | **8 / 5** |
| runtime | 1.00× | 3.74× (ceiling 5×) |

| split | before | after |
|---|---:|---:|
| `development` | 0.7083 | 0.7917 |
| `validation` | 0.9250 | 0.9500 |
| **`held_out`** | 0.7000 | **0.6750** |
| `challenge` | 0.7188 | 0.7500 |
| `cross_generator` | 0.8000 | 0.8000 |

**Integration gate: FAILED** — 4 of 7 pass.

| criterion | result |
|---|---|
| 1 improves validation | PASS |
| 2 improves held_out | **FAIL** (0.700 → 0.675) |
| 3 improves/ties cross_generator | PASS |
| 4 no catastrophic increase | **FAIL** (22 → 26 pairs) |
| 5 no per-family regression | **FAIL** (`ho_heavy_noise` 0.800 → 0.700) |
| 6 acceptable runtime | PASS (3.74× of 5×) |
| 7 stable across seeds | PASS (3 surfaces) |

The eight rescues are dramatic — `ho_scale_drift_006` 735px → 0.46px, `ch_worst_case_005` 631px →
0.59px, `ch_barrel_charging_003` 407px → 1.9px. But five previously-correct pairs break, **four of
them into new catastrophic errors** (`ho_heavy_noise_006` 0.36 → 119px, `ho_rotation_drift_001`
0.34 → 205px, `ho_scale_drift_007` 0.34 → 661px, `ch_worst_case_007` 0.26 → 54px).

### Recommendation: do not integrate

Trading a 0.36px answer for a 205px one is the wrong trade for an inspection tool, and it happens
four times. A +1.92pp pooled gain does not pay for a catastrophic rate rising 14.1% → 16.7%.

**Why the tuning surfaces were misleading — a repeat of a mistake from earlier this session.**
`tune_degraded` and `validate_fresh` share the same seven family definitions, which I wrote. They do
not contain `ho_heavy_noise`, `ho_vignette_gamma` or `val_linewidth_bias` in the frozen benchmark's
proportions. Three surfaces agreeing meant less than it appeared, because two of them were siblings.
**Independent seeds are not the same as independent family composition** — the same error that
produced the 74% reachability overstatement.

### The specific fix worth trying next

Escalation currently **always** accepts the dense-grid answer when a pair is flagged. It should
accept it only when the dense-grid pool is *more decisive* — exactly the rule production already
uses to choose between PSF arms (`localize._decisiveness`, gate exception 3). Under that rule a pair
whose escalated pool is less decisive reverts to the production answer byte-for-byte, which would
plausibly recover most of the five breaks while keeping the rescues.

That is a real proposal, not a rescue attempt: the mechanism already exists in production and is
proven. **But it must be tuned on surfaces with the frozen benchmark's family mix, not mine**, and
its frozen number would be the **second** frozen run for this line of work — which must be disclosed
when it is reported.

---

## B. Axis decomposition — the most valuable result in this batch

**Idea.** US8073242B2 (SEM addressing-point recipes) states *"ignoring the uniqueness of any APs in
the X-direction and attaching importance only to the uniqueness of APs in the Y-direction."* Your
survey §2 cites it — "our two axes may be separately solvable" — and never tested it. The patent's
own mechanism does not transfer (it chooses *where to place* a target; you don't choose the crop),
but the underlying question had never been asked of this data.

**Result, on the 22 reachable failures:**

| measurement | value |
|---|---:|
| median along-lattice-axis fraction of the error | **1.000** |
| failures >80% along a lattice axis | **18 / 22** |
| failures >90% along a lattice axis | 17 / 22 |
| **median cross-axis residual** | **1.47 px** |
| within a quarter-period of an integer multiple of the pitch | 17 / 22 |

**The residual failures are a one-dimensional problem.** The pipeline already has the cross-lattice
coordinate right to ~1.5px — comfortably inside the 5px tolerance — and is confused only about
*which lattice period* it sits on, along a single known direction.

**Why this matters.** Every disambiguation attempt so far has searched or scored in 2-D. The actual
residual ambiguity is a **discrete 1-D index**: roughly 20–40 candidate period offsets along one
measurable direction. A disambiguator only needs to be informative along that one axis, which is a
far weaker requirement than any of the thirteen rejected approaches faced.

**It also explains experiment A mechanistically.** A scale-hypothesis quantisation error accumulates
into displacement across the image: at the production step of 0.2 in ~10× scale (2%), the
accumulated shift across a 1000px Search image is ~20px — against a measured lattice pitch of ~14px,
that is **more than one full period**. Tripling grid density cuts the accumulated shift to well
under half a period. So the escalation rescues and the 1-D finding are the same phenomenon seen
twice, and both corroborate your existing forensics Finding 3 (the sawtooth against grid alignment)
and Finding 2 (62–85% of periodicity failures landing near integer pitch multiples — I measure
17/22 = 77%, inside that range).

**Nothing to integrate.** It is a diagnostic. But it is the clearest open lead the project has, and
it is cheap to act on.

---

## C. OTSDF / MACE correlation filters — REJECT

MOSSE/MACE/UMACE appear in your survey's source list but never became a P1–P6 proposal. A MACE
filter minimises average correlation energy away from the target — explicit sidelobe suppression,
aimed squarely at "many near-equal peaks". OTSDF adds a noise-tolerance parameter with a built-in
null: `H(f) = X*(f) / (α|X|² + (1−α)·mean|X|²)`, where α=0 is the plain matched filter.

Screened against the same pre-registered bar as DDIS — separate the median 0.029 ZNCC deficit on the
22 reachable failures, >70% to justify a harness:

| measure | median margin | prefers truth |
|---|---:|---:|
| ZNCC (reference) | −0.0293 | 0/22 (0%) |
| OTSDF α=0.0 *(null control)* | −0.0275 | 4/22 (18%) |
| OTSDF α=0.3 | −0.0106 | 11/22 (50%) |
| OTSDF α=0.6 | −0.0105 | 11/22 (50%) |
| OTSDF α=0.9 | −0.0112 | 10/22 (45%) |
| OTSDF α=1.0 (full MACE) | −0.0000 | 11/22 (50%) |

The null control behaves correctly (α=0 tracks ZNCC's sign pattern). Everything else sits at
**chance**. At α=1.0 the median margin collapses to −0.0000 — the inverse filter makes every
location look equally good, the known degenerate behaviour of MACE on noisy data.

This matches the prior recorded before running: OTSDF divides out low-frequency amplitude, and
`template_fidelity_ablation/` established that is where the discriminating content lives.

## D. Phase-based matching (MS-POFT-inspired) — REJECT

Local phase via a log-Gabor quadrature pair, correlated instead of intensity — contrast-invariant in
a way gradient magnitude is not. An approximation of the published method, labelled as such.

**4/22 (18%) — materially worse than chance**, median margin −0.0138. Discarding amplitude entirely
removes exactly the low-frequency boundary information that distinguishes the true location, so it
performs worse than the measure it was meant to improve on. Same mechanism as C, more severe.

---

## Where this leaves the project

- **Fifteen independent attempts at the selection problem have now failed or fallen short**
  (nine in `ACCURACY_90_CAMPAIGN.md`, four in `REACHABILITY_CAMPAIGN.md`, two here). Your scoring
  survey is fully explored: P1–P5 tested and rejected, P6 ruled out on resolution grounds, and the
  two source-list ideas that never became proposals (correlation filters, directional uniqueness)
  are now tested too.
- **The one lead that is genuinely open is B**, and it is a different *kind* of lead: not a new
  similarity measure, but a reduction of the problem from 2-D to a discrete 1-D index along a
  measurable axis.
- **A is worth one more iteration** with decisiveness-gated acceptance, tuned on surfaces matching
  the frozen benchmark's family mix.

## Run-count disclosure

- **Frozen 156-pair benchmark: 1 run** this batch (experiment A, factor 3, earned by three
  surfaces agreeing). Previously 0 for the whole campaign; plus 1 diagnostic
  (`reachability_verification`) which selected nothing.
- `development` 24: 1 baseline + 4 escalation factors. `tune_degraded` 40: 1 baseline + 3 factors.
  `validate_fresh` 40: 1 baseline + 1 factor.
- Experiments B, C, D: diagnostics on 22 pairs each, no tuning, no selection, no end-to-end runs.
- `factor` was fixed on `development` before the other surfaces were run; factor 4 was rejected
  there for saturating while breaching the runtime ceiling.

## Reproduce

```
python -m experiments.gated_escalation.run --surface development
python -m experiments.gated_escalation.run --surface tune_degraded --factors 1,2,3
python -m experiments.gated_escalation.run --surface validate_fresh --factors 1,3
python -m experiments.gated_escalation.frozen
python -m experiments.axis_decomposition.diagnose
python -m experiments.alternative_scores.diagnose
```

---

# ADDENDUM — decisiveness-gated escalation (frozen run #2)

Requested follow-up. **Recommendation: DO NOT INTEGRATE.** It is, however, the closest any change
has come, and the blocker is specific and fixable.

## The change

Accept the escalated answer only when its candidate pool is **more decisive** than production's —
the same `_decisiveness` comparison production already uses to choose between PSF arms (gate
exception 3). Otherwise the production answer stands byte-for-byte. **No free parameters:** a strict
comparison between two numbers, nothing to tune or select.

**Disclosure, stated plainly.** The idea was prompted by inspecting which pairs plain escalation
broke, and this is the **second frozen run** for this line of work. It is not parameter mining —
there are no parameters — but the +2.56pp below is **not a clean out-of-sample number**, and that
is the single most important caveat in this addendum.

## Result on the frozen benchmark (n=156)

| | production | plain escalation | **decisiveness-gated** |
|---|---:|---:|---:|
| accuracy@5px | 0.7756 | 0.7949 (+1.92pp) | **0.8013 (+2.56pp)** |
| rescued / broken | — | 8 / 5 | **5 / 1** |
| catastrophic (>50px) | 0.1410 | 0.1667 | 0.1538 |
| runtime | 1.00× | 3.74× | 3.74× |
| gate criteria passed | — | 4/7 | **4/7 (different, better set)** |

The gate is doing real work: escalation is accepted on only **16 of 55** flagged pairs (29%). Among
the 39 rejected, escalation *would have* rescued 3 and broken 4 — so the rule correctly filtered a
net-negative set, at the cost of 3 genuine rescues (including `ch_worst_case_005`, 631px → 0.59px).

| criterion | plain | gated |
|---|---|---|
| 1 improves validation | PASS | PASS (0.925 → 0.950) |
| 2 improves held_out | FAIL (regression) | **FAIL (tie, 0.700 → 0.700)** |
| 3 cross_generator | PASS | PASS |
| 4 no catastrophic increase | FAIL | **FAIL (0.1410 → 0.1538)** |
| 5 no per-family regression | FAIL | **PASS** |
| 6 runtime | PASS | PASS (3.74× of 5×) |
| 7 stable across seeds | PASS* | **FAIL — not validated** |

\* plain escalation's criterion 7 was credited to three surfaces that later proved to be siblings.

## Why not integrate

**1. Criterion 7 — no independent validation.** The decisive blocker. The variant was designed after
seeing frozen failures and measured once on that same benchmark. Everything else is secondary.

**2. Criterion 4 — catastrophic rate 14.1% → 15.4% (22 → 24 pairs).** Nuance worth recording: of
the three pairs that became catastrophic, **two were already failures** (`dev_single_mat_006`
11.9 → 732px; `ho_vignette_gamma_005` 25.4 → 444px) — already wrong, now more wrong, which
accuracy@5px cannot see but this criterion can. Only **one** turned a correct answer catastrophic
(`ho_rotation_drift_001`, 0.34 → 205px). And all three are pairs the pipeline **flags ambiguous**,
so they would route to review rather than be silently trusted. Real, but smaller than the headline.

**3. Criterion 2 — held_out ties.** A tie, not a regression (plain escalation regressed it). This is
the same structural blocker that held up `finer_hypothesis_grid`, which was later integrated once
independently validated.

## What would make it integratable

One specific thing: **independent validation on surfaces matching the frozen benchmark's family
mix.** The three surfaces used earlier (`development`, `tune_degraded`, `validate_fresh`) share
family definitions written for this campaign and are not representative — that is exactly why plain
escalation showed 7 rescued / 0 broken there and 8 / 5 on the frozen benchmark.

Concretely: generate two new datasets at fresh seeds using the **production `FAMILIES` table**
(`generator/dataset_generator.py`), not the degraded-only set, and re-run this variant there. If it
holds, it becomes a candidate for a documented gate exception on the same pattern as exceptions 1–3
— noting it would pass criteria 1, 3, 5, 6 and fail 2 (tie) and 4, against exception 3's
1, 2, 3, 6, 7 pass / 4, 5 fail.

Until then: **production stays at 77.6%.**

---

# ADDENDUM 2 — the validation that settles it

## Decisiveness-gated escalation: **DO NOT INTEGRATE.** Criterion 7 fails for real.

Validated on two datasets generated from the **production `FAMILIES` table** at fresh seeds — the
composition the change would actually face, which the earlier three surfaces were not.

| dataset | n | production | gated | Δ | rescued | broken | catastrophic |
|---|---:|---:|---:|---:|---:|---:|---|
| `prodfam_a` (seed 883021) | 136 | 0.7279 | 0.7132 | **−1.47pp** | 2 | 4 | 0.169 → 0.162 |
| `prodfam_b` (seed 517664) | 136 | 0.7132 | 0.7794 | **+6.62pp** | 10 | 1 | 0.177 → 0.118 |
| frozen benchmark | 156 | 0.7756 | 0.8013 | +2.56pp | 5 | 1 | 0.141 → 0.154 |

**The two seeds disagree in sign.** One regresses 1.5pp with three family regressions
(`dev_dense_periodic` 0.375 → 0.250, `val_multi_mat` 1.000 → 0.900,
`val_same_preset_boundary` 0.900 → 0.800); the other gains 6.6pp with the catastrophic rate
*improving* and 5 of 7 gate criteria passing.

> **Superseded 2026-08-18:** the full five-seed sweep has since completed — pooled **29 rescued /
> 10 broken, p = 0.0017**, mean **+2.79pp**, and **0 of 5 seeds pass the gate**. The two-seed
> figures below are kept as the record of what was known at the time. See
> `gated_escalation/REPORT.md` §6. The verdict is unchanged: DO NOT INTEGRATE.

Combined: **12 rescued / 5 broken**, sign test one-tailed **p = 0.072** — not significant. Pooled
accuracy across the two production-family seeds is 0.7206 → 0.7463 = **+2.58pp**, which lands
almost exactly on the frozen benchmark's +2.56pp.

**So the effect is real in expectation and unstable per draw.** That is a worse property than being
absent. A change whose sign depends on which dataset you happen to evaluate cannot be shipped, and
the variance (−1.5pp to +6.6pp across two draws of the same generator) is far too wide to justify
the runtime cost.

This is precisely the pattern the project has seen before: `pitch_aware_prominence` (#8 in
`ACCURACY_90_CAMPAIGN.md`) "passed 6/7 gate criteria on production seed, **failed second-seed
validation**", and #9 then showed the apparent gain was an artifact. Gate criterion 7 exists for
this, and it worked.

It also confirms the +2.56pp frozen figure was **not** trustworthy as an out-of-sample number: the
variant was designed after inspecting frozen failures, and the frozen benchmark sits inside the
seed-to-seed spread rather than above it.

**Also note:** criterion 6 (runtime) FAILED on `prodfam_a` at 4.47× — the margin under the 5×
ceiling is thinner than the frozen run suggested, because the flag rate varies by draw (61 of 136
flagged on `prodfam_a` versus 47 of 136 on `prodfam_b`).

## 1-D low-frequency signals: narrowly misses the bar, but is the best signal found

From `experiments/lowfreq_1d/`. The idea: **ZNCC is zero-mean and variance-normalised, so absolute
brightness and contrast are discarded before scoring.** On a Search image with vignette/gamma those
vary slowly across the field — exactly the kind of signal that could resolve "which lattice period"
along one axis. Sixteen prior attempts all modified mid/high-frequency scoring; none used what
normalisation throws away.

| signal | median margin | prefers truth |
|---|---:|---:|
| **mean brightness agreement** | +0.2983 | **15/22 (68%)** |
| contrast (std) agreement | +0.0035 | 13/22 (59%) |
| illumination envelope correlation | −0.0993 | 5/22 (23%) |
| majority of the three | — | 13/22 (59%) |

**Verdict against the pre-registered bar (>70%): does not clear it.** 68%, binomial p = 0.067 — not
significant on 22 pairs. The bar was fixed before running and is not being moved.

But in context this is the **strongest single signal found across roughly nineteen screened measures**
this session — DDIS 50%, OTSDF 50% at every α, phase-based 18%, envelope 23%. Mean brightness
agreement at 68% is the only one meaningfully above chance, and it is interesting precisely because
it is information the production scorer is structurally blind to.

**Recommendation:** worth a proper follow-up, **not tonight**. It needs (a) more than 22 pairs, and
(b) a check that Reference→Search brightness transfer is not just a family-specific artifact of the
vignette/gamma families. The `envelope` signal scoring 23% — worse than chance — is a warning that
the two images' degradation paths differ enough to make low-frequency comparisons unreliable, and
that needs explaining before anything is built on the mean-agreement result.

## Final position

**Nothing from this batch is integratable.** Production stays at **77.6%@5px**. The only change
integrated this session remains the ambiguity-threshold recalibration, which is verified,
prediction-neutral, and documented as gate exception 4.

The night's two most useful outputs are both negative results that prevented a mistake:

1. Escalation would have been shipped on a +2.56pp frozen number that **does not replicate** — one
   of the two properly-matched validation seeds regresses.
2. The 1-D framing is correct (`axis_decomposition`) but the obvious signal for exploiting it does
   not clear its bar.
