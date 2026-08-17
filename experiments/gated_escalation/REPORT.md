# experiments/gated_escalation — the strongest lead the project has found, and still NOT integrable: INTERIM

**Date:** 2026-08-17. **Production is unchanged at 77.6%@5px.** `pipeline/`, `generator/` and
`model/` untouched.

> **STATUS: INTERIM. NOT INTEGRATED. NOT VALIDATED.**
> The settled recommendation in `experiments/BATCH_2026-08-17_REPORT.md` (Addendum 2) is **DO NOT
> INTEGRATE**, for both variants. The extended production-family validation sweep that would decide
> the question is **incomplete at the time of writing** — three of five seeds finished, two still
> running. Nothing in this report should be read as a shipped or shippable result. See §6.

Consolidated findings: `BATCH_2026-08-17_REPORT.md` §A, Addendum, and Addendum 2.

## 1. The hypothesis, stated before running

Every one of the thirteen prior attempts applied its change to **all** pairs, which forces a global
runtime budget: any per-pair cost increase must be paid 156 times. This inverts that. Production
runs unchanged, and a more expensive second pass runs **only on pairs production already flags
`ambiguous`**.

Two things from earlier in the same session made it newly possible:

1. **The recalibrated ambiguity flag** (0.92 → 0.990, gate exception 4). At 0.92 it fired on 82% of
   pairs, so gating on it saved almost nothing. It now fires on **35.3%**, which is what makes a 4×
   second pass affordable: 1 + 3(0.353) = **2.06× predicted total runtime**, against the gate's 5×
   ceiling.
2. **`experiments/reachability_verification/`**, which identified exactly which pairs a second pass
   could possibly help.

**What it escalates to is not a new idea.** `experiments/finer_hypothesis_grid/` and
`finer_grid_validation/` doubled the scale/rotation grid density, measured net rescue **+16/+6** on
two independent sets, and **were integrated**. Density was never pushed further for exactly one
reason — global runtime. Gating removes that constraint.

**Measured ceiling, computed before running.** Of the 35 frozen-benchmark failures, 30 are flagged
and **19 of those are reachable**, capping this at **+12.2pp**. Nine of the 19 have the truth at
rank 2, median deficit **0.0165 ZNCC**.

**Risk, stated up front.** 25 currently-**correct** pairs are also flagged, so escalation runs on
them too and could break them. That is the failure mode to watch — and it is the one that
materialised.

**Why this is not the rejected `center_tiebreak_v2`.** That experiment widened the tie-break epsilon
so a heuristic could pick the runner-up, and let through a 497px catastrophic regression. This
changes no ranking rule: it recomputes the candidate pool at finer granularity so the true location
can win **on its own score**. If it still loses, nothing is overridden.

**Null control:** `--factor 1` escalates with the production grid and must reproduce production
bit-for-bit. **Verified on all three tuning surfaces, not assumed.**

## 2. Phase 1 — three tuning surfaces, all positive, all with zero breaks

| surface | n | baseline | factor 3 | Δ | rescued | broken |
|---|---:|---:|---:|---:|---:|---:|
| `development` | 24 | 0.7083 | 0.7917 | +8.3pp | 2 | **0** |
| `tune_degraded` (seed 314159) | 40 | 0.7000 | 0.7750 | +7.5pp | 3 | **0** |
| `validate_fresh` (seed 271828, held back) | 40 | 0.7250 | 0.7750 | +5.0pp | 2 | **0** |
| **combined** | **104** | | | **+6.7pp** | **7** | **0** |

7–0 gives a sign test of **p = 0.008**, stronger than the p = 0.031 that gate exception 3 cleared.
On that evidence a frozen run was warranted. `factor` was fixed on `development` before the other
surfaces were run; factor 4 was rejected there for saturating while breaching the runtime ceiling.

## 3. Phase 2 — frozen run #1 (plain escalation): the number that counts disagrees

| | production | gated escalation |
|---|---:|---:|
| accuracy@5px | 0.7756 | **0.7949 (+1.92pp)** |
| catastrophic (>50px) | 0.1410 | **0.1667 (worse)** |
| rescued / broken | — | **8 / 5** |
| runtime | 1.00× | 3.74× (ceiling 5×) |

| split | before | after |
|---|---:|---:|
| `development` | 0.7083 | 0.7917 |
| `validation` | 0.9250 | 0.9500 |
| **`held_out`** | 0.7000 | **0.6750** |
| `challenge` | 0.7188 | 0.7500 |
| `cross_generator` | 0.8000 | 0.8000 |

**Integration gate: FAILED — 4 of 7 pass.**

| criterion | result |
|---|---|
| 1 improves validation | PASS |
| 2 improves held_out | **FAIL** (0.700 → 0.675) |
| 3 improves/ties cross_generator | PASS |
| 4 no catastrophic increase | **FAIL** (22 → 26 pairs) |
| 5 no per-family regression | **FAIL** (`ho_heavy_noise` 0.800 → 0.700) |
| 6 acceptable runtime | PASS (3.74× of 5×) |
| 7 stable across seeds | PASS (3 surfaces) — later invalidated, see §4 |

The rescues are dramatic — `ho_scale_drift_006` 735px → 0.46px, `ch_worst_case_005` 631px → 0.59px,
`ch_barrel_charging_003` 407px → 1.9px. But five previously-correct pairs break, **four of them into
new catastrophic errors**: `ho_heavy_noise_006` 0.36 → 119px, `ho_rotation_drift_001` 0.34 → 205px,
`ho_scale_drift_007` 0.34 → 661px, `ch_worst_case_007` 0.26 → 54px.

**Verdict: do not integrate.** Trading a 0.36px answer for a 205px one is the wrong trade for an
inspection tool, and it happens four times. +1.92pp pooled does not pay for a catastrophic rate
rising 14.1% → 16.7%.

## 4. Why the three tuning surfaces were misleading

`tune_degraded` and `validate_fresh` share the same seven family definitions, written for this
campaign and deliberately weighted toward degraded acquisitions. They do not contain
`ho_heavy_noise`, `ho_vignette_gamma` or `val_linewidth_bias` in the frozen benchmark's proportions.

Three surfaces agreeing meant far less than it appeared, because two of them were siblings:
**independent seeds are not the same as independent family composition.** That is the same error
that produced the 74% reachability overstatement corrected in `REACHABILITY_CAMPAIGN.md`, and it is
why criterion 7's PASS in §3 does not stand.

It is also the direct cause of the 7 rescued / 0 broken on those surfaces against 8 / 5 on the
frozen benchmark.

## 5. Phase 3 — frozen run #2, decisiveness-gated

**The change.** Accept the escalated answer only when its candidate pool is **more decisive** than
production's — the same `localize._decisiveness` comparison production already uses to choose
between PSF arms (gate exception 3). Otherwise the production answer stands byte-for-byte.
**No free parameters:** a strict comparison between two numbers, nothing to tune or select.

**Disclosure, stated plainly.** The idea was prompted by inspecting which pairs plain escalation
broke, and this is the **second frozen run** for this line of work. It is not parameter mining —
there are no parameters — but the +2.56pp below is **not a clean out-of-sample number**, and that is
the single most important caveat on it.

| | production | plain escalation | **decisiveness-gated** |
|---|---:|---:|---:|
| accuracy@5px | 0.7756 | 0.7949 (+1.92pp) | **0.8013 (+2.56pp)** |
| rescued / broken | — | 8 / 5 | **5 / 1** |
| catastrophic (>50px) | 0.1410 | 0.1667 | 0.1538 |
| runtime | 1.00× | 3.74× | 3.74× |
| gate criteria passed | — | 4/7 | **4/7 (different, better set)** |

The gate does real work: escalation is accepted on only **16 of 55** flagged pairs (29%). Among the
39 rejected, escalation *would have* rescued 3 and broken 4 — so the rule correctly filtered a
net-negative set, at the cost of 3 genuine rescues, including `ch_worst_case_005` (631px → 0.59px).

| criterion | plain | decisiveness-gated |
|---|---|---|
| 1 improves validation | PASS | PASS (0.925 → 0.950) |
| 2 improves held_out | FAIL (regression) | **FAIL (tie, 0.700 → 0.700)** |
| 3 cross_generator | PASS | PASS |
| 4 no catastrophic increase | FAIL | **FAIL (0.1410 → 0.1538)** |
| 5 no per-family regression | FAIL | **PASS** |
| 6 runtime | PASS | PASS (3.74× of 5×) |
| 7 stable across seeds | PASS* | **FAIL — not validated** |

\* credited to three surfaces that later proved to be siblings (§4).

Nuance worth recording on criterion 4: of the three pairs that became catastrophic, **two were
already failures** (`dev_single_mat_006` 11.9 → 732px; `ho_vignette_gamma_005` 25.4 → 444px) —
already wrong, now more wrong, which accuracy@5px cannot see but this criterion can. Only **one**
turned a correct answer catastrophic (`ho_rotation_drift_001`, 0.34 → 205px). All three are pairs
the pipeline flags `ambiguous`, so they would route to review rather than be silently trusted. Real,
but smaller than the headline.

## 6. Phase 4 — production-family validation: INCOMPLETE, and the two finished seeds disagree in sign

Validation datasets are generated from the **production `FAMILIES` table**
(`generator/dataset_generator.py`) at fresh seeds — the same 15 families and per-family counts as
the frozen benchmark, i.e. the composition the change would actually face, which the §2 surfaces
were not. `cross_generator` has no generated analogue, so these carry 136 pairs against the frozen
benchmark's 156.

Five seeds are defined in `make_production_family_data.py`: `prodfam_a` (883021), `prodfam_b`
(517664), `prodfam_c` (240719), `prodfam_d` (661438), `prodfam_e` (105293).

| dataset | n | production | gated | Δ | rescued | broken | catastrophic | state |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `prodfam_a` (seed 883021) | 136 | 0.7279 | 0.7132 | **−1.47pp** | 2 | 4 | 0.169 → 0.162 | complete |
| `prodfam_b` (seed 517664) | 136 | 0.7132 | 0.7794 | **+6.62pp** | 10 | 1 | 0.177 → 0.118 | complete |
| `prodfam_c` (seed 240719) | — | — | — | **+3.68pp** | 7 | 2 | — | complete (interim) |
| `prodfam_d` (seed 661438) | — | — | — | — | — | — | — | **still running** |
| `prodfam_e` (seed 105293) | — | — | — | — | — | — | — | **still running** |
| frozen benchmark | 156 | 0.7756 | 0.8013 | +2.56pp | 5 | 1 | 0.141 → 0.154 | complete |

**What is settled, from the two seeds in the batch report.** The two seeds **disagree in sign**. One
regresses 1.47pp with three family regressions (`dev_dense_periodic` 0.375 → 0.250, `val_multi_mat`
1.000 → 0.900, `val_same_preset_boundary` 0.900 → 0.800); the other gains 6.62pp with the
catastrophic rate *improving* and 5 of 7 gate criteria passing. Combined: **12 rescued / 5 broken**,
sign test one-tailed **p = 0.072 — not significant**. Pooled accuracy across the two is 0.7206 →
0.7463 = **+2.58pp**, landing almost exactly on the frozen benchmark's +2.56pp.

**So the effect is real in expectation and unstable per draw.** That is a worse property than being
absent. A change whose sign depends on which dataset you happen to evaluate cannot be shipped, and a
−1.5pp to +6.6pp spread across two draws of the *same generator* is far too wide to justify the
runtime cost. It also confirms the +2.56pp frozen figure was **not** trustworthy as an out-of-sample
number: the variant was designed after inspecting frozen failures, and the frozen benchmark sits
*inside* the seed-to-seed spread rather than above it.

This is the pattern the project has seen before: `pitch_aware_prominence` (#8 in
`ACCURACY_90_CAMPAIGN.md`) passed 6/7 gate criteria on the production seed and **failed second-seed
validation**, with #9 then showing the apparent gain was an artifact. Gate criterion 7 exists for
this, and it worked.

**What is interim.** `prodfam_c` has since completed at **+3.68pp, 7 rescued / 2 broken** — positive,
consistent with `prodfam_b` in sign. Its accuracy, catastrophic-rate and per-family breakdown are
not yet consolidated into the batch report, and are not quoted here rather than guessed. Simple
arithmetic on the stated counts gives **19 rescued / 7 broken across the three completed seeds**
(12/5 from a+b, plus 7/2). **No significance test is recomputed on that tally**: two of five seeds
are outstanding, and computing a p-value now, then again after `prodfam_d` and `prodfam_e` land,
would be optional-stopping — the exact error criterion 7 exists to catch.

**`prodfam_d` and `prodfam_e` were still running at the time of writing.** The sweep is incomplete
and no conclusion is drawn from it.

**Runtime — criterion 6 also failed once.** On `prodfam_a` criterion 6 **FAILED**, with the
dataset-level multiplier at **4.47×**. The margin under the 5× ceiling is thinner than the frozen
run's 3.74× suggested, because the flag rate varies by draw: **61 of 136** flagged on `prodfam_a`
versus **47 of 136** on `prodfam_b`. The gate applies the ceiling per split
(`evaluation/benchmark.py`, `MAX_RUNTIME_MULTIPLIER = 5.0`, checked with `all(...)` across splits),
so a dataset-level figure under 5× does not by itself pass. This is a second, independent blocker
alongside criterion 7, and it was not visible on the frozen benchmark.

## 7. Status

- **NOT INTEGRATED. NOT VALIDATED.** Production stays at **77.6%@5px**.
- **Plain escalation: rejected.** +1.92pp, but 4 of 5 breaks turn sub-pixel answers into
  catastrophic errors; gate 4/7.
- **Decisiveness-gated escalation: rejected on the evidence available.** The closest any change has
  come (+2.56pp frozen, 5/1), but criterion 7 fails for real — one of two properly-matched
  validation seeds regresses.
- **Validation sweep incomplete:** `prodfam_a`, `prodfam_b`, `prodfam_c` complete; `prodfam_d` and
  `prodfam_e` still running. The extra seeds may narrow the variance estimate; they cannot retire
  the sign disagreement already observed on `prodfam_a`.
- **Criterion 6 is a second live blocker** (`prodfam_a`, 4.47×), driven by draw-to-draw variation in
  the flag rate.
- **What would change the verdict:** consistent sign across all five production-family seeds, plus
  a runtime margin that holds when the flag rate runs high. Anything less leaves a change whose sign
  depends on the draw, which cannot ship.

## 8. What survives regardless of the verdict

- **Gating on the recalibrated `ambiguous` flag works as designed.** Escalation touched only flagged
  pairs; the factor-1 null control reproduced production bit-for-bit on all three tuning surfaces.
  The mechanism is sound; the payload is not.
- **The mechanism behind the rescues is understood, not lucky.** `experiments/axis_decomposition/`
  shows a 2% scale-quantisation error accumulates ~20px across a 1000px Search image against a ~14px
  lattice pitch — more than one full period. Denser hypotheses cut that below half a period.
  `experiments/hypothesis_ensemble/` reaches the same conclusion from the other side: adding
  hypotheses helps because the truth matches only near the exact hypothesis, while aggregating over
  hypotheses rewards the decoy.
- **The methodological lesson is the durable output:** independent seeds are not independent family
  composition. Any future validation in this project should be generated from the production
  `FAMILIES` table, not from campaign-authored family sets.

## Reproduce

Tuning surfaces, frozen runs, then production-family validation, in order:

```
python -m experiments.gated_escalation.run --surface development
python -m experiments.gated_escalation.run --surface tune_degraded --factors 1,2,3
python -m experiments.gated_escalation.run --surface validate_fresh --factors 1,3
python -m experiments.gated_escalation.frozen
python -m experiments.gated_escalation.frozen_decisive
python -m experiments.gated_escalation.make_production_family_data
python -m experiments.gated_escalation.validate_prodfam --dataset prodfam_a
```

`validate_prodfam` takes one `--dataset` per invocation; repeat for `prodfam_b` … `prodfam_e`.
`make_production_family_data` must run first — it generates all five seeds.
