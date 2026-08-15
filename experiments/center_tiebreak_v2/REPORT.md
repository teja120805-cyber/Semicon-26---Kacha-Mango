# Experiment: a genuinely-working centre tie-break (v2)

**Status: REJECTED.** Fails the mandatory integration gate on criteria 2 (`improves_held_out`),
4 (`no_catastrophic_increase`), and 7 (`stable_across_seeds`) on the official frozen-benchmark
comparison — confirmed, not contradicted, by an independent fresh-dataset replication, which fails
different criteria (4 passes there, 5 fails) but tells the same underlying story. Production
(`pipeline/ranking.py`, `pipeline/localize.py`) is **unchanged**; `reports/TIE_BREAK_IMPLEMENTATION.md`'s
epsilon=1e-6 definition remains what ships.

## 1. What this experiment is

`reports/TIE_BREAK_IMPLEMENTATION.md` documents the one prior attempt at a non-inert centre
tie-break: reusing `AMBIGUITY_THRESHOLD=0.92` (a second-best/best ZNCC *ratio*) collapsed pooled
accuracy@5px from 71.2% to 33.3%, because ZNCC scores decay gradually across genuinely distinct
wrong-location candidates in this domain — the ratio treated large swaths of clearly-not-tied
candidates as tied. The definition that shipped instead (`TIE_SCORE_EPSILON=1e-6`) is provably
inert: 0 predictions changed on the frozen benchmark.

This experiment asks the question the task set out honestly: is there a threshold that sits
between those two extremes — non-trivial, but not so wide it recreates the 0.92 disaster? It builds
and evaluates one, as an isolated experiment (`experiments/center_tiebreak_v2/`), importing every
pipeline function unmodified (`pipeline/candidate_generation.py`, `pipeline/ranking.py`,
`pipeline/refinement.py` — byte-for-byte untouched; see `harness.py`). The only thing that varies
between "baseline" and "candidate" anywhere in this experiment is the `tie_score_epsilon` value
passed into the shared, unmodified `ranking.apply_center_tiebreak` — the same "same function,
different argument" pattern `experiments/finer_hypothesis_grid/` established as this project's
precedent for evaluating a real pipeline-behavior change without forking pipeline code.

## 2. Threshold investigation and the value chosen

`investigate_gaps.py` runs the unmodified candidate-generation → deduplication → classical-ranking
stack (no tie-break applied yet) over the **full 156-pair frozen benchmark** — `development`,
`validation`, `held_out`, `challenge`, `cross_generator` — not just `development` (n=24), which is
what the rejected 0.92 attempt used before generalizing a threshold from too small a sample
(`TIE_BREAK_IMPLEMENTATION.md`'s own stated smallest-gap figure, ~3.8e-4, came only from that
24-pair slice). Because `candidate_generation.deduplicate_by_location` already enforces >10px
separation between kept candidates (confirmed here: minimum observed top1/top2 distance across all
156 pairs was 10.05px), every gap measured is between two **genuinely different locations** — the
"is this really the same site detected twice" confound is structurally impossible post-dedup.

**Key findings, n=156:**

| | n | abs(top1−top2 score) — p5 | p25 | median | p75 |
|---|---:|---:|---:|---:|---:|
| Currently wrong (pre-tiebreak) | 42 | 0.00056 | 0.00148 | 0.00296 | 0.00572 |
| Currently correct (pre-tiebreak) | 114 | 0.00254 | 0.00868 | 0.01754 | 0.06964 |

The two populations separate, but not cleanly at the very bottom: the single smallest gap among
any of the 114 **currently-correct** pairs anywhere in the benchmark is **0.00104**
(`ch_speckle_saltpepper_007`) — the "risk floor." Sweeping the *real, unmodified*
`apply_center_tiebreak` across candidate thresholds (not a re-implementation — the exact
production function, called once per pair per threshold on the already-computed ranked list):

| Threshold | Pairs that fire | Currently-correct pairs touched (risk) | `genuine_ambiguity` opportunities touched |
|---:|---:|---:|---:|
| 1e-6 (shipped) | 0 | 0 | 0 |
| 0.0005 | 1 | 0 | 0 |
| **0.001** | **6** | **0** | **2 / 10** |
| 0.002 | 11 | 3 | 2 / 10 |
| 0.005 | 23 | 7 | 3 / 10 |
| 0.01 | 37 | 17 | 4 / 10 |
| 0.02 | 48 | 26 | 5 / 10 |
| 0.05 | 79 | 55 | 5 / 10 |
| 0.08 | 94 | 64 | 6 / 10 |

Risk (touching a currently-correct pair) grows far faster than opportunity (touching a documented
`genuine_ambiguity` near-tie — GT's own candidate present in the pool but narrowly outscored, per
`reports/ACCURACY_FORENSICS.md`'s taxonomy) as the threshold widens.

**Chosen: `CANDIDATE_TIE_SCORE_EPSILON = 0.001`** — the largest round value that stays strictly
below the empirical risk floor (0.00104) while sitting three orders of magnitude above the inert
1e-6 floor and non-trivially engaging 2 of the 10 documented near-ties. This is an **absolute**
ZNCC-score margin, the same form this project's own forensics already use
(`AMBIGUITY_MARGIN=0.02` in the candidate/ranking failure taxonomy), not a ratio — the rejected 0.92
attempt's failure mode was specifically that a *ratio* is scale-dependent in a way that made it far
too generous at the ~0.6–0.95 score levels this pipeline actually produces (a 0.92 ratio implies an
absolute gap of ~0.05–0.07 at those levels, 50–70x wider than the value adopted here). Relative
(fraction-of-top-score) margins were also computed (`rel_gap12` in `investigate_gaps.py`'s output)
and found nearly equivalent to absolute margins in this score regime, so the simpler, precedent-
consistent absolute form — which requires zero new pipeline logic, just a different argument to the
already-parameterized `apply_center_tiebreak` — was kept. **This value was fixed before
`run_experiment.py` was ever run — it was not tuned against the gate result below.**

## 3. Evaluation methodology

Exactly the shape of process `experiments/finer_hypothesis_grid/` used:

1. **Frozen benchmark** (`data/` — `validation`/`held_out`/`challenge`/`cross_generator`, n=132; the
   official gate comparison). Baseline is verified, not assumed, to reproduce the already-shipped
   `outputs/reports/per_pair_results.csv` byte-for-byte before anything is compared against it:
   max\|error_px diff\| = 5.7e-14 across all 132 pairs (float noise) — the harness is confirmed to
   match production exactly at the shipped epsilon.
2. **A genuinely fresh, independently-seeded dataset** (seed `647301` — distinct from production's
   `777001` and every other experiment's seed), `validation`/`held_out`/`challenge` only
   (`cross_generator` is external/fixed, no fresh analogue — same reasoning as
   `finer_hypothesis_grid`), n=112. Both baseline and candidate are run fresh on this data, to check
   the frozen-benchmark result isn't specific to one particular random draw.
3. `evaluation.benchmark.run_integration_gate` is called on both. This is a **deterministic,
   non-stochastic pipeline change** (no training, no RNG) — like `rotation_scale`/`periodicity`/
   `wider_candidate_pool` before it, criterion 7 has no literal "training seed" to check. Those three
   experiments asserted `seeds_agree=True` manually with a documented justification. Here,
   `seeds_agree` is instead **derived from actual agreement** between the frozen and fresh gate
   results on every criterion both datasets can evaluate (1, 2, 4, 5, 6 — criterion 3 is
   `cross_generator`-only and has no fresh analogue) — stricter than the prior precedent, since this
   experiment already had a second independent dataset available to check against rather than
   asserting stability by fiat.

## 4. Results

### 4.1 Gate criteria table

| # | Criterion | Frozen (official) | Fresh (informational replication) |
|---|---|---|---|
| 1 | Improves `validation`@5px | ✅ **PASS** (90.0% → 92.5%) | ✅ PASS (87.5% → 95.0%) |
| 2 | Improves `held_out`@5px | ❌ **FAIL** (60.0% → 60.0%, tied) | ❌ FAIL (70.0% → 67.5%, regressed) |
| 3 | Improves/ties `cross_generator`@5px | ✅ **PASS** (80.0% → 85.0%) | n/a — external, no fresh analogue |
| 4 | No catastrophic (>50px) increase, any split | ❌ **FAIL** (`held_out` 20.0% → 25.0%) | ✅ PASS (no split increased) |
| 5 | No per-family regression (acc@5px) | ✅ **PASS** (0 / 13 families) | ❌ FAIL (`ho_scale_drift` 80.0% → 70.0%) |
| 6 | Runtime ≤ 5× baseline | ✅ PASS | ✅ PASS |
| 7 | Stable across an independent seed/dataset | ❌ **FAIL** (disagrees with fresh on criteria 4 & 5) | ❌ FAIL (same disagreement) |
| | **Overall** | ❌ **REJECT** | ❌ **REJECT** |

Both datasets independently reject the change — on **different specific criteria** (frozen fails on
new catastrophic severity; fresh fails on a new per-family regression), which is itself informative:
this isn't one unlucky dataset producing a fluke failure, it's the same underlying mechanism (below)
surfacing real harm on two independent random draws, just through different specific pairs.

### 4.2 Per-split accuracy@5px (frozen benchmark, n=132)

| Split | n | Baseline | Candidate | Δ |
|---|---:|---:|---:|---:|
| `validation` | 40 | 90.0% | 92.5% | +2.5pp |
| `held_out` | 40 | 60.0% | 60.0% | 0.0pp (tied) |
| `challenge` | 32 | 75.0% | 75.0% | 0.0pp (tied) |
| `cross_generator` | 20 | 80.0% | 85.0% | +5.0pp |
| **Pooled** | **132** | **75.8%** | **77.3%** | **+1.5pp** |

### 4.3 Full metric set, pooled (frozen benchmark)

| Metric | Baseline | Candidate |
|---|---:|---:|
| Accuracy@1px | 68.9% | 70.5% |
| Accuracy@2px | 75.0% | 76.5% |
| Accuracy@4px | 75.0% | 76.5% |
| Accuracy@5px | 75.8% | 77.3% |
| Median error | 0.335px | 0.335px |
| **Mean error** | **34.91px** | **38.56px** ⚠️ |
| P90 error | 87.07px | 95.50px ⚠️ |
| **P95 error** | **250.78px** | **303.25px** ⚠️ |
| Max error | 725.05px | 725.05px (unchanged) |
| Failure rate >10px | 23.5% | 22.0% |
| **Failure rate >50px** | **13.6%** | **14.4%** ⚠️ |

Pooled accuracy@1–5px all improve. Every tail metric (mean, P90, P95, >50px failure rate) gets
*worse* — the pooled accuracy number alone would have told a misleadingly positive story on its own,
exactly the "pooled-only number can hide a family that got worse" risk the task named up front.

Same pattern replicates on the fresh dataset: accuracy@5px 69.6%→71.4% (+1.8pp) while mean error
essentially flat (45.0px→44.3px, this time slightly better) but `held_out` accuracy itself
regressed (70.0%→67.5%) via a genuine 5px-boundary break, not a severity increase — a different
manifestation of the same underlying instability.

### 4.4 The mechanism: rescue vs. collateral damage

| | Frozen (n=132) | Fresh (n=112) |
|---|---:|---:|
| Winners changed by the tie-break | 6 | 4 |
| Rescued (wrong→correct) | 2 | 3 |
| Broken (correct→wrong) | 0 | 1 |
| Net rescue | +2 | +2 |
| **New catastrophic (>50px) failures** | **2** | **0** |
| Catastrophic failures rescued | 1 | 1 |

On the frozen benchmark, the 6 pairs the tie-break touched were exactly the 6 predicted by
`investigate_gaps.py`'s threshold sweep (`val_same_preset_boundary_008`, `ho_heavy_noise_002`,
`ho_scale_drift_000`, `ho_vignette_gamma_005`, `cross_generator_00004`, `cross_generator_00012`) —
independent confirmation the harness and the investigation script agree. Of those:

- **`val_same_preset_boundary_008`** and **`cross_generator_00004`** are 2 of the benchmark's 10
  documented `genuine_ambiguity` cases (GT's own candidate present, narrowly outscored) — **both
  were rescued**. The mechanism works exactly as intended on the cases it targets: **2/2 hit rate**.
- **`ho_heavy_noise_002`** and **`cross_generator_00012`** were already-wrong `candidate_generation`/
  `candidate_ranking` failures (GT's own candidate not competitively present) — the tie-break
  reshuffled which *wrong* location is reported, but the outcome was already wrong either way,
  no rescue was ever possible.
- **`ho_scale_drift_000`** and **`ho_vignette_gamma_005`** were also already-wrong, but the
  reshuffle pushed them from a moderate error into >50px catastrophic territory. Concretely:
  `ho_vignette_gamma`'s family-level mean error nearly doubled (50.9px → 100.6px), P90 jumped
  71px → 486px, P95 jumped 277px → 504px — **while the family's accuracy@5px stayed at 80.0% in
  both directions**, because this pair was already wrong at the 5px tolerance before and after. A
  pure accuracy@5px view (including `run_integration_gate`'s own criterion 5, which only checks
  accuracy@5px) is structurally blind to this — it only surfaced at all because criterion 4 checks
  the 50px catastrophic boundary separately.

**Why this happens, and why it isn't fixable by re-tuning the threshold**: "closest to the Search-
image centre" has no informative relationship to "closer to the true location" among candidates
that are genuinely different sites (confirmed post-dedup, section 2). The threshold-selection
process in section 2 successfully avoided disturbing any *currently-correct* pair (0 at risk at
0.001, by construction) — but it has no mechanism to avoid, and cannot avoid, making an
*already-wrong* pair's error **larger** when it reshuffles among multiple wrong candidates, because
centre-proximity is simply uncorrelated with correctness for this class of pair. Any threshold wide
enough to catch a meaningful share of genuine near-ties (opportunity) will, in this benchmark,
inevitably also touch several already-wrong pairs where the reshuffle can go either direction in
severity — and empirically, on both independent datasets tested, it goes far enough in the wrong
direction often enough to fail the gate.

## 5. Verdict

**REJECT.** Fails criteria 2, 4, and 7 on the official frozen-benchmark comparison, independently
confirmed as a real (not fluke) effect by a fresh, independently-seeded dataset that fails a
different but related pair of criteria (4 passes there but 5 fails, for the mirror-image reason).
Per this project's integration rule — every criterion must pass, and a failure is reported honestly
rather than tuned away — **this is not integrated**.

This is a genuinely different outcome from `finer_hypothesis_grid`'s "near-miss": that experiment
failed on a ceiling effect (`validation` already at 90%, nothing left to improve) with **no
catastrophic-severity cost anywhere**. This experiment fails because of an actual harm mechanism —
new catastrophic failures on one dataset, a new per-family regression on the other — that a
pooled-accuracy-only read would have missed entirely (pooled accuracy@5px *improves* on both
datasets, 75.8%→77.3% and 69.6%→71.4%). The mechanism (section 4.4) also explains *why* no
different threshold value is likely to fix this without giving up almost all of the opportunity:
the same score-decay property that made the 0.92 ratio catastrophic operates, at smaller scale, at
every threshold above the inert floor — there is no threshold that captures a meaningful share of
genuine near-ties while remaining risk-free for already-wrong pairs, because the tie-break's
selection rule (centre-proximity) is fundamentally uninformative about which wrong candidate is
"more correct."

**On the original spec-compliance motivation**: the shipped `TIE_SCORE_EPSILON=1e-6` definition in
`pipeline/ranking.py` already implements the letter of the Applied Materials tie-break requirement
correctly — it resolves genuine numerical ties (which do occur, just not on this benchmark) by
centre-proximity rather than arbitrary sort order — while remaining provably inert on real,
non-tied ZNCC score distributions. Widening what counts as "several valid matches" beyond exact
equality, as this experiment set out to do, measurably trades pipeline accuracy for a broader
reading of that requirement on data that doesn't clearly need it. That is a real tension worth the
team's awareness, not something this report resolves unilaterally — the honest empirical finding is
simply that this particular resolution of it does not clear the accuracy bar this project holds
every other pipeline change to.

## 6. Production impact

**None.** `pipeline/ranking.py`, `pipeline/localize.py`, and `pipeline/test_ranking.py` are
byte-for-byte unchanged. `reports/TIE_BREAK_IMPLEMENTATION.md` is unchanged and remains accurate:
the shipped epsilon=1e-6 tie-break is still what runs in production, still provably inert on the
frozen benchmark. This experiment's own code (`experiments/center_tiebreak_v2/`) only ever passes a
different `tie_score_epsilon` value into the unmodified, shared `apply_center_tiebreak` function —
no pipeline code was forked or edited to produce any result in this report.

## 7. Artifacts

- `investigate_gaps.py` / `outputs/score_gap_investigation.csv` / `outputs/score_gap_summary.json` —
  the threshold derivation (section 2).
- `run_experiment.py` / `outputs/experiment_results.json` — the full gate evaluation (sections 3–4),
  including per-split and per-structural-family breakdowns on both datasets beyond what's excerpted
  above.
- `outputs/per_pair_frozen_baseline.csv`, `per_pair_frozen_candidate.csv`,
  `per_pair_fresh_baseline.csv`, `per_pair_fresh_candidate.csv` — full per-pair predictions/errors.
- `harness.py` — the shared instrumented-localization function both the investigation and the
  evaluation call (only unmodified `pipeline/` functions inside).
