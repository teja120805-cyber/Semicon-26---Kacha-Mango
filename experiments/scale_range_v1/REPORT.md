# experiments/scale_range_v1 — A2: widen scale search to literal 9:1–11:1

## Summary

**Verdict: SAFE, narrow real benefit, technically fails the literal 7-criterion gate for a
structural reason specific to this kind of change — see "Why criteria 1/2 fail" below.**
Recommend integrating as a documented exception, not a silent bypass. Production is
untouched pending that decision.

## 1. The compliance gap

The Applied Materials help doc and sponsor pptx both state: *"Nominal 10:1 scale;
robustness tests may span ~9:1–11:1."* The shipped pipeline falls ~2 percentage points
short of that stated edge on both sides of the mechanism:

- **Dataset**: the three families that exercise scale drift (`ho_scale_drift`,
  `ch_combined_acquisition`, `ch_worst_case`) sample `extra_scale ∈ [0.93, 1.07]` — ±7%,
  not the stated ±10%.
- **Pipeline**: `candidate_generation.DEFAULT_SCALE_HYPOTHESES` searches
  `(9.2, 9.4, ..., 10.8)` — ±8%, not ±10%.

## 2. The change

Two things widen together, to the same literal edge, since testing one without the other
doesn't answer the compliance question:

- **(a) Dataset**: `_scale_range` on the three affected families widens from `(0.93, 1.07)`
  to `(0.90, 1.10)` — literal 9:1–11:1 given the generator's exact 10x base ratio.
- **(b) Pipeline**: the scale hypothesis grid widens from 9 points (9.2–10.8, step 0.2) to
  11 points (9.0–11.0, same step 0.2) — same density/step convention as the shipped
  `finer_hypothesis_grid` change, just a wider span, not a finer one.

Both changes reuse unmodified shared functions
(`generator.dataset_generator.generate_dataset(families=...)`,
`pipeline.localize.localize(scale_hypotheses=...)` — both already exposed these as
parameters for exactly this purpose) — no pipeline/generator code was forked.

## 3. Evaluation

Two datasets, following the project's established two-dataset gate convention:

1. **Frozen benchmark** (`data/`, seed 777001, n=132 gate-relevant pairs): dataset is
   *not* widened here (can't retroactively change frozen ground truth) — isolates (b)
   alone. Does widening the search grid regress anything on data that never needed it?
2. **Fresh, independently-seeded dataset** (seed 913442, validation/held_out/challenge
   only — cross_generator is external/fixed, no fresh analogue): the three scale-drift
   families *are* widened per (a). Baseline = old grid on this harder data (the real
   compliance gap); candidate = new grid on the same data (does closing the grid gap
   recover it?).

Harness integrity confirmed against this sandbox's own baseline before trusting any
comparison: `max|error_px diff| = 1.42e-14` (`outputs/gate_summary.json`).

### Frozen benchmark (grid change only, unwidened data)

| Split | Baseline @5px | Candidate @5px | Baseline cat. rate | Candidate cat. rate |
|---|---|---|---|---|
| validation | 0.900 | 0.900 | 0.050 | 0.050 |
| held_out | 0.550 | 0.550 | 0.250 | 0.250 |
| challenge | 0.656 | 0.656 | 0.250 | 0.250 |
| cross_generator | 0.800 | 0.800 | 0.150 | 0.150 |

**Exact tie on every split, every family, every metric.** This is the expected and correct
result: on data whose true scale drift never exceeds ±7%, hypotheses at 9.0/11.0 are
simply never the winning candidate — widening the grid here is a genuine no-op, not a
regression. Runtime: 264.4s → 338.4s pooled (1.28x, well under the 5x gate ceiling — an
11/9 grid-size increase, exactly as expected).

### Fresh dataset (seed 913442, dataset widened + grid widened together)

| Split | Baseline @5px | Candidate @5px | Baseline cat. rate | Candidate cat. rate |
|---|---|---|---|---|
| validation (untouched families) | 0.825 | 0.825 | 0.075 | 0.075 |
| held_out (`ho_scale_drift` widened) | 0.700 | 0.700 | 0.225 | **0.200** |
| challenge (`ch_worst_case`/`ch_combined_acquisition` widened) | 0.5625 | **0.594** | 0.344 | **0.313** |

Pooled: **70.5% → 71.4%@5px** (n=112). Per-family: `ch_worst_case` (n=8) improves
**0.500 → 0.625**; every other family ties exactly (`outputs/gate_summary.json`
`per_family`). Zero regressions anywhere, in either dataset.

## 4. Why gate criteria 1/2 fail despite zero regressions

The automated gate's criteria 1/2 require **strict pooled improvement on validation AND
held_out**. This change only ever affects 3 specific families, none of which live in
`validation` (so validation can only ever tie — it structurally cannot "improve" from this
change), and `held_out`'s one affected family (`ho_scale_drift`) improved on catastrophic
rate but not enough to flip a pair across the 5px accuracy line this particular draw. The
gate's blanket "must broadly improve" bar was designed to catch a general ranking-algorithm
regression risk; it isn't calibrated for a narrowly-scoped, spec-compliance-driven fix that
is only ever expected to move 3 of 15 families. Every criterion that *does* generalize
(no catastrophic increase, no per-family regression, acceptable runtime) passes cleanly on
both datasets, and the one family/split combination where the fix has room to show up
(`ch_worst_case`/challenge) does show a real, reproducible gain.

## 5. Recommendation

Integrate, as a **documented exception** to gate criteria 1/2 (not a silent bypass) — this
is a hard, explicitly repeated Applied Materials requirement ("robustness tests may span
~9:1–11:1"), the change is provably safe on all available evidence (2 independent datasets,
zero regressions in any split or family, 4 independent generator families' worth of
existing behavior confirmed byte-for-byte unaffected), and it closes a genuine ~2
percentage-point literal-compliance gap. Runtime cost (1.22–1.28x pooled) is small.
Final integration decision and checklist update: left to the user, per established project
discipline that production is never changed without an explicit, informed decision.
