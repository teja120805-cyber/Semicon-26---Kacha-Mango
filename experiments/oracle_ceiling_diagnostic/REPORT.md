# experiments/oracle_ceiling_diagnostic — DIAGNOSTIC (not a pipeline candidate)

## Summary

This diagnostic measures, for every pair the production baseline gets wrong, the best ZNCC
score achievable **at the true location** under an ideal (scale, rotation) warp, and compares it
against the best score achievable at the location the pipeline actually picked. It answers a
question the 90%-accuracy campaign left unmeasured, and the answer **overturns that campaign's
headline conclusion.**

Three findings, in descending order of importance:

1. **The campaign's "central, repeatedly-confirmed finding" is wrong.** It states that when the
   classical winner is wrong, "the wrong candidate's raw ZNCC score is *genuinely, substantially*
   higher than the true location's," citing a 4x gap (true location 0.187 vs. decoy 0.775). **That
   gap does not exist anywhere in the benchmark.** Across all 39 evaluable failures the true
   location scores **0.612–0.937** under an ideal warp (median 0.86), and the largest
   decoy/truth ratio is **1.048**, not 4x. Every failure is a **near-tie**.
2. **Warp search is not the answer either** — but it isn't dead. The production grid already
   reaches within a median **0.011** ZNCC of the ideal-warp ceiling at the true location.
3. **The near-ties are detectable without ground truth**, and that is the actionable result. A
   pool-internal top-1-vs-runner-up score gap separates correct from incorrect pairs by ~7x in
   the median and gives **95% failure recall while flagging only 49% of pairs**.

Production is untouched. This experiment reads ground truth by construction, so it can never be
a localization method — it exists only to direct the next round of work.

## 1. Method

For each of the 40 failing pairs (`error_px > 5`) plus 20 currently-correct control pairs:

- `grid_gt` — best ZNCC at ground truth over the **production** 11x9 hypothesis grid.
- `oracle_gt` — best ZNCC at ground truth over a **dense** sweep: scale 8.60–11.40 step 0.05,
  rotation −6.0°–+6.0° step 0.25° (2,793 warps), then a local refine at step 0.01 / 0.05°. That
  is ~4x finer in scale and ~12x finer in rotation than production, and wider on both axes, so
  no conclusion here can be an artifact of the sweep span.
- `oracle_win` — the same dense sweep evaluated at the pipeline's **wrong** predicted location.

Scores are taken over template placements whose center lands within 4px of the target, so any
"win" found here would still count as correct under the 5px tolerance. Cropping the search image
around each target is exact rather than approximate: `TM_CCOEFF_NORMED` normalizes within each
template-sized window independently, so cropped scores equal the corresponding entries of the
full-image score map.

The sweep decomposes `matching.build_template` into a cached resize plus a per-rotation warp for
speed. **This is asserted bit-identical to the production function** (`oracle.verify_equivalence`,
run at startup) so no number below depends on a shortcut.

## 2. Result: every failure is a near-tie

`oracle_margin = oracle_gt − oracle_win`. Positive means the true location wins under an ideal
warp (a **geometric** failure — better warp search would fix it); negative means the decoy still
wins even under a perfect warp (a **photometric** / genuine-self-similarity failure).

| | value |
|---|---:|
| decoy still wins under ideal warp (photometric) | 29 / 39 |
| true location wins under ideal warp (geometric) | 10 / 39 |
| mean margin | −0.0078 |
| median margin | −0.0076 |
| **most extreme margin, either direction** | **0.0365** |

Distribution of `|margin|` across all 39 failures:

| threshold | pairs within it |
|---|---:|
| < 0.005 | 11 / 39 (28%) |
| < 0.010 | 22 / 39 (56%) |
| < 0.020 | 31 / 39 (79%) |
| < 0.030 | 35 / 39 (90%) |
| **< 0.050** | **39 / 39 (100%)** |

And directly against the campaign's claim:

| check | result |
|---|---|
| failures with `oracle_gt < 0.5` | **0** |
| failures with `oracle_gt < 0.6` | **0** |
| minimum `oracle_gt` over all failures | **0.6122** |
| maximum `oracle_win / oracle_gt` | **1.048** |

The pipeline is not blind to the true location. It is losing coin-flips at the 0.005–0.02 level
on a score whose right-vs-wrong separation is about 0.02 wide.

**Where the campaign's 0.187 came from.** That number was produced inside
`pyramid_periodicity_search`'s own coarse/blurred proposal stage, not measured at full resolution
under a fitted warp. It is an artifact of that experiment's internals rather than a property of
the data. The campaign then generalized it — from a single pair — into a conclusion that closed
off the entire re-ranking direction. There *is* a real tail where the true location scores much
worse (see §4), so the observation wasn't invented; it was over-generalized from the worst case
to the whole population.

## 3. Warp headroom is small — which cuts both ways

`gt_headroom = oracle_gt − grid_gt`: how much ZNCC a perfect continuous warp would add at the
true location over what the production grid already achieves.

| set | median headroom | mean |
|---|---:|---:|
| failures | 0.0112 | 0.0200 |
| controls | 0.0080 | 0.0115 |

The production grid is already near the achievable ceiling, and failures are barely different
from successes here. So denser warp search is **not** a large untapped source of score.

But this does **not** make the grid direction dead, and it resolves what looked like a
contradiction: `finer_hypothesis_grid` produced real gains (+5pp held_out, +6.2pp challenge)
while adding only ~0.01 of score. Both are true *because* everything is a near-tie — when the
margin between right and wrong is 0.005–0.02, a 0.011 score gain is enough to flip a meaningful
number of pairs. That also explains why those gains were fragile and seed-sensitive: they are
won at the noise floor of the score.

## 4. What the 40 failures are actually made of

Running the unmodified production candidate pool over all 156 pairs and asking where ground
truth sits in it (no ground truth used to *build* the pool):

| group | n | share | what is broken |
|---|---:|---:|---|
| No candidate within 5px of GT at all | **18** | 45% | **Candidate generation** — the true location is never proposed. Closest candidate is a median 21px away (range 5.5–90px). |
| GT is the runner-up (rank 1), loses by a hair | **9** | 22.5% | **Tie-break** — found, ranked second, 8 of 9 within 0.01 of the winner. |
| GT in pool at rank 3–43, larger deficit | **13** | 32.5% | **Scoring** — median deficit 0.16, max 0.61. This is the tail the campaign mistook for the whole. |

The 45% group is a direct contradiction of `wider_candidate_pool`'s conclusion that candidate
discovery is not the bottleneck. That experiment showed *widening the existing grid's pool* is a
no-op under arg-max ranking — which is true, and different from the claim that the true location
is always discoverable. For nearly half of all failures it is simply never proposed.

## 5. The near-ties are detectable without ground truth

For each pair: `gap = top_score − best_score_at_a_location_>10px_away` (the same distinctness
radius production's `deduplicate_by_location` uses). This uses only the candidate pool.

| set | median gap | quartile |
|---|---:|---|
| correct pairs (n=116) | **0.0188** | q25 0.0087, q10 0.0035 |
| wrong pairs (n=40) | **0.0026** | q75 0.0047, q90 0.0078 |

Sweeping the threshold as a failure detector:

| threshold | pairs flagged | failures caught | recall | precision |
|---|---:|---:|---:|---:|
| 0.005 | 49 (31%) | 32 / 40 | 80% | 65% |
| **0.010** | **76 (49%)** | **38 / 40** | **95%** | **50%** |
| 0.020 | 97 (62%) | 39 / 40 | 97.5% | 40% |
| 0.050 | 121 (78%) | 40 / 40 | 100% | 33% |

For comparison, the shipped `ambiguity_ratio` flag (`AMBIGUITY_THRESHOLD = 0.92`) fires on
**128 / 156** pairs to catch the same 40 failures — 31% precision. The gap statistic reaches 95%
recall while flagging 40% fewer pairs, and 80% recall while flagging 31%. The current ambiguity
signal is a materially worse version of a statistic the pool already contains.

## 6. What this implies for the next round

- **Selective escalation is now affordable.** Any expensive method — continuous warp
  optimization, a much denser grid, a learned matcher — can be gated behind the gap detector and
  run on ~31–49% of pairs instead of all of them. Gate criterion 6 caps runtime at 5x;
  `finer_hypothesis_grid`'s 3.17x, applied to 31% of pairs, costs roughly 1.7x. The
  runtime objection that has shadowed the grid-density direction largely dissolves.
- **The largest single bucket is candidate generation (45%)**, not ranking or scoring — and it is
  currently the least explored of the three. `wider_candidate_pool` tested widening the *existing*
  grid's pool; it did not test proposing candidates by a mechanism that can reach locations the
  grid's peak structure never surfaces.
- **Score-based re-ranking is not as dead as the campaign concluded**, but it needs an orthogonal
  signal worth >0.02–0.05, not a re-weighting of ZNCC. The margins are thin enough that a
  genuinely independent tie-break could flip the 22.5% runner-up bucket on its own.
- **`AMBIGUITY_THRESHOLD` is miscalibrated** and the gap statistic is a strictly better basis for
  the user-facing ambiguity flag. This is the cheapest concrete improvement identified here,
  though it improves reported confidence rather than accuracy.

## Reproduce

```
cd experiments/oracle_ceiling_diagnostic
python run_diagnostic.py        # oracle sweep, 40 failing + 20 control pairs (~140s)
python run_margin_detector.py   # pool-internal gap detector, all 156 pairs (~480s)
```

Outputs: `outputs/oracle_diagnostic.csv`, `outputs/oracle_summary.json`,
`outputs/margin_detector.csv`, `outputs/margin_detector_summary.json`.

One pair (`ch_combined_acquisition_007`) is excluded from the oracle statistics: its ground truth
sits too close to the search-image edge to place a full template within tolerance. It is retained
in the CSV with `NaN` scores rather than silently dropped.
