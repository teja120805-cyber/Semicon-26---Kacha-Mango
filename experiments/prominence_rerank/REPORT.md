# experiments/prominence_rerank — REJECT (real signal, but net harmful, not neutral)

## Summary

**Verdict: REJECT.** Unlike the four prior candidate-fusion/re-ranking experiments in this
campaign (all clean no-ops or negligible effects), local peak-prominence re-ranking is a
genuinely active signal: at sufficient strength it produces real catastrophic rescues (one
periodicity failure went from 525px to 41px error) - but it produces MORE real regressions than
rescues. At the dev-selected (disciplined, tuned-on-development-only) operating point, the
signal auto-tunes to zero (a no-op, bit-identical to baseline, 0.7436 pooled acc@5px). At an
exploratory full-benchmark strength beyond what the dev sweep selected, it broke 14-15 pairs for
every 4 it rescued, dropping pooled accuracy@5px from 74.4% to 67.9%. This is the first
experiment in this campaign whose failure mode is "actively harmful if pushed," not "inert" -
useful information for ruling out this entire signal family, not just this specific formulation.
Production is untouched.

## 1. The idea

Three independent prior experiments in this campaign
(`cross_hypothesis_consensus_rerank`, `hough_subpatch_voting`, `pyramid_periodicity_search`)
converged on the same mechanism finding: when the classical winner is wrong, it is virtually
never because the true candidate was unreachable or near-tied - it's because the wrong
candidate's raw ZNCC score is genuinely, substantially higher.
`pyramid_periodicity_search/REPORT.md` verified this directly: a candidate 0.4px from ground
truth scored 0.187, while the (8.5px-off) classical winner scored 0.775 - a 4x gap, nowhere near
any tie-break epsilon. Raw score conflates "this location matches distinctively well" with "this
location matches about as well as its immediate neighborhood, including elsewhere along a
periodic repeat." This experiment re-scores the top-K classical candidates by **prominence**:
peak score minus the strongest nearby competing peak in the SAME winning hypothesis's own
spatial correlation surface (not across hypotheses like `cross_hypothesis_consensus_rerank`, not
within one candidate's own template footprint like `hough_subpatch_voting`) - the intuition
being that a periodic decoy should have a near-as-good sibling peak close by, while a genuinely
distinctive match should stand out.

## 2. Implementation: a direct radial-profile measurement, not a guessed formula

A first attempt using the MEDIAN score in a wide annulus (6-35px) around each peak produced
near-zero, near-constant "background" values across every candidate (-0.005 to -0.019) -
useless for discrimination, because a periodic sibling peak is a small, sparse local maximum
within the annulus, not a uniformly elevated region; the median washes it out entirely. A direct
radial-profile probe on a real periodic pair (`dev_dense_periodic_006`) confirmed this: the
annulus at radius 4-6px had **mean -0.01 but max 0.744** (vs. the true peak's own score of
0.775) - a near-competitive sibling sitting right next to the winner, invisible to a median/mean
statistic. `prominence.py::compute_prominence` was corrected to use **MAX** over a **tight**
annulus (`inner_exclude_px=2.0` to skip the peak's own immediate blob,
`annulus_outer_px=12.0` - roughly 1.5-2x the ~5-7.5px periodic pitch measured directly via
autocorrelation in `pyramid_periodicity_search`), which produces real, varying prominence values
(e.g. 0.013-0.030 across the same pair's top-8 candidates) instead of a near-constant offset.
`prominence.rank_prominence(candidates, ..., top_k, gamma)` re-scores the top-K classical
candidates by `score + gamma * prominence` (gamma=0.0 is provably identical to
`rank_classical` - confirmed directly). `harness.py::localize_prominence` is structurally
identical to `pipeline.localize.localize`, swapping only the ranking step.

## 3. Evaluation

### Dev sweep (n=24, gamma ∈ {0, 0.5, 1, 2, 3, 5, 8} × top_k ∈ {4, 8})

| gamma | top_k | acc@5px | mean_error_px |
|---:|---:|---:|---:|
| 0.0 | 4 | 0.583 | 111.50 |
| 0.5 | 4 / 8 | 0.583 | 111.50 |
| 1.0 | 4 / 8 | 0.583 | 111.50 |
| 2.0 | 4 / 8 | 0.583 | 111.50 |
| 3.0 | 4 / 8 | 0.583 | 111.50 |
| 5.0 | 4 | **0.542** | 143.52 |
| 5.0 | 8 | **0.458** | 146.33 |
| 8.0 | 4 | **0.500** | 144.18 |
| 8.0 | 8 | **0.417** | 147.00 |

Below gamma=5, prominence's magnitude (typically 0.01-0.03) is too small relative to the score
gaps between distinct candidates (typically 0.05+) to ever change the ranking - a real but inert
regime. At gamma>=5, it starts flipping winners, and every flip tested on development was a NET
LOSS (accuracy dropped monotonically with gamma; the best "active" config, gamma=5/top_k=4, was
still 4.1pp worse than baseline on development). The disciplined dev-only selection procedure
correctly picked the highest-scoring config, which is one of the tied `gamma<=3` no-ops
(`gamma=0.0, top_k=4`).

### Full frozen benchmark at the dev-selected (no-op) config

| Split | Baseline acc@5px | Candidate acc@5px |
|---|---:|---:|
| development | 0.583 | 0.583 |
| validation | 0.900 | 0.900 |
| held_out | 0.650 | 0.650 |
| challenge | 0.750 | 0.750 |
| cross_generator | 0.800 | 0.800 |
| **Pooled (n=156)** | **0.7436** | **0.7436** |

Bit-identical to baseline, as expected for an auto-selected no-op.

### Exploratory: does the "active but harmful on development" pattern hold at full scale?

Ran `gamma=5.0, top_k=4` (the least-bad active dev config) over the full 156-pair frozen
benchmark, purely as a diagnostic - **not** the official gate verdict (the dev-only selection
already and correctly rejected any active gamma):

| Metric | Value |
|---|---:|
| Baseline pooled acc@5px | 0.7436 |
| Candidate pooled acc@5px | **0.6795** |
| Pairs rescued across the 5px line | 4 |
| Pairs broken across the 5px line | **14** |
| Rescues with >50px error improvement | 4 |
| Regressions with >50px error worsening | **15** |

The pattern holds and sharpens at full scale: prominence re-ranking is a real, non-random signal
(the 4 large rescues are genuine periodicity catastrophic-failure recoveries, not noise) but it
breaks **3.5x more pairs than it fixes**. Net pooled accuracy drops 6.4 percentage points.

## 4. Why this is net harmful, not just ineffective

Prominence conflates two distinct situations that raw ZNCC score alone doesn't separate well
either, but in the opposite direction from what was hoped: a true match on a **non-periodic**
family (`strip_anchor`, `single_mat`) can have LOW prominence too, whenever there happens to be
*any* other reasonably-similar-looking structure nearby in the same correlation surface - not
because of periodicity, but simply because ZNCC's correlation surface is rarely perfectly
"spiky" even for a correct, unique match. The heuristic has no way to distinguish "this
candidate's low prominence is because it's a periodic decoy" from "this candidate's low
prominence is incidental, and it's still the right answer." Several of the pairs broken in the
exploratory run were previously near-perfect (sub-1px baseline error) - correct matches with
very high absolute confidence that nonetheless had a competing nearby peak within the tight
12px annulus, enough to lose under `gamma=5`'s re-weighting despite being unambiguously correct
by every other measure in the pipeline.

## 5. What this means for the broader "raw ZNCC score is fooled" finding

This is a materially different and more informative negative result than the campaign's prior
no-ops: it demonstrates that the "wrong candidate scores genuinely higher" finding
(`pyramid_periodicity_search`) is real, but **naively discounting non-distinctive-looking
matches is not a viable fix** - the discount function needs to be far more selective than
"any nearby competing peak" to avoid punishing ordinary correct matches. A future attempt in
this direction (out of scope here) would need a much narrower discriminator - e.g. requiring the
competing peak to be at a spacing consistent with a *specific, measured* periodic pitch for that
pair, not any nearby peak at all - to avoid this experiment's false-positive problem.

## Reproduce

```
cd experiments/prominence_rerank
python run_experiment.py   # official dev-selected result (no-op)
```

The exploratory full-benchmark check at `gamma=5.0, top_k=4` is a one-off diagnostic script
(not checked in as a separate file); its outputs are saved at
`outputs/per_pair_results_prominence_exploratory.csv` and `outputs/exploratory_gate_result.json`.

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_prominence.csv`,
`outputs/prominence_metrics.json`, `outputs/integration_gate_result.json`,
`outputs/per_pair_results_prominence_exploratory.csv`, `outputs/exploratory_gate_result.json`.
