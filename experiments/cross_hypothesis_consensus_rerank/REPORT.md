# experiments/cross_hypothesis_consensus_rerank — REJECT (clean no-op)

## Summary

**Verdict: REJECT.** Every tested strength of the cross-hypothesis consensus signal made
development-split accuracy equal or worse than plain `rank_classical`; the dev-only sweep
correctly auto-selected `alpha=0.0` (a no-op), which reproduces the classical baseline
bit-for-bit on the full frozen benchmark (156/156 identical predictions, pooled accuracy@5px
0.7436 → 0.7436). Production is untouched; this experiment never imported/modified `pipeline/`.

## 1. The idea

`pipeline/ranking.py::rank_classical` picks the single highest-ZNCC-score candidate
(pure arg-max) from the deduplicated candidate pool. `reports/ACCURACY_FORENSICS.md` Finding 1
shows periodicity-driven candidate-generation failures are frequently near-multi-way ties
(score margins of 0.002–0.007 between the winning wrong candidate and the true location's
runner-up). The hypothesis: a genuinely correct match is often found as a competitive peak by
*multiple* scale/rotation hypotheses (the true structure tolerates a slightly-wrong scale/
rotation guess — this is literally why `candidate_generation.deduplicate_by_location` exists),
while an isolated periodic-aliasing peak might be more hypothesis-specific. If true, weighting
each deduplicated candidate's raw ZNCC score by how much cross-hypothesis "support" (summed
score of every raw, pre-dedup candidate within a radius) backs its location could out-rank a
periodic decoy that only one or two hypotheses happen to favor.

## 2. Implementation

`rerank.py::rank_consensus(deduped, raw, alpha, radius_px)` — for each deduplicated candidate,
sums the ZNCC scores of every raw (pre-dedup) candidate within `radius_px`, normalizes by the
pool's max support, and re-ranks by `score + alpha * normalized_support`. `alpha=0` is
mathematically identical to `rank_classical` (a built-in sanity check). `harness.py::
localize_consensus` is structurally identical to `pipeline.localize.localize`, swapping only
the ranking step — candidate generation, dedup, center tiebreak, refinement, and ambiguity
reporting all call the unmodified production functions directly.

## 3. Evaluation

Two-stage, following this project's established dev-only-tuning discipline (same pattern as
`ranking.py`'s own `TIE_SCORE_EPSILON`/multiway-tier tuning, and `model/train.py`'s early
stopping): swept `alpha ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}` × `radius_px ∈ {10, 15, 25}` on
`development` only (n=24, never validation/held_out/challenge), then ran the single chosen
config once over the full frozen benchmark.

### Dev sweep (n=24)

| alpha | best radius_px | acc@5px | mean_error_px |
|---:|---:|---:|---:|
| 0.0 (baseline) | — | **0.583** | **111.5px** |
| 0.05 | 10/15/25 (tied) | 0.583 | 140.7–143.0px |
| 0.1 | 15 or 25 | 0.583 | 145.1–149.2px |
| 0.1 | 10 | 0.500 | 138.9px |
| 0.2 | any | 0.417–0.458 | 144.2–162.7px |
| 0.3 | any | 0.375–0.500 | 176.3–193.1px |
| 0.5 | any | 0.333–0.458 | 165.2–176.3px |

No configuration ever *exceeds* baseline accuracy on development; several tied at the same
accuracy but with **worse mean error** (i.e. correctly-ranked pairs stayed correct, but some
previously-close near-misses got pushed further away — consistent with the signal being noise
rather than a real discriminator). Selection logic (`max acc, tie-break on min mean_error`)
correctly picked `alpha=0.0`, i.e., "don't use this."

### Full frozen benchmark at the chosen (no-op) config

| Split | Baseline acc@5px | Candidate acc@5px |
|---|---:|---:|
| development | 0.583 | 0.583 |
| validation | 0.900 | 0.900 |
| held_out | 0.650 | 0.650 |
| challenge | 0.750 | 0.750 |
| cross_generator | 0.800 | 0.800 |
| **Pooled (n=156)** | **0.7436** | **0.7436** |

Bit-identical per-pair predictions (confirmed via the integration gate's per-family regression
check — zero families regressed or improved, because `alpha=0.0` produces the exact same
ranking as `rank_classical`).

## 4. Why this didn't work

The dev sweep is the real finding: cross-hypothesis support is not a useful discriminator
between true matches and periodic decoys at any tested strength. A plausible mechanism —
periodic-lattice decoys are, by construction, *also* found by many nearby scale/rotation
hypotheses (the whole point of a periodic pattern is that it looks similar under small
perturbations), so "how many hypotheses agree on this location" doesn't reliably separate a
true match from a repeat one pitch away. This is a different-shaped negative result from the
already-rejected `experiments/periodicity/` (alternative whole-template scoring functions) and
`wider_candidate_pool` (structural no-op under pure arg-max) — it specifically rules out
"cross-hypothesis redundancy as a ranking signal," a mechanism neither of those tested.

## 5. What this doesn't rule out

This only tested a linear `score + alpha * support` combination with a spatial-radius support
metric. It doesn't rule out more sophisticated consensus formulations (e.g. requiring
consensus from hypotheses that are *far apart* in scale/rotation space specifically, rather
than just counting raw pool density) — but given the monotonic degradation with increasing
alpha, there's no evidence in this data that any weighting of this signal would help.

## Reproduce

```
cd experiments/cross_hypothesis_consensus_rerank
python run_experiment.py
```

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_consensus.csv`,
`outputs/consensus_metrics.json`, `outputs/integration_gate_result.json`.
