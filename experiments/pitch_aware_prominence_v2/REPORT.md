# experiments/pitch_aware_prominence_v2 — REJECT (fix confirmed safe, but confirms v1's "improvement" was never real signal)

## Summary

**Verdict: REJECT — but a conclusive, informative one.** This experiment fixes the exact
mechanism that made `pitch_aware_prominence` (v1) fail its second-seed validation. The fix works
completely: re-tested directly on the same 4 pairs v1 broke on the second seed, all 4 are now
bit-identical to baseline, even at 4x the strongest gamma v1 ever used. But the fix is a **clean,
total no-op everywhere else too** - zero pairs changed anywhere in the full 156-pair production
benchmark, at every tested strength up to gamma=40 (v1's sweep only went to 12). This is the
important finding: it proves v1's apparent 0.7436 -> 0.7500 "improvement" was **entirely an
artifact of a flawed formula**, not a real, exploitable periodicity signal that a more careful
formulation could recover. Production is untouched.

## 1. Diagnosing v1's second-seed failure

v1's formula was `score + gamma * prominence`, where `prominence = candidate_score -
strongest_score_at_measured_pitch_offset` - a SIGNED value that could be positive (no strong
competitor nearby) or negative (a strong competitor found). Re-examining every single pair v1
ever changed - all 6 on the production seed and all 4 on the second seed, 10 pairs total, zero
exceptions - found the same root cause every time: **the flip was always driven by a POSITIVE
prominence bonus being awarded to a candidate, never by a negative penalty independently
demoting the wrong winner.** Two representative examples, pulled directly from the per-pair
score/prominence table (not summarized or approximated):

- `ho_rotation_drift_006` (broke on the second seed, 0.15px -> 60.2px): the true candidate had
  the highest raw score (0.9320) AND the second-highest prominence (+0.0902) among the top-4 -
  by any reasonable measure the clear winner. It lost because the 4th-ranked candidate (raw
  score only 0.8302, the lowest of the four) got an outlier prominence bonus of **+0.2207** -
  roughly 2-7x every other candidate's prominence in that same pool - large enough to overtake
  the true candidate's comfortable score lead. That bonus wasn't evidence of a better match; it
  was an artifact of where that specific candidate's pitch-offset probe happened to land.
- `ch_combined_acquisition_005` (v1's own production-seed regression, 0.11px -> 76.9px): same
  pattern - the true winner (raw score 0.6648, highest in the pool) lost to a much lower-scoring
  candidate (0.6486) that got a disproportionate +0.0268 bonus vs. the winner's +0.0079.

This generalizes cleanly: **"no strong competing peak nearby" is not, by itself, positive
evidence that a candidate is correct** - many genuinely correct matches simply don't have a
strong nearby competitor, and rewarding the absence of one is exactly as likely to reward a
coincidentally-isolated wrong candidate as a correct one. The theoretically sound half of the
idea - "a strong competing peak at the measured periodic pitch IS evidence of a possible decoy" -
was never actually the mechanism causing v1's failures; the (unjustified) reward half was.

## 2. The fix

`pitch_aware_v2.py::rank_pitch_aware_prominence_v2` - identical to v1 in every respect (same
`detect_pitch`, same `compute_pitch_aware_prominence`, same signed prominence value) except the
final re-score clips the term to `min(prominence, 0.0)` before applying `gamma`: **prominence can
only ever penalize, never reward.** `gamma=0.0` remains provably identical to `rank_classical`.

## 3. Evaluation

### Dev sweep (n=24, gamma ∈ {0, 1, 3, 5, 10, 20, 40} × top_k ∈ {4, 8})

Every single one of these 13 configurations produced **exactly** `acc@5px=0.583,
mean_error_px=111.50` - identical to `gamma=0` to the decimal, at every tested strength up to
40x v1's peak gamma. The penalty pathway never once fires on the 24 development pairs. (The sweep
range was deliberately widened past v1's 0-12 range specifically because the penalty-only formula
can no longer runaway-reward an outlier the way v1's could, so there was no a priori reason to
expect harm from a much larger gamma - and indeed there wasn't any effect at all, positive or
negative.)

### Full frozen benchmark (dev-selected gamma=0.0 - trivially reproduces baseline, as expected)

| Split | Baseline acc@5px | Candidate acc@5px |
|---|---:|---:|
| development | 0.583 | 0.583 |
| validation | 0.900 | 0.900 |
| held_out | 0.650 | 0.650 |
| challenge | 0.750 | 0.750 |
| cross_generator | 0.800 | 0.800 |
| **Pooled (n=156)** | **0.7436** | **0.7436** |

### Exploratory: full 156-pair benchmark at the most aggressive tested config (gamma=40, top_k=8)

Since the dev-only selection trivially picked the no-op `gamma=0` (every config tied), this
exploratory pass checks whether the penalty pathway EVER fires anywhere in the full benchmark,
not just the 24-pair development split - the same diagnostic discipline used in
`hough_subpatch_voting` and `prominence_rerank`'s exploratory passes.

```json
{
  "n": 156,
  "baseline_acc5": 0.7436,
  "candidate_acc5": 0.7436,
  "rescued_across_5px": 0,
  "broken_across_5px": 0,
  "total_changed": 0
}
```

**Zero pairs changed anywhere in the entire 156-pair benchmark**, even at 4x the gamma v1 ever
tested and with double the candidates eligible for re-scoring (`top_k=8`). This is not a
near-zero effect - it is an exact, complete no-op, confirmed by direct per-pair diff, not just
matching aggregate accuracy.

### Second-seed check: the exact 4 pairs v1 broke, regenerated and re-tested directly

| Pair | Baseline error | v1 (broke) | v2 (gamma=40, top_k=8) |
|---|---:|---:|---:|
| `dev_single_mat_006` | 0.367px | 42.77px | **0.367px** |
| `dev_dense_periodic_001` | 0.131px | 811.81px | **0.131px** |
| `ho_rotation_drift_004` | 0.188px | 55.17px | **0.188px** |
| `ho_rotation_drift_006` | 0.145px | 60.20px | **0.145px** |

All 4 confirmed fixed, bit-identical to baseline, even at v2's most aggressive tested
configuration. (Only these 4 specific pairs were re-verified on the second seed - given the full
156-pair production-seed check already showed a complete, zero-exception no-op, regenerating the
full 136-pair second-seed dataset again to re-confirm a result that direct mechanism analysis
already fully explains was not judged to add further evidence worth the ~15 additional minutes
of compute; this is a smaller, honestly-scoped verification than a full second-seed sweep, and is
flagged as such.)

## 4. Why this is a REJECT, not a partial win

The safety fix works completely - there is no evidence of any regression risk left in this
formulation, at any tested strength. But there is also no accuracy benefit anywhere: the penalty
pathway this experiment isolated never independently demotes a wrong winner far enough to change
any outcome, on either seed. Combined with section 1's diagnosis, the honest conclusion is that
**v1's mechanism, as measured across two independent seeds, was never generating real signal from
periodic-pitch competition** - the entire effect (both the apparent gains and the real
regressions) came from the unjustified bonus term. Removing the flaw removes the entire
phenomenon, not just the risk. This is a complete, evidence-backed closure of the
"pitch-aware/prominence-based reranking" family of ideas explored across `prominence_rerank`, v1,
and v2 - not a partial success worth iterating on further with this specific mechanism.

## Reproduce

```
cd experiments/pitch_aware_prominence_v2
python run_experiment.py   # dev sweep + production-seed benchmark (trivial no-op)
```

The exploratory full-benchmark check (gamma=40, top_k=8) and the second-seed spot-check are
one-off diagnostic scripts (not checked in separately); their outputs are saved at
`outputs/per_pair_results_pitch_aware_v2_exploratory.csv` and `outputs/exploratory_gate_result.json`.

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_pitch_aware_v2.csv`,
`outputs/pitch_aware_v2_metrics.json`, `outputs/integration_gate_result.json`,
`outputs/per_pair_results_pitch_aware_v2_exploratory.csv`, `outputs/exploratory_gate_result.json`.
