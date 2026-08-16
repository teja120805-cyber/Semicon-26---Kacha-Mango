# experiments/pyramid_periodicity_search — REJECT (mechanism engages, still loses on score)

## Summary

**Verdict: REJECT — bit-identical to baseline on every one of 156 pairs, at every tested config.**
Coarse-resolution (area-downsampled) candidate proposal followed by local full-resolution
refinement produces a real, spatially-precise proposal on roughly half the development set
(12/24 pairs had a pyramid-derived candidate within 20px of ground truth) - including two
currently-failing pairs where the proposal landed within **1px** of ground truth. But its ZNCC
score is always far below the classical grid's own (slightly-off) winner, so it never wins under
`rank_classical`'s arg-max. Pooled accuracy@5px: 0.7436 → 0.7436, confirmed bit-identical
per-pair (`error_px` diff < 1e-6 on all 156 pairs, not just tied aggregate accuracy). Production
is untouched.

## 1. The idea

`reports/ACCURACY_FORENSICS.md` traces periodicity-driven failures to the classical matcher's
correlation surface being fooled by repeat-pitch decoys. Direct measurement (not assumed) via
self-correlation of a `dev_dense_periodic` reference template found the word/bit periodic pitch
to be roughly 5-7.5px in the same pixel space `matching.correlate` operates in - small relative
to macro structural features like mat boundaries (tens to hundreds of pixels). The hypothesis:
`cv2.INTER_AREA` downsampling by a factor large enough to push that ~5-7.5px pitch toward/below
the coarse image's Nyquist limit should average away the periodic repeat while boundary-scale
structure survives, giving a coarse correlation pass a shot at NOT being fooled the same way the
full-resolution 99-hypothesis grid is. Coarse proposals are then refined by a full-resolution,
locally-windowed 99-hypothesis correlation (cheap - windows are ~100-150px, not the full ~1000px
search image) and merged into the standard pool as ADDITIONAL candidates, never replacing the
existing grid. This is a genuinely different mechanism from every previously-tried idea: not an
alternative whole-template scoring function (`periodicity/`, rejected), not a re-ranking pass
over the existing pool (`cross_hypothesis_consensus_rerank`, `hough_subpatch_voting`, both
rejected), not a different candidate-proposal geometry (`keypoint_candidate_fusion`, rejected) -
this changes the SPATIAL RESOLUTION candidate generation operates at.

## 2. Implementation

`pyramid_search.py::build_coarse_to_fine_candidates` - downsamples both Reference and Search by
the same factor (so the scale/rotation hypothesis grid stays physically valid), runs the
unmodified `candidate_generation.build_candidate_pool` at that coarse resolution, dedups and
keeps the top-K coarse locations, then for each one runs a full 99-hypothesis correlation
confined to a small local window (`window_margin_px` beyond the template footprint) around the
coarse proposal's full-resolution position. Cost analysis (confirmed by observed runtimes):
`downsample=3` coarse pass costs ~11% of one full-image correlation call per hypothesis
(image area scales as 1/9), and each of the `top_k_coarse` local-window passes costs roughly 1-2%
of a full-image call (window area ~130x130 vs ~1000x1000) - the whole mechanism added only
~0.9s/pair on top of baseline's ~3.2s/pair (see runtime table below), comfortably inside the
gate's `MAX_RUNTIME_MULTIPLIER=5.0`. `harness.py::localize_pyramid` is structurally identical to
`pipeline.localize.localize`, merging the pyramid candidates into the raw pool before the
unmodified `deduplicate_by_location`/`rank_classical`/`apply_center_tiebreak`/`refinement.refine`.

## 3. Evaluation

### Dev sweep (n=24, downsample_factor ∈ {3, 4, 6} × top_k_coarse ∈ {3, 5}, window_margin_px=15 fixed)

| downsample_factor | top_k_coarse | acc@5px | mean_error_px |
|---:|---:|---:|---:|
| 3.0 | 3 | 0.583 | 111.50 |
| 3.0 | 5 | 0.583 | 111.50 |
| 4.0 | 3 | 0.583 | 111.50 |
| 4.0 | 5 | 0.583 | 111.50 |
| 6.0 | 3 | 0.583 | 111.50 |
| 6.0 | 5 | 0.583 | 111.50 |

Every configuration reproduced the exact baseline development accuracy and mean error - no
config ever changed a single winning candidate. Chosen (first-tied) config:
`downsample_factor=3.0, top_k_coarse=3`.

### Full frozen benchmark at the chosen config

| Split | Baseline acc@5px | Candidate acc@5px |
|---|---:|---:|
| development | 0.583 | 0.583 |
| validation | 0.900 | 0.900 |
| held_out | 0.650 | 0.650 |
| challenge | 0.750 | 0.750 |
| cross_generator | 0.800 | 0.800 |
| **Pooled (n=156)** | **0.7436** | **0.7436** |

Confirmed **bit-identical per-pair `error_px` on all 156 pairs** (diff < 1e-6 everywhere, not
just matching aggregate accuracy). Runtime: baseline mean 3.18s/pair vs. candidate mean 4.06s/pair
(1.28x - well within budget).

## 4. Mechanism diagnosis — why this didn't work, verified directly (not inferred)

Unlike the dev-sweep-level "never flips" results in `cross_hypothesis_consensus_rerank` and
`hough_subpatch_voting`, this mechanism DOES actively produce spatially-precise, competing
candidates: **12/24 development pairs had a pyramid-derived candidate within 20px of ground
truth.** Direct inspection of two currently-failing pairs (`dev_single_mat_004`,
`dev_dense_periodic_006`) shows exactly why that never translates into a win:

| Pair | Baseline error | Classical winner score | Pyramid candidate distance to GT | Pyramid candidate score |
|---|---:|---:|---:|---:|
| `dev_single_mat_004` | 12.8px | 0.857 | 5.9px | 0.064 |
| `dev_dense_periodic_006` | 8.5px | 0.775 | **0.4px** | 0.187 |

On `dev_dense_periodic_006`, the pyramid mechanism found a candidate **0.4px from ground truth**
- essentially exact - but scored it 0.187, while the classical grid's own top-1 candidate (8.5px
from ground truth, using a neighboring scale/rotation hypothesis) scored 0.775, over 4x higher.
Both candidates are well outside the two default tie-break epsilons
(`TIE_SCORE_EPSILON=1e-6`, `MULTIWAY_TIE_SCORE_EPSILON=0.005`), so no tie-break logic could ever
have intervened even if the near-exact candidate had reached the ranking stage. This is not a
"the mechanism can't find the right spot" failure - it's a "ZNCC score at the exactly-correct
subpixel location is intrinsically far weaker than at a nearby, hypothesis-grid-quantized,
slightly-wrong location" failure. That is consistent with (and sharpens) the residual-
quantization-gap finding from `subpixel_grid_refinement` (real subpixel shifts exist, but are
too small to matter) - here the gap is coarser (several px, not a fraction of a px) but the
underlying cause is the same family of issue: ZNCC's raw score surface near a true periodic
match does not necessarily peak exactly at the true location, and no post-hoc candidate-fusion
mechanism that still arbitrates purely by ZNCC score can fix that without changing the scoring
function itself - the exact class of fix `periodicity/`'s gradient-domain and ensemble scoring
variants already tried and had net-negative results on.

## 5. What this doesn't rule out

This confirms the SPATIAL location can sometimes be found via a different (coarse-to-fine)
search strategy even when the classical grid's own winner is off by several pixels - a genuinely
new, verified fact this benchmark didn't previously have direct evidence for. It doesn't rule out
a scoring formulation that weights LOCAL SPATIAL PRECISION (e.g., agreement between a coarse and
a fine-resolution search) rather than raw ZNCC magnitude - but that would need a fundamentally
different arbitration rule than "highest ZNCC wins," a materially bigger and more speculative
change than this bounded experiment was scoped to test.

## Reproduce

```
cd experiments/pyramid_periodicity_search
python run_experiment.py
```

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_pyramid.csv`,
`outputs/pyramid_metrics.json`, `outputs/integration_gate_result.json`.
