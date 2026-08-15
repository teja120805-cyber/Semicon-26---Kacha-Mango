# Experiment: rotation_scale (coarse-to-fine local refinement)

**Status: REJECTED, not integrated.** Weaker net result than the already-tested
`experiments/finer_hypothesis_grid/` at similar (not dramatically lower) runtime cost — doesn't
offer a clear advantage over that already-documented near-miss.

## What this experiment was

`experiments/finer_hypothesis_grid/` showed a globally denser scale/rotation grid (81 vs. 25
hypotheses) rescues grid-misalignment failures (net +5) but at ~3.17x runtime (corrected
measurement). This experiment tests a cheaper alternative: run the unmodified 25-hypothesis coarse
search, then locally refine only the **top-8 candidate locations** with a small rotation/scale
neighborhood search restricted to a small crop around each candidate (`experiments/rotation_scale/
harness.py::refine_candidate_locally`) rather than re-correlating the whole image at every finer
hypothesis. Re-rank the refined candidates, then apply the same production subpixel refinement.

## Result

| Split | Baseline | Candidate | Change |
|---|---:|---:|---|
| validation | 90.0% | 90.0% | tied |
| held_out | 55.0% | **60.0%** | +5.0pp |
| challenge | 68.8% | 68.8% | tied |
| cross_generator | 65.0% | 65.0% | tied |

Rescue/break: **4 rescued** (`ho_rotation_drift_000`, `ho_rotation_drift_001`, `ho_scale_drift_005`,
`ch_combined_acquisition_005`), **2 broken** (`ho_rotation_drift_006`, `ch_worst_case_002`) — **net
+2**. `ch_worst_case` regresses in aggregate (50%→37.5%) because of the one broken pair, so the
gate's per-family-regression criterion fails. Runtime: 1.17-1.70s/pair (~2.3-3.3x baseline) —
passes the 5x budget cleanly, without needing a contention correction this time.

Notably, **both broken pairs are the exact same pairs `finer_hypothesis_grid` also broke**
(`ho_rotation_drift_006`, `ch_worst_case_002`) — the same double-edged-sword mechanism: giving any
extra rotation/scale hypotheses more chances to test also gives competing wrong locations more
chances to find a spuriously good alignment, and these two specific pairs are fragile to that in
either design.

## Why coarse-to-fine underperformed the simpler global grid

The expectation going in was that restricting the fine search to a small local window around
already-found candidates would be much cheaper than a globally denser grid, for the same accuracy
benefit. In practice: (1) the runtime saving was smaller than expected (~2.3-3.3x vs. 3.17x for the
full grid) — `cv2.matchTemplate`'s per-call overhead doesn't shrink linearly with window size, so 64
small local correlations cost nearly as much as the full grid's extra 56 full-image ones; (2) the
accuracy benefit was also smaller (net +2 vs. +5) — coarse-to-fine can only re-rank among the top-8
*already-discovered* candidate locations, whereas a globally denser grid can surface an entirely new
peak location that never made the original top-8. Both the cost saving and the benefit were smaller
than hoped, and the combination doesn't beat the simpler design on either axis.

## Verdict

**Reject.** Not just "fails the gate" — it fails the gate in a way that isn't compensated by a
runtime or engineering advantage over the already-tested alternative. If rotation/scale
grid-refinement is revisited, `finer_hypothesis_grid`'s simpler global-grid design (already a
near-miss, one criterion away) is the stronger starting point, not this one.

## Production impact

None. `pipeline/candidate_generation.py` and `pipeline/matching.py` are imported unmodified; this
experiment's refinement logic lives entirely in `experiments/rotation_scale/`.
