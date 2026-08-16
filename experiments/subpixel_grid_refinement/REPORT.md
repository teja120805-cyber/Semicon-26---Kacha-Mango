# experiments/subpixel_grid_refinement — REJECT (real effect, but below measurement resolution)

## Summary

**Verdict: REJECT — no measurable accuracy@5px effect, but not a no-op.** Unlike
`cross_hypothesis_consensus_rerank`, this technique demonstrably works: it changes the final
predicted coordinate on **156/156 pairs** (never a true no-op) by continuously interpolating
scale and rotation between the hypothesis grid's fixed points. But the shifts are tiny — mean
0.019px, max 0.305px — far below the resolution needed to flip any pair across the 5px
threshold. Pooled accuracy@5px: 0.7436 → 0.7436 (unchanged). Production is untouched.

## 1. The idea

`pipeline/refinement.py` already does parabolic subpixel interpolation independently along x
and y at the winning hypothesis's own correlation peak. The SCALE and ROTATION axes stay
coarse — a fixed grid (`candidate_generation.py::DEFAULT_SCALE_HYPOTHESES`/
`DEFAULT_ROTATION_HYPOTHESES`). `reports/ACCURACY_FORENSICS.md` Finding 2 originally measured a
real "sawtooth" accuracy pattern tracking distance to the nearest tested scale/rotation
hypothesis (e.g. `extra_scale=1.02`, exactly between two then-existing grid points, collapsed
to 42.5%@5px vs. 70% at grid-aligned values). This experiment extends the SAME parabolic-
interpolation idea to the scale and rotation axes: sample correlation score under the winning
hypothesis's immediate grid neighbors (holding x, y fixed), parabolically interpolate a
continuous scale/rotation estimate, re-render one template at that estimate, then finish with
the existing x/y parabolic refinement.

**Important caveat found while grounding this idea**: Finding 2's sawtooth measurement predates
two subsequent grid-density changes that are now in production — `finer_hypothesis_grid`
(5×5 → 9×9 scale×rotation combos) and `scale_range_v1`/A2 (9 → 11 scale points, span widened to
the literal 9:1–11:1). The current production grid (11 scale × 9 rotation = 99 combos, 0.2
scale step, 1.25° rotation step) is considerably denser than what Finding 2 was measured
against. This experiment is a legitimate test of whether *residual* quantization error remains
after those two integrations — see the result below.

## 2. Implementation

`joint_refine.py::refine_joint` — given the winning candidate, locates its scale/rotation
indices in the (unmodified) hypothesis tuples, samples 2 neighboring-scale correlation scores
at the same (x, y) location, parabolically interpolates a continuous scale, repeats for
rotation at the refined scale, then re-renders one final template and does the existing x/y
parabolic refinement. Costs ≤5 extra correlation calls per pair (negligible next to the 99
hypothesis correlations candidate generation already performs — confirmed in the runtime
numbers below, no meaningful slowdown). `harness.py::localize_joint_refine` is structurally
identical to `pipeline.localize.localize`, swapping only the refinement step — candidate
generation, dedup, and classical ranking/tiebreak all call the unmodified production functions.

## 3. Evaluation

No hyperparameters to tune (the refinement is deterministic parabolic interpolation against
the existing grid) — straight to the full frozen benchmark, once.

| Split | Baseline acc@5px | Candidate acc@5px |
|---|---:|---:|
| development | 0.583 | 0.583 |
| validation | 0.900 | 0.900 |
| held_out | 0.650 | 0.650 |
| challenge | 0.750 | 0.750 |
| cross_generator | 0.800 | 0.800 |
| **Pooled (n=156)** | **0.7436** | **0.7436** |

Integration gate: **FAILED** (criteria 1/2, `validation`/`held_out` must strictly improve, not
tie) — same structural reason `wider_candidate_pool` and `finer_hypothesis_grid` (pre-
integration) failed it: a tie is not an improvement under the gate's literal criteria.

### Did anything actually change? (yes — direct per-pair coordinate diff)

Confirmed by diffing every predicted `(x, y)` against the baseline CSV, pair by pair:

- **156/156 pairs have a nonzero coordinate shift** — this is NOT a no-op like
  `cross_hypothesis_consensus_rerank`'s auto-selected `alpha=0`. The joint refinement is doing
  real work on every pair.
- **Mean shift: 0.019px. Max shift: 0.305px** (`ch_combined_acquisition_003`). Every shift is
  sub-pixel.
- Mean error_px: 55.650px → 55.651px (unchanged to 3 decimal places). Median error_px:
  0.3340px → 0.3360px (a ~0.002px change, noise-level).
- No pair crossed the 5px threshold in either direction.

## 4. Why this didn't move accuracy

The mechanism is real (verified directly, not inferred) but the residual it corrects is now
too small to matter: the two prior grid-density integrations
(`experiments/finer_hypothesis_grid/`, `experiments/scale_range_v1/`) already closed most of
the quantization gap Finding 2 originally measured. What's left for this refinement to recover
is a fraction-of-a-pixel correction on top of an already-dense 99-combination grid — real, but
below the 5px tolerance the benchmark scores against. This is a useful negative result: it
suggests the scale/rotation hypothesis grid is close to its useful density ceiling for this
tolerance level; further grid refinement (finer steps, or this kind of continuous
interpolation) is unlikely to be a productive direction without also addressing candidate-
generation-stage failures (periodicity, boundary absence) first, which remain the dominant
bottleneck per `reports/ACCURACY_FORENSICS.md`.

## Reproduce

```
cd experiments/subpixel_grid_refinement
python run_experiment.py
```

Outputs: `outputs/per_pair_results_joint_refine.csv`, `outputs/joint_refine_metrics.json`,
`outputs/integration_gate_result.json`.
