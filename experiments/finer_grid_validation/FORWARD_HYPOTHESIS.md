# Forward hypothesis (Phase 1 forensic inspection)

Source: `experiments/finer_hypothesis_grid/harness.py` + `run_experiment.py` (read in full before
writing this).

## 1-2. Exact mechanism changed

Exactly one thing: the `scale_hypotheses`/`rotation_hypotheses` tuple passed into the **unmodified**
`pipeline.candidate_generation.build_candidate_pool(reference, search, scale_hypotheses=...,
rotation_hypotheses=...)`.

- Production: 5 scale x 5 rotation = 25 hypotheses. `(9.2, 9.6, 10.0, 10.4, 10.8)` x
  `(-5.0, -2.5, 0.0, 2.5, 5.0)`.
- Candidate: 9 x 9 = 81 hypotheses, **identical span**, half the step size:
  `(9.2, 9.4, ..., 10.8)` x `(-5.0, -3.75, ..., 5.0)`.

Nothing downstream changes: `candidate_generation.deduplicate_by_location`, `ranking.rank_classical`
(pure arg-max by ZNCC score), and `refinement.refine` (parabolic subpixel) are all imported and
called exactly as production does, unmodified, in the same order. The experiment's
`instrumented_localize` is a diagnostic wrapper, not a reimplementation.

## 3. Ground truth use — confirmed clean

`gt_x`/`gt_y` are never passed into `build_candidate_pool`, `rank_classical`, or `refine` — those
three calls only ever see `reference`/`search` pixel arrays and the hypothesis grid. GT is read only
*after* the winner is already decided, purely to compute `error_px` and diagnostic labels
(`gt_in_pool`, `gt_candidate_rank`, `failure_location`) — identical in spirit to how
`evaluation/evaluate.py` scores production predictions. No leakage into the decision path.

## 4. Test/validation leakage — confirmed clean

No training occurs (this is a deterministic classical algorithm, no gradient updates, nothing
learned from data). The only leakage question that applies is "was the grid design tuned by peeking
at validation/held_out results?" — no: the 81-hypothesis grid was chosen a priori as the natural
"double the density, same span" design, not iteratively re-selected after seeing accuracy numbers.
`experiments/finer_hypothesis_grid/fresh_data/` (seed `424242`) was verified to have zero
signature overlap with `data/validation`/`held_out`/`challenge` before any comparison was run.

## 5. Ranking logic — confirmed unaltered

`ranking.rank_classical` is called exactly as production calls it (arg-max ZNCC score over the
combined candidate pool). The finer grid does not change *how* candidates are ranked, only *how many
(rotation, scale) hypotheses* get a chance to propose a candidate in the first place. This is the
whole mechanism, and it's a minimal one: denser sampling of an existing search space, not a new
algorithm.

## 6. Failure modes this targets

Per `reports/ACCURACY_FORENSICS.md` Finding 3: a true rotation/scale value that falls **between** two
tested hypotheses scores worse (under its nearest tested neighbor) than a wrong location does under
a well-matched hypothesis — so the classical arg-max ranker picks the wrong place *for the right
reason* (best available score). Halving the step size directly shrinks the worst-case distance from
any true value to its nearest tested hypothesis, so:

- **Primary target**: candidate-generation and candidate-ranking failures caused by hypothesis-grid
  misalignment on pairs with nonzero rotation and/or scale drift.
- **Not targeted, and not expected to help**: pure periodic-aliasing failures with zero rotation/
  scale drift (`dev_dense_periodic`-style cases) — the grid change does nothing for a hypothesis that
  was already exactly right (rotation=0, scale=1.0 is tested in both grids). The prior evaluation's
  own finding that high-periodicity cases improved too is treated as needing re-confirmation here,
  not assumed.
- **Not targeted**: pure noise/dose degradation, boundary-driven cases (both already handled well by
  the classical baseline regardless of grid density).

## What this validation round needs to establish

The prior re-validation (`experiments/finer_hypothesis_grid/REPORT.md`) already showed net rescue +4
on two datasets and a consistent -6 candidate-generation-failure reduction, but neither dataset was
*deliberately constructed* to stress-test the specific mechanism above — they were general-purpose
benchmark-style sets where rotation/scale-affected pairs were a minority (~20%). This round builds a
validation set that deliberately over-represents the exact conditions in section 6, to test the
hypothesis under conditions actually designed to expose it, per Phase 2.
