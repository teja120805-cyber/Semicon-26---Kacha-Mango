# DriftSense V2 — Model Evaluation Report

**Candidate**: `model/architecture.py::EmbeddingNet` (64,992 params), trained per
`model/TRAINING_PROTOCOL.md` on 72 hard-negative triplets from the `development` split only, evaluated
via `pipeline/ranking.py::rank_with_model` re-ranking the top 12 classical candidates.

**Verdict: FAILS the integration gate, decisively and consistently across all 3 required seeds.
NOT integrated. The production pipeline's default ranking remains classical
(`pipeline/ranking.py::rank_classical`), unchanged.**

This is the honest result, not a disappointing one to soften: the brief is explicit that a failing
model must not be integrated regardless of effort invested, and that failures should be documented
plainly rather than hidden. That is what follows.

## Integration gate outcome (`reports/V2_ARCHITECTURE_PLAN.md` section 8)

| # | Criterion | seed 20260101 | seed 20260102 | seed 20260103 |
|---|---|---|---|---|
| 1 | Improves validation@5px | FAIL | FAIL | FAIL |
| 2 | Improves held_out@5px | FAIL | FAIL | FAIL |
| 3 | Improves/ties cross_generator@5px | FAIL | FAIL | FAIL |
| 4 | No catastrophic-failure increase | FAIL | FAIL | FAIL |
| 5 | No per-family regression | FAIL (12/13 families regressed) | FAIL (12/13) | FAIL (13/13) |
| 6 | Acceptable runtime | PASS | PASS | PASS |
| 7 | Stable across seeds | N/A — see below | | |

**Every accuracy/failure criterion fails, on every seed.** Criterion 7 ("stable across seeds") is
reported as FAIL in the raw gate output because it demands stable *improvement*, and there is no
improvement to be stable — but the failure itself is highly consistent across seeds (see below), which
is at least informative: this is not one unlucky initialization, it is the expected behavior of this
approach at this data scale.

## Accuracy@5px, classical baseline vs. learned re-ranker (all 3 seeds)

| Split | Classical baseline | Learned (seed 101) | Learned (seed 102) | Learned (seed 103) |
|---|---:|---:|---:|---:|
| validation | 87.5% | 55.0% | 60.0% | 60.0% |
| held_out | 62.5% | 27.5% | 17.5% | 12.5% |
| challenge | 56.2% | 28.1% | 21.9% | 21.9% |
| cross_generator | 65.0% | 25.0% | 20.0% | 25.0% |

## Catastrophic failure rate (>50px), classical baseline vs. learned re-ranker

| Split | Classical baseline | Learned (seed 101) | Learned (seed 102) | Learned (seed 103) |
|---|---:|---:|---:|---:|
| validation | 7.5% | 22.5% | 22.5% | 20.0% |
| held_out | 25.0% | 60.0% | 67.5% | 65.0% |
| challenge | 34.4% | 62.5% | 62.5% | 59.4% |
| cross_generator | 25.0% | 50.0% | 60.0% | 60.0% |

The learned re-ranker roughly **doubles to triples the catastrophic-failure rate on every split**,
consistently across all 3 seeds. This is a large, decisive, reproducible effect — not noise.

## Why this happened (diagnosis, not an excuse)

`model/TRAINING_PROTOCOL.md` stated this as the likely outcome before training: 72 triplets, mined
from only 2 of V2's structural families (`dev_strip_anchor`, `dev_single_mat`), is a very small and
narrow training signal for a from-scratch CNN to generalize from onto 13 other families it never saw
during training. The training curves confirm this concretely: `experiments/embedding_reranker_v1/training_histories/training_history_seed*.json`
shows **training loss collapsing to ~0 within 10-15 epochs** while the internal early-stopping loss
(on the held-out `dev_dense_periodic` family) plateaus around 0.20–0.29 and does not track it down —
textbook overfitting on a training set this size, exactly as anticipated.

The practical consequence: re-ranking the classical top-12 candidates by an overfit embedding's cosine
similarity is closer to **injecting noise into an already-reasonable classical ranking** than to
correcting it. Since the classical baseline's top-1 pick is already correct roughly 70% of the time
pooled (`reports/V2_BASELINE_REPORT.md`), a noisy re-ranker has much more opportunity to break a correct pick
than to fix an incorrect one — which is exactly the doubled-to-tripled catastrophic-failure pattern
observed above.

This is a data-scale finding, not evidence that a learned re-ranker is a bad idea in principle for
this problem. The rotation/scale-drift failure mode `reports/V2_BASELINE_REPORT.md` identifies as the
classical baseline's biggest weakness is still the most promising target for a learned component — but
this experiment shows that fixing it needs either (a) meaningfully more training pairs than 24
development pairs provide, (b) synthetic/augmented triplets, or (c) a differently-scoped model (e.g.
one trained per-condition rather than to generalize across all of them from one small pool) — not a
straightforward "train the same architecture longer" fix, since the failure is a training-data-scale
problem, not an undertrained-for-more-epochs problem (training loss already reaches ~0).

## Disposition

Per `reports/V2_ARCHITECTURE_PLAN.md` section 8 (mandatory): this candidate is **not integrated**.
`pipeline/ranking.py::rank_classical` remains the sole production ranking path
(`pipeline/localize.py`'s default `ranking_mode="classical"` is unchanged). All artifacts from this
experiment — checkpoints, training histories, per-seed integration-gate JSON, per-pair learned-ranking
results — are filed under `experiments/embedding_reranker_v1/` rather than treated as a pipeline
component. `model/` itself remains in the repository as a working, tested package (architecture,
training, inference all run correctly end-to-end) — what failed is the *result* on this data scale, not
the code.

## What would change this verdict

Documented honestly as a forward pointer, not attempted in this pass (would need new data/training
runs, which is out of scope for this evaluation and would be its own new experiment):

- Training on a substantially larger set of hard-negative triplets (more development-scale pairs, or a
  supplementary synthetic triplet source not requiring new Reference/Search image generation).
- Testing whether the re-ranker helps when restricted to exactly the failure mode it might plausibly
  fix (rotation/scale-drift candidates only) rather than being asked to generalize across every
  structural family from a training set that only saw two of them.
- A regularization or transfer-learning approach better suited to ~70-triplet-scale data than training
  a randomly-initialized CNN from scratch.
