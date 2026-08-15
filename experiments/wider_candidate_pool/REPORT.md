# Experiment: wider_candidate_pool

**Status: REJECTED — structurally a no-op, not integrated.** Not a failed attempt at an improvement;
a clean negative result that sharpens the diagnosis in `reports/ACCURACY_FORENSICS.md` Finding 2.

## What this experiment was

Forensics found that most periodicity-driven failures are `candidate_generation` failures: no
candidate near ground truth survives into the pool at all. The natural first hypothesis was that the
production defaults (`PEAKS_PER_HYPOTHESIS=2`, `SUPPRESSION_RADIUS_PX=8`) prune too aggressively,
discarding a correct-but-lower-scoring peak near a dense mat's ~5-8px pitch. This experiment widened
both (6 peaks/hypothesis, 4px suppression radius) and re-ran the frozen benchmark's gate splits
through the unmodified `candidate_generation`/`ranking`/`refinement` functions (no `pipeline/` code
changes — `localize()` doesn't expose these two parameters, so the experiment calls the lower-level
functions directly, mirroring `localize()`'s own orchestration).

## Result

**Every one of the 132 evaluated pairs (validation/held_out/challenge/cross_generator) produced a
bit-identical prediction to the baseline** — not just the same accuracy@5px, the same `error_px` to
better than 0.01px on every single pair. Runtime rose ~3.2-3.4x (0.51s → 1.65-1.75s/pair) for zero
behavioral change. Integration gate: **failed** (no split improved; `improved=false` everywhere;
runtime alone would have been within the 5x budget).

## Why this is a structural certainty, not bad luck

`pipeline/ranking.py::rank_classical` is `sorted(candidates, key=score, reverse=True)[0]` — pure
arg-max over the whole pool. The global arg-max across all 25 scale×rotation hypotheses is, by
definition, equal to the best of the 25 *per-hypothesis* top-1 peaks — and the top-1 peak of every
hypothesis is retained regardless of `PEAKS_PER_HYPOTHESIS` (as long as it's ≥1, which it always is).
Keeping more *lower-scoring* peaks per hypothesis, or loosening the intra-hypothesis NMS radius that
governs which of those lower peaks get extracted, cannot change which candidate has the single
highest score — so it cannot change the winner under pure top-1-score ranking, full stop. This holds
regardless of how the peaks/radius values are tuned.

## What this actually tells us (the useful part)

This rules out "retain more candidates" as a fix in isolation, and sharpens Finding 2's diagnosis:
for the periodicity failures, **the ZNCC score at the true location is genuinely not the highest
score in the correlation landscape** — it isn't that the right candidate gets pruned away before
scoring is considered; it's that scoring itself, at this resolution and pitch density, sometimes
rates a wrong periodic repeat higher than the true location. A wider pool only has a chance to help
if it's paired with something that can look *past* the top-1 classical score — e.g. a re-ranking
stage over a shortlist (`pipeline/ranking.py::rank_with_model` already has this shape: it takes the
top-N classical candidates and re-scores them by a different signal). This is exactly the
architecture family the existing (rejected) `embedding_reranker_v1` experiment already tested — its
rejection was a training-data-scale problem (72 triplets, 2 families), not evidence against the
reranking *shape* itself. A better-scoped future experiment, if the periodicity bottleneck is
revisited, should target *that* combination (wider shortlist + a properly-trained reranker over it),
not pool width with the current pure-argmax classical ranker.

## Production impact

None. `pipeline/` and `generator/` untouched.
