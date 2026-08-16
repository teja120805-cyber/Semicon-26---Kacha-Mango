# experiments/hough_subpatch_voting — REJECT (true no-op, verified directly)

## Summary

**Verdict: REJECT.** Sub-patch geometric-consistency re-ranking never once disagreed with
`rank_classical`'s top-1 choice — confirmed by direct per-pair coordinate diff, not just
matching aggregate accuracy. This holds both at the dev-selected config (`top_k=3, beta=0.0`,
a literal no-op by construction) **and** at an exploratory full-benchmark check with a much
more aggressive config (`top_k=8, beta=0.2`, tested on all 156 pairs): **zero coordinate
changes anywhere.** Production is untouched.

## 1. The idea

For each of the top-K classically-ranked candidates, decompose the reference template
(rendered at that candidate's scale/rotation hypothesis) into 5 overlapping sub-patches (4
corner quadrants + center), independently re-match each sub-patch against a small local window
of the search image around where it should fall if the candidate is genuinely correct, and
combine the sub-patch agreement into a re-ranking bonus (`score + beta * mean_subpatch_score`).
The hypothesis: a genuinely correct match should have all sub-patches independently confirm the
same alignment, while a high whole-template score that comes from periodic/textural similarity
might not hold up sub-patch by sub-patch. Unlike the project's prior `periodicity/` experiment
(alternative whole-template scoring functions, both rejected) or `wider_candidate_pool`
(structural no-op under pure arg-max because MORE candidates doesn't change WHICH one wins),
this specifically targets `candidate_ranking`/`genuine_ambiguity` failures — cases where the
correct candidate is already in the classical pool but a different one outscores it
(`reports/ACCURACY_FORENSICS.md`'s failure taxonomy, 11 + 7 = 18/156 pairs). It structurally
cannot help `candidate_generation` failures (31/156) where the truth never enters the pool.

## 2. Implementation

`subpatch.py::subpatch_consistency` — verified directly to produce real, varying scores (not a
degenerate constant): on a sample dev pair (`dev_dense_periodic_000`), the top-8 classical
candidates' consistency scores ranged 0.6824–0.7942, clearly discriminating between candidates.
`rank_subpatch(ranked, reference, search, top_k, beta)` re-scores only the top-K already-
classically-ranked candidates and re-sorts them; candidates beyond top-K are untouched.
`harness.py::localize_subpatch` is structurally identical to `pipeline.localize.localize`,
inserting this re-ranking pass between classical ranking and the center tiebreak — candidate
generation, dedup, tiebreak, and refinement are all unmodified production calls.

## 3. Evaluation

### Dev sweep (n=24, top_k ∈ {3,5,8} × beta ∈ {0, 0.1, 0.2, 0.4})

Every single one of the 10 tested configurations produced **exactly** `acc@5px=0.583,
mean_error_px=111.50` — identical to `beta=0` down to the decimal. This is not a rounding
coincidence: the consistency scores vary meaningfully per-candidate (confirmed above), but on
every one of the 24 development pairs, the candidate `rank_classical` already picks as #1 also
happens to have the highest sub-patch consistency among the top-K considered — so re-ranking
never changes the winner. The dev-only selection procedure correctly (if unhelpfully) chose
`top_k=3, beta=0.0`.

### Full frozen benchmark at the dev-selected (no-op) config

Pooled accuracy@5px: 0.7436 → 0.7436, bit-identical (as expected — `beta=0.0` cannot do
anything else).

### Exploratory check: does a much more aggressive config ever disagree, at full scale?

Since the dev sweep never got a chance to demonstrate disagreement (0/24 dev pairs flipped even
at `beta=0.4`), a natural question is whether a wider net over more data would find one.
Ran **`top_k=8, beta=0.2`** (a stronger, wide-net config never selected by the disciplined
dev-only procedure) directly over the full 156-pair frozen benchmark, purely as a diagnostic —
**not** used as the experiment's official gate verdict, to keep the dev-only tuning discipline
intact for the number that matters.

**Result: 0/156 pairs had any coordinate change whatsoever.** Every single pair's classical
top-1 candidate was also the sub-patch-consistency argmax among its top-8, everywhere in the
benchmark, at every split.

## 4. Why this didn't work

The sub-patch consistency signal is real (it varies meaningfully and would, in principle,
discriminate between candidates) but it never disagrees with the classical whole-template score
on this benchmark's actual `candidate_ranking`/`genuine_ambiguity` failures. The likely
mechanism: when the classical ranker picks the wrong candidate over the true one, it does so
because that wrong candidate is a genuinely strong structural match — periodic DRAM mat
patterns are self-similar not just in aggregate (whole-template ZNCC) but also locally
(sub-patch level), since a periodic repeat looks like the true structure at every spatial scale
that matters here, not just the coarse one. Splitting the template into quadrants doesn't add
information the classical matcher didn't already have, because the sub-patches themselves are
still drawn from the same locally-periodic structure. This rules out "geometric consistency
across sub-regions of the SAME hypothesis" as a way to break periodic ties — a structurally
different (and now also ruled out) idea from the whole-template scoring-function variants
`experiments/periodicity/` already rejected.

## Reproduce

```
cd experiments/hough_subpatch_voting
python run_experiment.py            # official dev-selected result
python /path/to/exploratory_subpatch.py   # optional: exploratory full-benchmark check (see outputs/)
```

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_subpatch.csv`,
`outputs/subpatch_metrics.json`, `outputs/integration_gate_result.json`,
`outputs/per_pair_results_subpatch_exploratory.csv`, `outputs/exploratory_gate_result.json`.
