# Accuracy-improvement campaign — pooled accuracy@5px 74.36% toward ~90%

> **Correction (2026-08-17) — the baseline this campaign was measured against has since moved.**
> 74.36%@5px (n=156) was the verified production baseline *at the time of this campaign* and is the
> number every result below is compared against. Production is now **77.6%@5px (n=156)**, raised
> after this campaign by PSF-matched dual-arm candidate generation
> (`reports/GATE_EXCEPTIONS.md` exception 3, `experiments/psf_gated_selection/REPORT.md`). Every
> rejection below stands — none of the nine ideas was integrated — but read "the production
> baseline" throughout as the historical 74.36%, not the current figure.

**Status: goal not reached. This campaign left the then-current production baseline
(74.36%@5px, n=156) unchanged.**
Nine independent, evidence-grounded ideas were designed, implemented, and rigorously
benchmarked in sandbox-isolated `experiments/` folders, per standing instructions: never modify
`pipeline/`, `generator/`, or `model/`; keep every idea in its own separate, fully-documented
experiment; never fabricate a result. All nine came back REJECT after full evaluation - several
looked promising at intermediate stages and are documented in detail below precisely because
understanding *why* they failed is itself valuable for whoever picks up this problem next. This
document is the honest, consolidated summary of that work; each experiment's own `REPORT.md` has
the full technical detail, all per-pair data, and reproduction instructions.

## How every result here was produced (so the numbers can be trusted)

- **Sandbox data was verified byte-for-byte against the user's real production machine** before
  any experiment ran: `opencv-python-headless==5.0.0.93` pinned, all 322 data files staged from
  the project's `data/` directory, baseline re-run in sandbox and confirmed
  to match the user's real `outputs/reports/per_pair_results.csv` to full float precision
  (`accuracy_at_5px: 0.7435897435897436`, n=156) - this is the number every experiment below is
  compared against.
- **Dev-only tuning discipline**: any hyperparameter sweep (thresholds, gammas, radii) was tuned
  exclusively on the 24-pair `development` split, exactly once, then the single chosen
  configuration was run one time over the full frozen 156-pair benchmark
  (`development`+`validation`+`held_out`+`challenge`+`cross_generator`). No experiment tuned
  against the same data its final number was reported on.
- **The unmodified integration gate** (`evaluation/benchmark.py::run_integration_gate`, the exact
  same 7-criterion gate used for the two production changes that DID get integrated) was run
  against every candidate.
- Every experiment imports `pipeline/`, `generator/`, `model/` unmodified - none of them ever
  edited production code, and production remains exactly as delivered (`rank_classical`,
  `DEFAULT_SCALE_HYPOTHESES`, `DEFAULT_ROTATION_HYPOTHESES`, `apply_center_tiebreak`, etc.
  untouched).

## Results table

| # | Experiment | Mechanism | Verdict | Pooled acc@5px (baseline -> candidate) |
|---|---|---|---|---|
| 1 | [`cross_hypothesis_consensus_rerank`](cross_hypothesis_consensus_rerank/REPORT.md) | Re-rank by cross-hypothesis spatial support density | REJECT - clean no-op | 0.7436 -> 0.7436 (bit-identical) |
| 2 | [`subpixel_grid_refinement`](subpixel_grid_refinement/REPORT.md) | Joint scale/rotation subpixel interpolation | REJECT - real but sub-threshold | 0.7436 -> 0.7436 (156/156 pairs shifted <0.31px, none crossed 5px) |
| 3 | [`hough_subpatch_voting`](hough_subpatch_voting/REPORT.md) | Re-rank by sub-patch geometric self-consistency | REJECT - true no-op, verified directly | 0.7436 -> 0.7436 (0/156 coordinate changes, even at aggressive settings) |
| 4 | [`learned_reranker_v2`](learned_reranker_v2/REPORT.md) | CNN embedding re-ranker, 7.5x more training data (537 vs. 72 triplets) than the original rejected attempt | REJECT - catastrophic regression, all 3 seeds | matched-splits 0.7727 -> 0.386-0.409 |
| 5 | [`keypoint_candidate_fusion`](keypoint_candidate_fusion/REPORT.md) | ORB keypoint + RANSAC candidate proposal, fused into the pool | REJECT - real proposals (56/156 pairs), never win | 0.7436 -> 0.7436 (0/156 wins) |
| 6 | [`pyramid_periodicity_search`](pyramid_periodicity_search/REPORT.md) | Coarse (blurred) candidate proposal + local full-res refinement | REJECT - finds the right spot, scores it too low to win | 0.7436 -> 0.7436 (bit-identical, verified per-pair) |
| 7 | [`prominence_rerank`](prominence_rerank/REPORT.md) | Re-rank by peak prominence vs. generic local annulus | REJECT - real signal, net HARMFUL when active | dev-selected: 0.7436 -> 0.7436 (no-op); exploratory active config: 0.7436 -> 0.6795 |
| 8 | [`pitch_aware_prominence`](pitch_aware_prominence/REPORT.md) | Prominence vs. a per-pair MEASURED periodic pitch (surgical follow-up to #7) | REJECT - passed 6/7 gate criteria on production seed, **failed second-seed validation** | production seed: 0.7436 -> 0.7500 (near-miss); independent second seed: 0.6618 -> 0.6324 (net regression, 0 rescues / 4 breaks) |
| 9 | [`pitch_aware_prominence_v2`](pitch_aware_prominence_v2/REPORT.md) | Penalty-only fix for #8 (`score + gamma*min(prominence,0)`, never a bonus) | REJECT - fix confirmed fully safe (all 4 second-seed breaks fixed), but a **complete no-op everywhere** (0/156 pairs changed at any tested strength up to 4x #8's peak gamma) | 0.7436 -> 0.7436 (bit-identical, both seeds) |

None of the nine passed. #8 and #9 together are worth reading in full even though both are
REJECT - #8 is the closest thing to a real lead this campaign produced, and #9's diagnosis-then-
fix shows conclusively that #8's apparent improvement was entirely an artifact of a flawed
formula (a symmetric bonus/penalty that let the wrong candidate get an outsized *reward*, not a
real periodicity-based penalty signal): every one of the 10 pairs #8 ever changed, across both
seeds, was traced to that same specific bonus-driven flip, with zero exceptions. Fixing the flaw
removes all the risk - but also removes the entire effect, confirming there was never a real
periodicity signal being captured. This is an important, hard-won piece of evidence about how
easy it is for a flawed re-scoring formula to look like progress on a single-seed benchmark, and
it fully closes out the "peak-prominence reranking" family of ideas (`prominence_rerank`, #8,
#9) as a productive direction with this scoring mechanism.

> ## ⚠️ CORRECTION (2026-08-16) — the section below is WRONG
>
> `experiments/oracle_ceiling_diagnostic/` measured the claim in the next section directly, on
> every failing pair, and **falsified it**. The "4x gap" (0.187 at ground truth vs. 0.775 at the
> decoy) does not occur anywhere in the benchmark: across all 39 evaluable failures the true
> location scores **0.612–0.937** under an ideal warp, and the decoy's largest advantage is a
> ratio of **1.048** — every failure is a *near-tie*, not a blowout. The 0.187 figure was produced
> inside `pyramid_periodicity_search`'s own blurred coarse stage, not measured at full resolution
> under a fitted warp, and was then generalized from one pair to the whole population.
>
> The correct decomposition of the 40 failures is 45% candidate-generation (true location never
> proposed), 22.5% tie-break (true location is the runner-up, losing by <0.01), and 32.5% scoring
> (the real tail the original claim mistook for the whole). See that experiment's `REPORT.md`.
> Read the section below as a record of what was believed during the campaign, not as fact.

## The central, repeatedly-confirmed finding

Four independent, structurally different mechanisms (#1, #3, #5, #6 above - spatial consensus,
sub-patch self-consistency, sparse keypoint matching, and coarse-to-fine resolution search)
converged on the same result: **when the classical winner is wrong, it is almost never because
the true location is unreachable, ambiguous, or narrowly out-ranked.** It's because the wrong
candidate's raw ZNCC score is *genuinely, substantially* higher than the true location's -
verified directly, not inferred, in `pyramid_periodicity_search`: a coarse-to-fine search found a
candidate **0.4px from ground truth**, and it scored 0.187, while the classical grid's own
(8.5px-off) winner scored 0.775 - a 4x gap, nowhere near either tie-break epsilon
(`TIE_SCORE_EPSILON=1e-6`, `MULTIWAY_TIE_SCORE_EPSILON=0.005`).

This has a real, testable consequence: **any fix that still arbitrates by raw (or lightly
adjusted) ZNCC score cannot work**, because the problem isn't discovery or ranking - it's the
score itself. Experiments #7 and #8 tried to directly discount scores that look
"non-distinctive" (nearby competing peaks); #7's generic version was net-harmful (false positives
on ordinary correct matches), #8's more surgical, per-pair-measured-pitch version looked
promising on one dataset draw but failed to generalize to a second. This is consistent with the
project's own pre-existing `periodicity/` experiment (gradient-domain and gradient+intensity
ensemble scoring, both already rejected before this campaign) - between that result and this
campaign's four, the evidence that **classical ZNCC-based re-scoring, in any form tried so far,
cannot reliably fix this** is now quite strong. A fix would need either a fundamentally different
scoring representation (not just a re-weighting of ZNCC) or a fundamentally different learned
approach - and the learned approaches tried in this project (`embedding_reranker_v1`,
`learned_candidate_generator`, and this campaign's `learned_reranker_v2` at 7.5x more data) have
all failed too, for reasons unrelated to data volume (`learned_reranker_v2/REPORT.md` section 4
lists what those reasons might be, none confirmed).

## Honest assessment of the ~90% target

No experiment in this campaign moved the production pipeline off the verified 74.36%@5px (n=156)
baseline it started from (production has since reached 77.6% by other means — see the correction at
the top of this document). The two structural
fixes that DID work and are already integrated (A2 `scale_range_v1`, widening the scale
hypothesis grid to the literal 9:1-11:1 span, and A6 `multiway_tiebreak_v1`, the multiway-gated
center tie-break) were both from before this campaign and addressed narrower, more mechanical
issues (hypothesis-grid quantization and genuine multi-way periodic ties) than the dominant
remaining failure mode this campaign focused on (candidate-generation-stage periodicity, ~20% of
all failures per `reports/ACCURACY_FORENSICS.md`). Closing the gap from 74.36% to ~90% would
require either substantially more benchmark data (to escape the small-sample noise this
campaign's #8 result shows is a real risk at n=156, and specifically the `validation` split's
90%-of-40-pairs ceiling that blocked both #8 and the pre-existing `finer_hypothesis_grid`
near-miss from passing the gate) or a genuinely different matching/scoring approach than
ZNCC-based classical correlation and the learned alternatives tried so far - not incremental
re-ranking or candidate-fusion tricks layered on top of the existing scoring function, which this
campaign now provides fairly comprehensive negative evidence against.

## What wasn't tried (for whoever picks this up next)

- **A properly-regularized version of #8** (pitch-aware prominence): its failure mode was
  overfitting to one draw's specific pairs, not a fundamentally broken idea - the per-pair
  measured-pitch mechanism is more principled than #7's generic version, and a much more
  conservative `gamma` combined with evaluation across 3+ independent seeds (rather than 1) might
  find a genuinely robust operating point, or might confirm the effect is simply too fragile to
  use. This campaign didn't have budget to run a 3+ seed sweep on top of everything else tried.
- **A completely different scoring representation** (not gradient-domain or intensity-ensemble,
  both already rejected in `periodicity/`) - e.g. a learned but *fixed, pretrained* feature
  descriptor (not trained from scratch on this project's ~500-pair synthetic data, which
  `learned_reranker_v2` and `embedding_reranker_v1` both show doesn't work) matched via a metric
  more robust to periodic self-similarity than raw ZNCC.
- **Expanding `validation` beyond 40 pairs** specifically, to remove the ceiling effect that has
  now blocked two different near-miss results (`finer_hypothesis_grid` pre-campaign, and this
  campaign's `pitch_aware_prominence`) from passing the gate on that one criterion alone.

## Reproduce anything in this table

Each experiment folder is self-contained: `cd experiments/<name> && python run_experiment.py`
(and, for #8 only, `python validate_second_seed.py` for the second-seed check). None require
network access beyond what's already installed; all read from the shared `data/` directory
read-only and write only under their own `outputs/`.
