# Experiment: periodicity (gradient / ensemble candidate scoring)

**Status: REJECTED (both variants), not integrated.** Neither alternative correlation
representation improves on raw-intensity ZNCC; both are net-negative on rescue/break and one fails
the runtime budget outright.

## What this experiment was

`reports/ACCURACY_FORENSICS.md` diagnosed periodicity-driven `candidate_generation` failures as a
genuine scoring problem: the true location's raw-intensity ZNCC score is sometimes not the highest
in the correlation landscape (confirmed by `experiments/wider_candidate_pool/` — retaining more
candidates never changes the arg-max winner, since a wider pool can't fix which score is highest).
This experiment tests whether a **different correlation representation** changes which location
scores highest: `experiments/periodicity/harness.py` mirrors `pipeline/candidate_generation.py`'s
exact structure (same template construction, same top-k/NMS peak extraction — only the score map's
representation differs), with two variants:

- **`gradient`**: correlate Sobel-gradient-magnitude images instead of raw intensity.
- **`ensemble`**: average the intensity and gradient score maps per hypothesis (candidate must look
  good under both representations).

Evaluated over the frozen benchmark's gate-relevant splits (132 pairs), same production
`pipeline/refinement.py` for the final subpixel step.

## Result

| Variant | validation | held_out | challenge | cross_generator | rescue | break | net | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 90.0% | 55.0% | 68.8% | 65.0% | - | - | - | 1x (0.51s/pair) |
| `gradient` | 90.0% | **50.0%** | **50.0%** | 60.0% | 1 | 10 | **-9** | ~1.9x |
| `ensemble` | **87.5%** | 55.0% | **59.4%** | 65.0% | 0 | 4 | **-4** | ~3.5-5.2x (fails 5x budget) |

Both variants are worse than baseline on every split they change at all. `gradient` is a clear,
substantial regression (net -9, held_out and challenge both drop by ~19pp). `ensemble` is a milder
but still net-negative regression, and its runtime (roughly doubling the per-hypothesis cost, since
it computes both score maps) pushes some splits over the gate's 5x runtime budget.

## Why this is informative, not just a failed guess

Gradient/edge representations lose information that raw intensity ZNCC actually uses productively
for this synthetic pattern — likely because the periodic lines/contacts render at nearly uniform
edge strength regardless of which repeat they belong to, so gradient magnitude is *more* ambiguous
between repeats than intensity, not less (intensity carries subtle brightness/contact-pattern
differences from jitter and pattern-collapse that gradient magnitude discards). Averaging with
intensity (`ensemble`) doesn't fix this because it still lets the gradient term drag a correct
intensity-only decision toward a wrong one whenever gradient disagrees.

**This rules out "try a different classical correlation representation" as a fix for periodicity**,
at least for these two representations. Combined with `wider_candidate_pool`'s finding (more
candidates can't help under arg-max ranking) and `embedding_reranker_v1`'s finding (a learned
embedding failed only because of data-scale, not because the architecture shape is wrong), the
weight of evidence now points toward periodicity requiring either genuinely more image context per
candidate (not tested here) or a properly-data-scaled learned representation (per
`reports/DATASET_AUDIT.md` section 5's contingent recommendation) — not a cheaper classical
scoring tweak.

## Production impact

None. `pipeline/` untouched; both variants live entirely in `experiments/periodicity/`.
