# experiments/nms_spatial_diversity — is the candidate pool spatially degenerate? No. REJECT

**Date:** 2026-08-17. **Diagnostic only — no production change, nothing tuned or selected.**

## Hypothesis

`matching.top_k_peaks` suppresses an 8px neighbourhood around each peak. On a DRAM array the
correlation surface is itself periodic. **If the lattice pitch exceeds 8px, the "top 2 peaks" of a
hypothesis could be two adjacent lattice repeats in the same neighbourhood** rather than two
distinct candidate locations — capping recall for reasons unrelated to how many peaks are kept.

This was a real gap in the project's evidence: both prior experiments that touched pool width moved
the radius **down** (`wider_candidate_pool/` used 4px; `discriminability_weighted/pool_recall.py`
tested 8 and 4). A *larger* radius had never been measured.

## Result — the premise holds, the conclusion does not

The pitch does exceed the radius, on almost every pair:

- correlation-surface pitch: **median 14.0 px** (range 8–28)
- production NMS radius: **8 px**
- pitch exceeds radius on **21 of 24 pairs**

And it makes no difference whatsoever:

| NMS radius | median nearest-neighbour distance in pool | recall @k=2 | @k=4 | @k=8 |
|---:|---:|---:|---:|---:|
| **8 (production)** | 16.4 px | **0.8333** | 0.8333 | 0.8333 |
| 16 | 19.0 px | 0.8333 | 0.8333 | 0.8750 |
| 32 | 19.0 px | 0.8333 | 0.8750 | 0.8750 |
| 64 | 20.0 px | 0.8333 | 0.8750 | 0.9167 |
| 128 | 20.8 px | 0.8333 | 0.8333 | 0.8750 |

**Recall at equal k is identical — 0.8333 — across a 16× range of suppression radius.** The
pre-registered GO criterion ("a larger radius must raise recall at EQUAL k") is not met.

## Why the premise doesn't imply the conclusion

The pool is already spatially diverse without help from within-hypothesis NMS. Median
nearest-neighbour distance among pooled candidates is **16.4 px at r=8** — twice the suppression
radius, and comparable to what r=128 achieves (20.8 px). Two mechanisms explain it:

1. `deduplicate_by_location` already collapses candidates within 10px **across the whole pool**,
   after aggregation. Within-hypothesis suppression is largely redundant with it.
2. Spatial diversity comes from **hypothesis diversity**, not from peak-picking. 81 scale×rotation
   hypotheses × 2 PSF arms each contribute their own peaks, and those land in different places.

So the radius was never the binding constraint. The small gains visible at k≥4 are `k` doing the
work, and `experiments/wide_pool_rescoring/` already established that extra candidates convert to
zero rescues.

## Status

**REJECT.** Not a candidate for integration; nothing changed. Measured on `development` only —
sufficient, because the result is a flat line across the whole radius grid rather than a marginal
call, and because `experiments/reachability_verification/` has since shown pool recall on the frozen
benchmark is already **0.917**, leaving very little for any recall-oriented change to win.

**Value retained:** the pitch-vs-radius relationship is now measured (median 14px vs 8px) and the
"pool is spatially degenerate" hypothesis is closed with a mechanism, so it need not be re-derived.

## Reproduce

```
python -m experiments.nms_spatial_diversity.run --surface development
```
