# Experiment C: spatial context vs. periodic decoys

**Status: INCONCLUSIVE FOR A CLASSICAL FIX — evidence points to an information ceiling for
hand-crafted context measures, not proof no context helps at all.** Not integrated (this experiment
doesn't propose a pipeline change; it answers a diagnostic question that gates whether Experiment B
is worth pursuing and how to scope it).

## Methodology

Question: does a wider spatial context window around a candidate location contain genuinely
separating information between the true site and a periodic decoy — using only information a real
inference-time algorithm could compute from the search image itself (no ground truth fed into any
scoring decision, no injected fingerprints, no generator changes)?

`experiments/spatial_context/harness.py::periodicity_strength` measures how strongly a crop
autocorrelates with itself at a nonzero spatial lag (2D FFT-based autocorrelation, excluding the
trivial zero-lag peak) — high value = deep periodic interior with no nearby boundary; low value =
a boundary/strip/unique structure breaks the repetition. Defined `distinctiveness = 1 -
periodicity_strength` so that, matching the brief's framing, a **more positive** `separation =
distinctiveness(true) - distinctiveness(decoy)` means wider context favors the true location.

Dataset: the 45 largest known failures on the frozen benchmark (from
`outputs/reports/baseline_failure_decomposition.csv`, spanning all three failure categories —
candidate-generation, ranking, and genuine-ambiguity) plus a matched sample of 45 currently-correct
pairs as a sanity control. For each, extracted crops centered on the true location and on the
pipeline's actual winning (decoy) location, at 5 window sizes: 100/125/150/200/300px (1.0x-3.0x the
100px reference footprint).

## Result

| Window | Failure cases: mean separation | frac. favoring true | n valid |
|---:|---:|---:|---:|
| 100px (1.0x) | +0.0018 | 56.8% | 44 |
| 125px (1.25x) | +0.0026 | 55.0% | 40 |
| 150px (1.5x) | +0.0012 | 54.1% | 37 |
| 200px (2.0x) | -0.0084 | 39.1% | 23 |
| 300px (3.0x) | -0.0036 | 57.1% | 14 |

(Success-case control shows the same near-zero, non-trending pattern — separation values an order
of magnitude smaller than the 0-1 metric range, `frac_positive` consistently *below* 50% there,
which is expected: for already-correct pairs "true" and "decoy" aren't really competing, so this
measure isn't meaningful for them — included only as a sanity check that the measurement pipeline
itself isn't systematically biased.)

**Mean separation stays within +/-0.008 of zero at every window size, with no consistent upward
trend — if anything, it is noisier and briefly negative at 200px before recovering.** `frac_positive`
hovers at 39-57%, statistically indistinguishable from a coin flip. Sample size shrinks at larger
windows (44 -> 14) because many of the worst failures sit near the 1000px search-image edge, where a
300px-radius crop runs out of bounds — itself a mildly interesting side-finding (the worst failures
are not uniformly distributed across the image), but not large enough to change the conclusion at
the sizes where data does exist.

## Interpretation

**No reliable separating signal was found via this measure, at any tested context size up to 3x the
reference footprint.** This is consistent with an information ceiling for a hand-crafted, autocorrelation-based
notion of "context" — not proof that no information exists at any window size or via any
representation. Two important qualifications, stated honestly rather than overclaimed:

1. This tests one specific classical proxy (self-repetition strength via autocorrelation). It does
   not test whether a *learned* representation could extract a different, more subtle discriminating
   signal from the same wider crop (e.g. the specific pattern of line-position jitter or
   pattern-collapse events, which are generated independently per mat instance and are — by
   construction, per `generator/pattern_renderer.py` — genuinely different between any two mats,
   including two mats sharing the same preset). That is a real, physically-grounded reason a
   sufficiently-trained learned model might still succeed where this measure doesn't; it is exactly
   the mechanism `embedding_reranker_v1` was trying to learn (and failed to, from data starvation,
   not because the signal doesn't exist).
2. Only two representative points (true location, winning decoy) were compared per case, not a
   dense sweep of every periodic repeat — a genuinely exhaustive test would compare the true location
   against *all* competing repeats, not just the one that happened to win.

## What this means for Experiment B's scope

This tempers, but does not eliminate, the case for a learned candidate generator: it should not be
expected to succeed merely by "seeing more context" in the way a classical statistic does. If
pursued, it should be judged on whether it can learn the subtler per-mat jitter/collapse signature
directly (which this experiment did not test), not treated as a foregone conclusion either way.

## Production impact

None. This is a diagnostic analysis, not a pipeline candidate.
