# experiments/ddis_diversity — P5, Deformable Diversity Similarity: REJECT at the diagnostic stage

**Date:** 2026-08-17. **No production change.** No harness was built, no sweep run, and the frozen
benchmark was used only as a source of failing pairs — nothing was tuned or selected.

## Summary

P5 was the last untested proposal in `reports/RESEARCH_SURVEY_SCORING.md`. Neither
`ACCURACY_90_CAMPAIGN.md` (nine experiments) nor `REACHABILITY_CAMPAIGN.md` (six) had run it.

**It performs at chance and is rejected.** On the 22 reachable failures, DDIS prefers the true
location on **11 of 22 — exactly 50%**, against a pre-registered bar of >70%.

| | value |
|---|---:|
| reachable failures tested | 22 |
| median ZNCC deficit to overcome | −0.0293 |
| median DDIS margin | **+0.0009** |
| DDIS prefers truth | **11/22 (50%)** |
| pre-registered bar | >70% |

A median margin of +0.0009 on a statistic ranging 0–1 is indistinguishable from zero.

## Why it was worth a thirteenth attempt

Twelve prior attempts all arbitrated by ZNCC or a monotone function of it — reweighting it,
prewhitening it, penalising its peaks, or replacing it with a score learned on this data. DDIS is
categorically different: it measures how many template patches find **distinct** nearest neighbours
in the candidate window, ignoring match quality entirely. At a true match most patches find unique
correspondences; at a periodic decoy they collapse onto a few repeating positions. That collapse is
the periodic signature measured directly, rather than inferred from a correlation peak — so it
carries information ZNCC provably does not.

The reasoning was sound. The signal is not there.

## Method, and why the bar was set before running

`experiments/reachability_verification/` established that the frozen benchmark's 35 failures split
into 13 unreachable and **22 reachable**, the latter with the true location at median rank 3 and a
median deficit of 0.029 ZNCC. That gave a precise, pre-committed target: *separate a 0.029 gap on
those 22 pairs.*

So this experiment measured **only** that separation — no end-to-end harness, no parameter sweep, no
integration path. Total cost ~3 minutes. Had the earlier experiments in this project been gated the
same way, several would have been closed far more cheaply.

Implementation: template patches of 8px at stride 4, each searching a ±12px neighbourhood of its own
position for the lowest-SSD match; the score is the fraction of *distinct* destinations. Match
quality is deliberately discarded so the measure cannot degenerate into another ZNCC variant.

## Honest limits of this result

- One implementation of DDIS, at one patch size and search radius. A different parameterisation
  might behave differently — but a signal sitting at exactly chance is not a promising starting
  point for tuning, and tuning it on the same 22 pairs used to measure it would be circular.
- The diagnostic tests only whether DDIS *prefers truth on failures*. It does not test damage to the
  121 currently-correct pairs, which is where every prior re-scorer actually lost. That test was not
  needed: a signal at chance cannot pass it.
- DDIS's published results are on natural-image template matching with occlusion and deformation.
  Periodic industrial texture is a different regime, and this is evidence it does not transfer —
  consistent with the survey's own warning that no benchmark evaluates these methods on periodic
  texture.

## Consequence: the survey is now fully explored

| Proposal | Status |
|---|---|
| P1 spectral prewhitening | REJECT — `parallel_pipeline/` |
| P2 lattice notching | REJECT — `parallel_pipeline/` §3 |
| P3 discriminability-weighted ZNCC | REJECT — `discriminability_weighted/` |
| P4 PSR | REJECT ×2 — `psr_confidence/` (but produced the threshold recalibration) |
| **P5 DDIS** | **REJECT — this experiment** |
| P6 frozen pretrained features | Not attempted; survey rates it last resort and coarse-only |

Every ranked proposal from the external survey has now been tested and rejected, with the single
exception of P6 — which the survey itself scoped as a coarse re-ranker that "can only ever be a
coarse re-ranker, never the locator", and whose 16px token resolution is far too coarse for a 5px
tolerance.

**Thirteen independent attempts at the selection problem have now failed.** The near-tie is real,
reachable, and small — and nothing in the classical or survey-recommended space has separated it.

## Reproduce

```
python -m experiments.ddis_diversity.diagnose
```
