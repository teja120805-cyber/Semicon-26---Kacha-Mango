# experiments/alternative_scores — OTSDF/MACE correlation filters and phase-based matching: both REJECT

**Date:** 2026-08-17. Diagnostic only; **no production change.** No harness was built, no sweep run,
no end-to-end evaluation. The frozen benchmark was used only as a source of failing pairs — nothing
was tuned or selected. Consolidated findings: `experiments/BATCH_2026-08-17_REPORT.md` §C and §D.

## 1. The hypotheses, and the prior, both stated before running

Two measures were screened. Neither had ever been a P1–P6 proposal in
`reports/RESEARCH_SURVEY_SCORING.md`, but both sit in that survey's source list.

**OTSDF / MACE correlation filters.** MOSSE, MACE and UMACE are designed to give a sharp peak at the
target while *minimising average correlation energy everywhere else* — explicit sidelobe
suppression, aimed squarely at the "many near-equal peaks" problem this project actually has. The
Optimal Trade-off SDF adds a noise-tolerance parameter and, usefully, a built-in null control:

    H(f) = X*(f) / ( alpha*|X(f)|^2 + (1-alpha)*mean|X|^2 )

with `alpha = 0` reducing to the plain matched filter (production correlation) and `alpha = 1` to
the MACE / inverse filter.

**Phase-based matching**, the idea behind MS-POFT (2025) and phase congruency generally: local phase
is invariant to contrast and illumination in a way gradient magnitude is not. Implemented as a
log-Gabor quadrature pair producing a local-phase map, correlated instead of intensity. This is an
approximation of the published method and is labelled as such in the module.

**The honest prior, recorded in the module docstring before the run:** *both are expected to fail.*
`experiments/template_fidelity_ablation/` established that the discriminating aperiodic content
lives substantially at **low** spatial frequency, and both of these suppress it — OTSDF by dividing
out low-frequency amplitude, phase by discarding amplitude entirely. That is the mechanism that sank
P1. They were run because they are cheap, and because "we expect it to fail" is not a measurement.

**The bar, pre-registered and identical to DDIS's:** separate the median 0.029 ZNCC deficit on the
22 reachable failures from `experiments/reachability_verification/`, with **>70%** preference for
truth to justify building a harness. Chance is 50%.

## 2. Result — OTSDF sits at chance

| measure | median margin | prefers truth |
|---|---:|---:|
| ZNCC (reference) | −0.0293 | 0/22 (0%) |
| OTSDF α=0.0 *(null control)* | −0.0275 | 4/22 (18%) |
| OTSDF α=0.3 | −0.0106 | 11/22 (50%) |
| OTSDF α=0.6 | −0.0105 | 11/22 (50%) |
| OTSDF α=0.9 | −0.0112 | 10/22 (45%) |
| OTSDF α=1.0 (full MACE) | −0.0000 | 11/22 (50%) |

The null control behaves correctly — α=0 is the matched filter and tracks ZNCC's sign pattern, with
a median margin of −0.0275 against ZNCC's −0.0293. Everything else sits at **chance**. Nothing comes
near the 70% bar.

The α=1.0 row is the informative one: the median margin collapses to **−0.0000**. The inverse filter
makes every location look equally good — the known degenerate behaviour of MACE on noisy data. It
does not prefer the decoy; it stops discriminating at all.

## 3. Result — phase-based matching is materially *worse* than chance

**4/22 (18%)**, median margin **−0.0138**. It does not merely fail to beat ZNCC; it prefers the
wrong location more often than a coin would.

## 4. Why, and what it closes

Both failures share one mechanism, and it is the mechanism that was predicted:

> The aperiodic content that distinguishes one lattice position from another — mat boundaries, strip
> edges, array transitions — is a **large-scale intensity change**, not fine texture. OTSDF divides
> low-frequency amplitude out of the response; phase discards amplitude entirely. Both therefore
> remove the discriminating signal and leave the periodic cell texture to arbitrate, which is
> exactly what already fails.

Phase being the more severe of the two (18% vs 50%) is consistent with it being the more complete
discard: OTSDF attenuates amplitude, phase throws it away.

This is the same mechanism recorded in `REACHABILITY_CAMPAIGN.md` §5, which closed frequency-domain
reweighting with three independent causes. These two results are further instances of it, arrived at
from a different family of methods. The standing requirement stated there applies unchanged: **any
future proposal in this family must show it preserves low-frequency boundary content.** Neither of
these does, and neither was ever going to.

The complementary result from the same session is `experiments/lowfreq_1d/`, which attacks the
problem from the opposite side — using the low-frequency information ZNCC discards, rather than
removing more of it. That is the direction this experiment's mechanism points toward.

## 5. Status

- **OTSDF / MACE: REJECT.** At chance at every α tested; degenerate at α=1.0.
- **Phase-based matching: REJECT.** 18%, materially worse than chance.
- **Nothing to integrate**, and nothing worth a harness — the pre-registered bar existed precisely
  to decide that before any harness was built, and it did, at the cost of one diagnostic run.
- With these two, `BATCH_2026-08-17_REPORT.md` counts **fifteen independent attempts** at the
  selection problem that have failed or fallen short: nine in `ACCURACY_90_CAMPAIGN.md`, four in
  `REACHABILITY_CAMPAIGN.md`, and the two measured here. The scoring survey is fully explored, and
  the two source-list ideas that never became proposals — correlation filters here, directional
  uniqueness in `experiments/axis_decomposition/` — are now tested too.

## Honest limits of this result

- One implementation of each measure. The OTSDF α sweep spans the full range from matched filter to
  inverse filter, so that family is covered end to end; the log-Gabor front end is a single setting
  and is an approximation of MS-POFT, not a reimplementation.
- The diagnostic asks only whether a measure *prefers truth on failures*. It says nothing about
  damage to the currently-correct pairs, which is where every prior re-scorer actually lost. That
  test was not needed: a measure at or below chance cannot pass it.

## Reproduce

```
python -m experiments.alternative_scores.diagnose
```
