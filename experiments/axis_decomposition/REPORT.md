# experiments/axis_decomposition — the residual failures are a ONE-DIMENSIONAL problem: hypothesis SUPPORTED

**Date:** 2026-08-17. Diagnostic only; **no production change** and nothing here is a candidate for
integration. It changes what is worth trying next, not the pipeline. Consolidated findings:
`experiments/BATCH_2026-08-17_REPORT.md` §B, where this is recorded as the most valuable result in
that batch.

## 1. The question, stated before running

US8073242B2 (SEM addressing-point recipes) is cited in `reports/RESEARCH_SURVEY_SCORING.md` §2 —
*"ignoring the uniqueness of any APs in the X-direction and attaching importance only to the
uniqueness of APs in the Y-direction"* — flagged there with the note "our two axes may be separately
solvable", and never turned into a proposal.

The patent's own mechanism does not transfer, and the module says so up front: it is a
**recipe-design** strategy for choosing where to place an addressing point, and here the crop is
given rather than chosen. What transfers is the underlying question, which is cheap to answer and
had never been asked of this data:

> **When the pipeline picks the wrong location, is the error along a lattice axis, or in an
> arbitrary direction?**

**The pre-registered pass condition**, written into the script before it ran: *a 1-D framing is
supported only if the along-axis fraction is high **and** the cross-axis residual is small* — the
second half matters as much as the first, because a large along-axis fraction with a large
cross-axis residual would still be a 2-D problem. If the errors are isotropic, the framing is wrong
and the direction is closed.

Measured on the same 22 reachable failures used by every other diagnostic this session
(`experiments/reachability_verification/`), decomposing the error vector (chosen − truth) onto the
lattice basis estimated from the template itself.

## 2. Result — both halves of the condition are met, decisively

| measurement | value |
|---|---:|
| median along-lattice-axis fraction of the error | **1.000** |
| failures >80% along a lattice axis | **18 / 22** |
| failures >90% along a lattice axis | 17 / 22 |
| **median cross-axis residual** | **1.47 px** |
| within a quarter-period of an integer multiple of the pitch | 17 / 22 |

At the median, **100% of the error lies along a lattice axis.** The pipeline already has the
cross-lattice coordinate right to ~1.5px — comfortably inside the 5px tolerance — and is confused
only about *which lattice period* it sits on, along a single measurable direction.

The last row is the same statement made a different way: the error is not just directionally aligned
but quantised, landing within a quarter-period of an integer multiple of the pitch on 17 of 22.

## 3. Why this matters more than the measurement

Every disambiguation attempt in this project so far has searched or scored in **2-D**. The actual
residual ambiguity is a **discrete 1-D index** — roughly 20–40 candidate period offsets along one
measurable direction.

A disambiguator therefore only needs to be informative along that one axis. That is a far weaker
requirement than any of the thirteen rejected approaches faced, and it is the reason this diagnostic
is worth more than any of the scoring measures screened alongside it.

## 4. It also explains the escalation result mechanistically

A scale-hypothesis quantisation error accumulates into displacement across the image. At the
production step of 0.2 in ~10× scale (2%), the accumulated shift across a 1000px Search image is
~20px — against a measured lattice pitch of ~14px, that is **more than one full period**. Tripling
grid density cuts the accumulated shift to well under half a period.

So `experiments/gated_escalation/`'s rescues and this 1-D finding are the same phenomenon seen
twice. The ~14px pitch is independently corroborated: `REACHABILITY_CAMPAIGN.md` §7 measured a
correlation-surface pitch of median 14px while rejecting a larger NMS radius.

Two existing forensics results also line up:

- **Finding 3** (the sawtooth against grid alignment) — this is that sawtooth measured in the error
  vector rather than in accuracy.
- **Finding 2** (62–85% of periodicity failures landing near integer pitch multiples) — measured
  here as 17/22 = 77%, inside that range.

`experiments/hypothesis_ensemble/` reaches the same place from the third direction: a
lattice-aligned decoy matches across a whole neighbourhood of hypotheses while the truth matches
only at very nearly the correct one, so **adding** hypotheses helps and **aggregating over** them
does not.

## 5. Status

- **Hypothesis SUPPORTED.** The residual ambiguity is one-dimensional and discrete.
- **Nothing to integrate.** This is a diagnostic; it proposes no change and measures no candidate.
- It is the clearest open lead the project has, and it is cheap to act on — the requirement it
  places on a disambiguator is materially weaker than the one thirteen rejected approaches failed.
- The first attempt to exploit it, `experiments/lowfreq_1d/`, **did not clear its own
  pre-registered bar** (68% against >70%). The framing being correct does not make any particular
  signal work, and that experiment is the evidence.

## Honest limits of this result

- 22 pairs, all failures. The decomposition says nothing about the currently-correct pairs and does
  not establish that a 1-D disambiguator would leave them intact — the failure mode that has sunk
  every re-scorer in this project.
- The lattice basis is estimated from the template
  (`experiments/discriminability_weighted/weights.py::estimate_lattice_vectors`), and the along-axis
  fraction is taken against whichever of the estimated vectors aligns best. Pairs with no estimable
  lattice are recorded and excluded rather than counted as aligned.
- "Along a lattice axis" is a statement about direction, not about which period is correct. It
  narrows the search; it does not resolve it.

## Reproduce

```
python -m experiments.axis_decomposition.diagnose
```
