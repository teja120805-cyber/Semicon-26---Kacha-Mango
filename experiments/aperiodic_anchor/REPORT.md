# experiments/aperiodic_anchor — anchor on the aperiodic sub-region to restrict the search: REJECT

**Date:** 2026-08-16. **Not integrated. Production stays at 77.6%@5px.** Frozen benchmark: **0 runs**.
`development`, `validate_fresh`: **0 runs**.

## Summary

The one structurally untried direction in this campaign: every prior change either **re-scored** the
pool or **enlarged** it. This **removes decoys** instead — match on the aperiodic sub-region only,
use it as a spatial prior, and discard full-template candidates far from it.

**REJECT. Zero rescues at every radius tested, 6–7 pairs broken.**

| restriction radius | acc@5px | Δ | rescued | broken | net | restriction fired |
|---:|---:|---:|---:|---:|---:|---:|
| ∞ (production) | **0.7000** | — | — | — | — | — |
| 150px | 0.5500 | −15.0pp | **0** | 6 | −6 | 39/40 |
| 80px | 0.5500 | −15.0pp | **0** | 6 | −6 | 38/40 |
| 40px | 0.5250 | −17.5pp | **0** | 7 | −7 | 36/40 |

## 1. Why this was worth trying

Two of `reports/RESEARCH_SURVEY_SCORING.md`'s own cited sources point here, and neither became one
of the P1–P6 proposals:

- **KLA US9830421B2** explicitly rejects device features that are "repeating" or "lacking
  uniqueness" and targets array corners and boundaries instead. Our own forensics agree: 89.8%
  accuracy on boundary-crossing crops against 54.4% on crops crossing neither.
- **Cognex US6324299B1** "sub-models" — match a sub-region rather than the whole model.
- **US20090103799A1** — classify the pattern, then *clamp the search range*.

And it is genuinely distinct from P3, which failed. P3 kept the full 100×100 window and reweighted
inside it, so the periodic 90% still dominated the ZNCC **denominator** even at weight ≈ 0.
Restricting the *support* removes that content entirely — it contributes nothing, not even to
normalization. The sub-window is located by reusing P3's lattice-shift dissimilarity map, which
P3's own report established is a working discriminability estimator (Gini 0.57, 44% more mass on
top-decile gradient pixels) even though weighting by it did nothing.

## 2. The diagnostic predicted the result before the end-to-end run

Stage 1 measured, per pair, the distance from the aperiodic prior's peaks to ground truth
(`d_truth`) and to the winner production actually chose (`d_chosen`). The GO criteria were written
into the script **before** it ran:

> GO if the prior sits nearer truth than the decoy on most failures **AND** covers nearly every
> correct pair (else restriction will break what already works).

Measured on 40 pairs:

| group | n | result |
|---|---:|---|
| usable (aperiodic ratio ≥ 1.15) | 40/40 | every Reference *has* an aperiodic region |
| **failures** | 12 | median `d_truth` 70.5px vs `d_chosen` 50.9px — prior nearer truth on only **5/12** |
| **correct** | 28 | median `d_truth` **0.8px**, but only **20/28** within 80px |

**Neither criterion was met** — 5/12 is not "most", 20/28 is not "nearly every" — so a net loss was
the prediction, and −6 is what happened. The stage-2 run is reported for completeness rather than
because the outcome was in doubt.

## 3. The mechanism — why sub-model anchoring cannot work here

The `d_truth` distribution on correct pairs is **bimodal**: median 0.8px (the prior lands essentially
exactly on truth) with a fat tail (8 of 28 beyond 80px). It is either spot-on or badly wrong, rarely
in between.

Critically, **the sub-template fails on the same pairs the full template fails on.** It is nearer
truth on only 5 of 12 failures — barely better than chance — while being sub-pixel accurate on most
pairs that already work. Its errors are **correlated** with the existing failure mode, not
complementary to it.

That is the fatal property, and it generalises beyond this implementation: a spatial prior is only
useful as a filter if its errors are *independent* of the thing it is filtering. Here both the
sub-template and the full template are driven by the same aperiodic content in the same image, so
when that content is ambiguous or corrupted, both go wrong together — and the prior then confidently
deletes the correct candidate. This is why zero rescues appear at every radius: on the pairs where
restriction could have helped, the prior was pointing at the decoy too.

**Consequence for the project:** the whole KLA/Cognex "anchor on non-repeating structure" family is
closed here, not because the doctrine is wrong, but because in this pipeline the anchor carries no
information the full template does not already have. A useful prior would have to come from a
genuinely different measurement channel.

## 4. Null control

`radius_px = ∞` keeps every candidate and must reproduce `pipeline.localize.localize` bit-for-bit
on x, y and confidence. Verified on 5 pairs: 0 mismatches. The baseline row reproduces production's
0.7000 on this surface exactly.

The restriction is also written to **never fail closed** — if the aperiodic ratio is below threshold,
if no prior peak exists, or if restriction would empty the pool, the unrestricted ranking stands.
That safety path is why 36–39 of 40 pairs were restricted rather than all 40.

## 5. Honest status

- **REJECT.** Not a candidate for integration. Production untouched.
- One 40-pair surface. Zero rescues across three radii with a consistent 6–7 pair loss does not need
  a second surface; a positive result would have.
- A confidence-gated variant (restrict only when the sub-template's own peak is decisive) is the
  obvious salvage attempt and is **not** recommended: §3 says the prior's errors are correlated
  with the pipeline's, so gating would mostly disable the mechanism on exactly the pairs it was
  meant to fix, buying back the losses without buying any rescues. It would also add another tuned
  threshold to a project that already has miscalibrated ones.

## 6. Run-count disclosure

- Frozen 156-pair benchmark: **0 runs.**
- `tune_degraded` (40): 1 diagnostic, 1 baseline, 3 end-to-end radii, 1 null-control pass.
- `development` (24), `validate_fresh` (40): **0 runs.**

## Reproduce

```
python -m experiments.aperiodic_anchor.run --surface tune_degraded
```
