# experiments/wide_pool_rescoring — wider pool + re-scoring, tested together: REJECT, decisively

**Date:** 2026-08-16. **Not integrated. Production stays at 77.6%@5px.** Frozen benchmark: **0 runs**.

## Summary

Two things the project had each rejected separately were tested **together** for the first time,
because each was useless for a reason the other supplies:

- `experiments/wider_candidate_pool/` — widening was a **structural no-op** under a pure arg-max
  ranker, so it could never help alone.
- `experiments/discriminability_weighted/` (P3) — a re-scorer that looks past the top score, which
  had almost nothing to work with because **74% of failures are unreachable** (ground truth is not
  within 5px of any pooled candidate).

Widening raises recall from 0.750 to 0.900 on the degraded tuning surface. So the question was
whether a re-scorer can convert that recall into accuracy.

**It cannot, and the failure is unusually clean.** Across **144 configurations** (k × scheme × α ×
tie_eps × max_group), on 40 pairs:

| pool width k | recall | best acc@5px | worst acc@5px | **rescued** | max broken |
|---:|---:|---:|---:|---:|---:|
| 2 (production) | 0.750 | 0.700 | 0.625 | **0** | 3 |
| 4 | 0.800 | 0.700 | 0.600 | **0** | 4 |
| 8 | 0.800 | 0.700 | 0.575 | **0** | 5 |
| 12 | **0.900** | 0.700 | 0.550 | **0** | 6 |

**Zero rescues in 144 configurations**, while breaks rise monotonically with recall. Every extra
candidate the widened pool makes visible is one the re-scorer prefers *over* the truth. The
weighted score does not merely fail to discriminate — over this candidate set it has **negative**
discriminative value, and handing it more candidates strictly hurts.

This is the cleanest possible refutation of the "recall is the bottleneck, so widen and re-rank"
hypothesis. Recall was genuinely the bottleneck; lifting it did not help, because the second half of
the plan does not work.

## 1. Method

`run.py` captures the top 12 peaks per (scale, rotation) hypothesis for both PSF arms in one pass
per pair, then derives every narrower k by truncation. That is exact, not an approximation: greedy
NMS is prefix-stable, asserted directly in `pool_recall._assert_prefix_stable` rather than assumed.

Production's PSF arm choice is made at **production width (k=2)** and the chosen arm is then
widened. Re-deciding decisiveness on the wide pool would silently alter the integrated dual-arm
rule (gate exception 3), which is a different change and must not ride along inside this one.

Null control: `k=2, α=0, tie_eps=0` reproduces production. Verified — the baseline row is
production's own 0.700@5px on this surface.

## 2. Side finding — a documented conclusion has partly expired

The no-op check inside `run.py` reported `argmax identical to production: False` at k=4, 8 and 12.
That contradicts `experiments/wider_candidate_pool/`, whose result
`reports/PROJECT_STATUS.md` records as a settled structural fact.

`noop_expiry.py` tested the attribution directly instead of reasoning about it — re-running the
widening with the multiway tier disabled:

| k | multiway tier | predictions changed vs production |
|---:|---|---:|
| 4 | ON | 1 |
| 8 | ON | 1 |
| 12 | ON | 1 |
| 4 | OFF | **0** |
| 8 | OFF | **0** |
| 12 | OFF | **0** |

**With the multiway tier off, widening is a bit-exact no-op at every k. With it on, it is not.**

The mechanism: `wider_candidate_pool` was measured *before* the multiway centre tie-break was
integrated (2026-08-15, gate exception 2). `ranking.apply_center_tiebreak`'s second tier fires when
at least `MULTIWAY_MIN_GROUP_SIZE` (3) candidates sit within `MULTIWAY_TIE_SCORE_EPSILON` (0.005)
of the top. A wider pool retains more near-top candidates and can therefore reach that group size
where the narrow pool could not.

**The original arg-max argument is still correct.** What changed is that the arg-max is no longer
the last word — a later integration created a path through which pool width matters. Practical
impact on this surface is 1 pair and **accuracy-neutral** (0.700 either way), so this is a
documentation-correctness issue rather than an accuracy one. It is worth recording because
`PROJECT_STATUS.md` presents the no-op as structural and permanent, and future work that widens the
pool for any other reason would otherwise expect bit-identical behaviour and not get it.

Recommended edit to `reports/PROJECT_STATUS.md`, if the finding is accepted: note that
`wider_candidate_pool`'s no-op held under the ranker as it existed then, and that the multiway tier
integrated later makes pool width observable.

## 3. Why the re-scorer fails, mechanistically

At k=12 the pool averages 230 candidates. Recall reaches 0.900, so the truth is present for 36 of
40 pairs. The re-scorer still selects it exactly as often as arg-max does, and worse as tie_eps
widens the group it may choose from.

Combined with `discriminability_weighted/REPORT.md` §5 — the weighted margin favours the *decoy* on
4 of 5 reachable failures — the picture is consistent: weighted ZNCC's ordering of near-tied
candidates is close to uninformative, so enlarging its choice set is a pure liability. Adding the
truth to a set the scorer ranks near-randomly does not make it findable.

## 4. Honest status and limits

- **REJECT.** No configuration is a candidate for integration. Production untouched.
- Measured on **one 40-pair surface**. `development` was not swept here because the k-sweep needs a
  degraded family to be meaningful, and dev has none. A 0-rescue result across 144 configurations
  is strong evidence in the negative direction and does not need a second surface to be believed —
  a *positive* result would have.
- The negative result is specific to **this** re-scorer. It does not show that no re-scorer can use
  the extra recall; it shows that discriminability-weighted ZNCC cannot. Given P3's own diagnostic,
  extending this to "re-scoring is dead" would overclaim.
- What it does establish firmly: **recall is not the binding constraint at k ≥ 4.** Any future
  candidate-generation work should be judged on whether a *selector* can exploit it, not on recall
  alone — recall improved 15pp here and bought exactly nothing.

## 5. Run-count disclosure

- Frozen 156-pair benchmark: **0 runs.**
- `tune_degraded` (40 pairs, seed 314159): 1 baseline, 144 swept configs, 8 attribution runs.
- `validate_fresh` (40 pairs, seed 271828): **0 runs** — nothing survived to warrant validation.

144 configurations on 40 pairs would be a severe multiple-comparisons risk if anything had been
selected. Nothing was: the best configuration ties baseline and none rescued a pair, so there is no
selected result to correct for.

## Reproduce

```
python -m experiments.wide_pool_rescoring.run          --surface tune_degraded
python -m experiments.wide_pool_rescoring.noop_expiry  --surface tune_degraded
```
