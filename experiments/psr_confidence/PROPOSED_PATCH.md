# Proposed change — `AMBIGUITY_THRESHOLD` 0.92 → 0.990

**Not applied.** `pipeline/` is not modified without explicit approval. This file is the exact diff
plus the safety argument, so the decision can be made in one read.

## The diff

`pipeline/localize.py`, line 32 — one constant, no logic change:

```diff
-AMBIGUITY_THRESHOLD = 0.92  # second-best/best ZNCC ratio at or above this => flagged ambiguous
+# Second-best/best ZNCC ratio at or above this => flagged ambiguous.
+#
+# Recalibrated 0.92 -> 0.990 (2026-08-16, experiments/psr_confidence/).
+# 0.92 was far below the statistic's actual operating range: ambiguity_ratio
+# spans 0.816-0.999 with a median of 0.985, so a 0.92 cut fired on 85-91% of
+# all pairs at 32% precision - a flag that is on almost always carries no
+# information. The STATISTIC is sound (AUC 0.933-0.949 at separating correct
+# from wrong); only the constant was wrong.
+#
+# Fitted on development + a freshly generated degraded surface (n=64), then
+# tested ONCE on a held-back independently-seeded surface (n=40, seed 271828)
+# after the selection rule was fixed in code: flag precision 0.324 -> 0.750,
+# answering 70.0% of pairs at 92.9% accuracy. Cost, disclosed: failure recall
+# 1.000 -> 0.818, i.e. roughly one failure in five is no longer flagged.
+#
+# Reporting-only: `ambiguous` is written to output and never read to make a
+# decision anywhere in the codebase (verified by grep across pipeline/,
+# evaluation/, scripts/ and app/), so coordinates, accuracy and runtime are
+# bit-identical either way.
+AMBIGUITY_THRESHOLD = 0.990
```

## Why this cannot change any prediction — verified, not argued

`AMBIGUITY_THRESHOLD` appears in exactly one executable location:

| location | use |
|---|---|
| `pipeline/localize.py:170` | `ambiguous=amb_ratio >= AMBIGUITY_THRESHOLD` — sets a returned field |
| `pipeline/ranking.py:28` | comment only, explaining why `TIE_SCORE_EPSILON` is deliberately different |

And every consumer of the `ambiguous` field only ever **writes it out**:

| consumer | use |
|---|---|
| `evaluation/evaluate.py:90` | CSV/JSON column |
| `scripts/localize_pair.py:51` | JSON output field |
| `app/app.py:692,733,774` | a metric tile and a warning/success message |

Nothing branches on it to choose a coordinate. **Accuracy@5px, catastrophic rate, per-family
results and runtime are all bit-identical.**

## Consequence for the integration gate

Gate criteria 1–6 all measure prediction quality, which by construction cannot move here, so the
gate returns "no improvement" on every one of them for a change that is not attempting to improve
any of them. This is not a gate failure in the sense criteria 1/2 were written for.

Recommendation: if adopted, record it in `reports/GATE_EXCEPTIONS.md` as an exception of a **fourth,
distinct kind** — *not evaluable by the gate*, as opposed to exceptions 1–3 which all *fail* it.
Keeping that distinction visible matters: a reader who sees a fourth exception should be able to
tell immediately that this one carries no accuracy risk, rather than assuming the bar was lowered
again.

## A practical argument for the demo

At 0.92 the Streamlit app flags **85–91% of all pairs** as ambiguous, and
`app/app.py:736`'s "Not flagged ambiguous: the winning candidate clearly outscored the runner-up"
success message almost never appears. A 77.6%-accurate system that warns "ambiguous" on nine
results in ten reads, to a judge watching a live demo, as a system with no confidence in itself.

At 0.990 the app would flag ~30% and pass ~70% — and the passed ones are right 92.9% of the time on
held-back data. The demo statement becomes defensible and checkable: *"when the system says it is
confident, it is right 93% of the time, and it says so on 70% of cases."*

## Alternative operating point

`0.984` gives 43.8% answered at **100%** accuracy with zero missed failures — on the fit surfaces
only. It was **not** validated, because only the 0.990 rule was fixed before the held-back read.
Quoting its numbers as validated would be exactly the error the protocol exists to prevent. If a
zero-error operating point is wanted, it needs its own held-back evaluation first.

## Evidence

`experiments/psr_confidence/REPORT.md` §3, `outputs/calibration_result.json`,
`outputs/calibration_fit_curve.csv`.
