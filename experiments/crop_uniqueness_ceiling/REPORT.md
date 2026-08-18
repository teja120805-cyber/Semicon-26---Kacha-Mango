# experiments/crop_uniqueness_ceiling — DIAGNOSTIC + a deliverable result

## Summary

Follow-up to `oracle_ceiling_diagnostic`, which found that 45% of failures never get the true
location proposed at all. This experiment set out to test whether those pairs are simply
**ill-posed** — reference crops of pure repeating pattern whose origin cannot be recovered from
image content. That hypothesis is **refuted**, and refuting it produced a sharper diagnosis than
confirming it would have.

Four results:

1. **Accuracy is governed by `uniqueness_score`, and periodicity is a confound, not a cause.**
   Crops with `uniqueness_score = 0` score 43.8% (n=48); everything else scores 88.0% (n=108).
   Holding uniqueness fixed, high vs. low periodicity changes accuracy by **0.4pp** — nothing.

   > **Recomputed 2026-08-18 on the current 77.6% production run:** 50.0% (24/48) versus
   > 89.8% (97/108). The conclusion is unchanged; only the magnitudes moved. Note also that
   > `uniqueness_score` is *computed from* the boundary flags
   > (`generator/metadata.py::uniqueness_score`), so this split and the boundary split are the
   > same 88 internal pairs under two names — not two independent measurements. Internal-only
   > figures (excluding the 20 `cross_generator` pairs, which carry no boundary metadata):
   > 50.0% (24/48) versus 92.0% (81/88).
2. **The task is nevertheless well-posed.** 154 of 156 reference crops have a *unique* origin at a
   0.95 content-identity threshold, and **zero** of the 40 failures picked a location whose image
   content matches ground truth above 0.95 (median 0.73). The pipeline is not picking a
   genuinely indistinguishable twin — it is picking a visibly different place.
3. **The bottleneck is template fidelity.** The two Search locations differ from each other at
   ZNCC **0.732**, yet the template scores them within **0.0098** of each other. The information
   that separates them survives in the Search image and is destroyed in template construction.
4. **A deliverable result:** gating on the pool-internal gap statistic, the pipeline answers
   **51% of pairs at 97.5% accuracy**, or **69% at 92.5%** — both above the 90% target, on a
   coverage-restricted basis, with no change to the matching algorithm at all.

Production is untouched. Stages 1–3 read ground truth and are diagnostics; stage 4's gate uses
only the candidate pool and is a genuine pipeline candidate.

## 1. Uniqueness governs accuracy; periodicity does not

| `uniqueness_score` | n | accuracy |
|---:|---:|---:|
| 0.00 | 48 | **0.438** |
| 0.60 | 14 | 0.857 |
| 0.95 | 53 | 0.868 |
| 1.00 | 21 | **1.000** |

27 of the 40 failures (67.5%) are `uniqueness_score = 0` pairs. Crossed against periodicity:

| | n | accuracy |
|---|---:|---:|
| unique crop, low periodicity | 74 | 0.878 |
| unique crop, **high** periodicity | 34 | 0.882 |
| non-unique crop, low periodicity | 19 | 0.421 |
| non-unique crop, **high** periodicity | 29 | 0.448 |

Periodicity moves accuracy by 0.4pp within the unique group and 2.7pp within the non-unique
group. Uniqueness moves it by ~44pp. **`reports/ACCURACY_FORENSICS.md` names periodicity/aliasing
as a dominant failure mechanism; on this benchmark it is a confound of crop uniqueness.** The
90%-accuracy campaign spent nine experiments targeting periodicity, several of them explicitly
(`periodicity/`, `prominence_rerank`, `pitch_aware_prominence` v1/v2). That is likely a large part
of why all nine were no-ops: they were treating the correlate rather than the cause.

A related, simpler predictor: crops that cross a mat or strip boundary score **0.898** (n=88);
crops that cross neither score **0.544** (n=68).

## 2. The task is well-posed — the pipeline picks a visibly different place

Measured Search-content against Search-content, with no template, hypothesis, or pipeline stage
involved, so these bound what *any* method could achieve:

- `identity(gt, pred)` — ZNCC between the Search patch at ground truth and the Search patch at
  the predicted location.
- `K@thr` — how many distinct Search locations match the ground-truth patch at or above `thr`.

| `identity(gt, pred)` on the 40 failures | |
|---|---:|
| median | **0.7322** |
| q25 | 0.6403 |
| max | 0.9350 |
| failures above 0.90 | 2 / 40 |
| **failures above 0.95** | **0 / 40** |

| threshold | crops with unique origin (K=1) | ambiguous (K>1) |
|---:|---:|---:|
| 0.85 | 86 / 156 | 69 (median K=6) |
| 0.90 | 128 / 156 | 27 (median K=7) |
| **0.95** | **154 / 156** | **1** |

Only two pairs (`dev_single_mat_007`, `val_same_preset_boundary_008`) are genuine near-duplicates
at 0.93. **The ill-posedness hypothesis is wrong**: the discriminating information is present in
the data for essentially every pair. The ceiling is not structural.

## 3. The actual bottleneck: the template cannot see what the Search image retains

Combining the two experiments' per-pair measurements over the 39 evaluable failures (medians):

| quantity | value |
|---|---:|
| template vs. Search@ground-truth (`oracle_gt`) | 0.8442 |
| template vs. Search@predicted (`oracle_win`) | 0.8514 |
| **Search@ground-truth vs. Search@predicted (content)** | **0.7322** |
| **template's separation between the two** | **0.0098** |

The two locations are 0.27 apart in content. The template — even under an ideal fitted warp —
separates them by 0.01. It is a ~0.845-fidelity rendering of the true location, and *any* decoy
that also renders at ~0.85 is indistinguishable to it.

Critically, template fidelity at ground truth is **the same on failures (0.8442) and on controls
(0.8490)**. Failing pairs do not have worse templates. The template is uniformly ~0.85-faithful,
and a pair fails whenever some decoy happens to land in the same 0.85 band. That is why every
failure is a near-tie, and why score-based re-ranking on top of that template could never work —
the information it would need was already discarded upstream.

This also reframes the 45% "candidate generation" bucket from the previous experiment. Those
pairs are not a separate discovery problem: the true location fails to reach any hypothesis's
top-2 peaks precisely *because* the template scores it no better than a dozen decoys.

The likely sources of the 0.845 ceiling are mechanical and testable: the Reference and Search
acquisitions differ in effective PSF (Reference is blurred at 1000px scale then resampled 10x;
Search is blurred at 10000px scale then area-averaged 10x), in dose/noise, and the Search path
adds shear/jitter the template never models. Raising template fidelity toward the ~0.95 the
content supports would open a large margin over decoys that sit at 0.73. **That is the first
direction this project has that attacks the cause rather than the symptom** — and it is
untouched by all nine campaign experiments, which operated downstream of template construction.

## 4. Deliverable: abstention beats the 90% target today, with no algorithm change

Using the pool-internal gap statistic from `oracle_ceiling_diagnostic` (top-1 score minus the
best score at a location >10px away) as a confidence gate — no ground truth, no new matching:

| gap threshold | pairs answered | coverage | **accuracy on answered** | wrong answers kept |
|---:|---:|---:|---:|---:|
| 0.000 (current) | 156 | 100.0% | 0.7436 | 40 |
| 0.005 | 107 | 68.6% | **0.9252** | 8 |
| **0.010** | **80** | **51.3%** | **0.9750** | **2** |
| 0.020 | 59 | 37.8% | 0.9831 | 1 |
| 0.050 | 35 | 22.4% | **1.0000** | 0 |

For a metrology tool, declining to answer is materially more useful than a confidently wrong
coordinate — and the current pipeline already emits an `ambiguous` flag, so the output contract
supports it. This does not raise pooled accuracy on the frozen benchmark and therefore does not
pass the integration gate as an accuracy change; it is a **confidence-calibration** change and
should be evaluated as one. It is offered as a candidate, not as a claimed gate pass.

## 5. What this closes and what it opens

- **Closed:** periodicity as a primary target; the ill-posedness hypothesis; any re-ranking that
  consumes the existing template's scores.
- **Open, and now the priority:** template-fidelity improvement (PSF/passband matching between
  the Reference and Search acquisition paths before correlation). The measured headroom is
  0.845 → ~0.95 against decoys at 0.73.
- **Open, cheap:** replacing `AMBIGUITY_THRESHOLD`'s basis with the gap statistic; the shipped
  flag fires on 128/156 pairs at 31% precision to catch the same failures the gap statistic
  catches at 50–65%.

## Reproduce

```
cd experiments/crop_uniqueness_ceiling
python run_experiment.py    # ~20s; requires experiments/oracle_ceiling_diagnostic outputs for stage 4
```

Outputs: `outputs/content_multiplicity.csv`, `outputs/uniqueness_ceiling_summary.json`.

Caveats stated plainly: stages 1–3 read ground truth and can never be a localization method.
Stage 4's thresholds are read off the full 156-pair benchmark rather than tuned on `development`
alone, so the specific coverage/accuracy pairs are descriptive, not a dev-tuned operating point —
choosing a threshold for production would need the usual dev-only selection. The two genuinely
ambiguous pairs are reported rather than excluded.
