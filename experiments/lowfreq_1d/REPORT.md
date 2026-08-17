# experiments/lowfreq_1d — the information ZNCC throws away: NEAR-MISS, does not clear the bar

**Date:** 2026-08-17. Diagnostic only; **no production change.** No harness, no sweep, no end-to-end
run. Consolidated findings: `experiments/BATCH_2026-08-17_REPORT.md`, Addendum 2.

**Lead with the verdict: the pre-registered bar was >70% and the best signal reached 68%. It does
not clear it, and the bar is not being moved.**

## 1. The hypothesis, stated before running

Two results from the same session combine into a specific, previously-untested question.

- `experiments/axis_decomposition/` — the residual failures are **one-dimensional**: the error is
  100% along a lattice axis at the median, with a cross-axis residual of only **1.47px**. The
  pipeline is confused about *which lattice period* it sits on, and nothing else.
- `experiments/template_fidelity_ablation/` — the content that discriminates one location from
  another lives substantially at **low spatial frequency**; high-pass filtering destroys it, with a
  monotone dose-response.

The observation that ties them together:

> **ZNCC is zero-mean and variance-normalised by construction** (`cv2.TM_CCOEFF_NORMED`): it
> subtracts the window mean and divides by the window standard deviation. The absolute brightness
> level and the absolute contrast of a candidate window are therefore **discarded before scoring
> ever happens.** On a Search image carrying vignette, gamma or illumination falloff those
> quantities vary slowly and smoothly across the field — exactly the kind of signal that could
> distinguish "period n" from "period n+1" along an axis.

Every one of the sixteen prior attempts modified how mid/high-frequency structure is *scored*. None
used the low-frequency content that normalisation *removes*. That is the gap, and it is why this is
not another ZNCC reweighting.

Three signals, evaluated at the true and the chosen location under the winner's own hypothesis:

| signal | definition |
|---|---|
| `mean_agree` | −&#124;mean(T) − mean(W)&#124; — absolute brightness match |
| `std_agree` | −&#124;log(std(T)/std(W))&#124; — absolute contrast match |
| `envelope` | correlation of heavily-blurred T, W — slowly-varying illumination shape |

**Honest caveat, recorded before running.** The Reference and Search travel different degradation
paths (`image_reference` vs `image_search`), so absolute brightness may simply not transfer between
them. If so, these signals are uninformative for a reason that has nothing to do with the lattice —
and that would be the finding.

**Bar (pre-registered, same as DDIS and OTSDF):** >70% preference for truth on the 22 reachable
failures to justify building anything. Chance is 50%.

## 2. Result — 68%, a near-miss

| signal | median margin | prefers truth |
|---|---:|---:|
| **mean brightness agreement** | +0.2983 | **15/22 (68%)** |
| contrast (std) agreement | +0.0035 | 13/22 (59%) |
| illumination envelope correlation | −0.0993 | 5/22 (23%) |
| majority of the three | — | 13/22 (59%) |

**Against the pre-registered bar: does not clear it.** 15/22 is 68%, binomial p = 0.067 — not
significant on 22 pairs. The bar was fixed before running and is not being moved after the fact;
doing so is exactly the error pre-registration exists to prevent, and this project has a recorded
instance of that error (`pitch_aware_prominence`, #8/#9 in `ACCURACY_90_CAMPAIGN.md`).

## 3. Why it is still worth recording

In context, this is the **strongest single signal found across roughly nineteen screened measures**
this session, per the batch report's tally. Every other candidate sat at or below chance:

| measure | prefers truth | source |
|---|---:|---|
| **mean brightness agreement** | **15/22 (68%)** | this experiment |
| DDIS (P5) | 11/22 (50%) | `experiments/ddis_diversity/` |
| OTSDF, best α | 11/22 (50%) | `experiments/alternative_scores/` |
| illumination envelope | 5/22 (23%) | this experiment |
| phase-based (log-Gabor) | 4/22 (18%) | `experiments/alternative_scores/` |

Mean brightness agreement is the only one meaningfully above chance, and it is interesting precisely
because it is information the production scorer is **structurally blind to** — not information it
weighs badly, information it deletes before it ever scores.

## 4. The warning sign that must be explained first

The `envelope` signal scored **5/22 (23%)** — worse than chance, and worse than any measure screened
this session except phase-based matching. That is the pre-registered caveat materialising: the two
images' degradation paths differ enough that a low-frequency *shape* comparison is actively
misleading.

That the coarsest summary (a single mean) is the most informative while the richest low-frequency
comparison (the envelope) is anti-informative is not a comfortable pattern. It is consistent with
brightness transferring only as a scalar offset, and it needs explaining before anything is built on
the mean-agreement result.

## 5. Status and what a proper follow-up requires

- **Bar not cleared: 68% against >70%, p = 0.067.** Near-miss, reported as a near-miss.
- **Nothing to integrate**, and no harness earned.
- Recommended as a genuine follow-up, but **not on this evidence alone**. It needs:
  1. **more than 22 pairs** — the bar is a coin-flip away at this sample size in both directions;
  2. **a check that Reference→Search brightness transfer is not a family-specific artifact** of the
     vignette/gamma families, rather than a general property;
  3. **an explanation for the envelope signal scoring 23%**, per §4.
- If it is ever built, `ACCURACY_90_CAMPAIGN.md` #9's lesson applies: penalty-only formulation,
  screened against *correct* pairs as well as failures, and validated on at least two
  production-family seeds before any frozen run.

**The 1-D framing from `experiments/axis_decomposition/` is correct. The most obvious signal for
exploiting it does not clear its bar.** Both halves of that sentence are the result.

## Reproduce

```
python -m experiments.lowfreq_1d.diagnose
```
