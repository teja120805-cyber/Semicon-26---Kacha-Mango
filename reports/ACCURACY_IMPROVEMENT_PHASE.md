# Accuracy Improvement Phase

> **Current production accuracy: 77.6%@5px pooled (n=156), as of 2026-08-16.** Reached via
> PSF-matched dual-arm candidate generation (`reports/GATE_EXCEPTIONS.md` exception 3,
> `experiments/psf_gated_selection/REPORT.md`), which corrected a ~16x sharpness mismatch between
> the correlation template and the Search image. Any pooled-accuracy figure below that predates
> that change is historical.
> 
> A second, larger campaign followed this one and is summarised in
> `experiments/ACCURACY_90_CAMPAIGN.md` (nine further ideas, all rejected) plus the
> diagnostic chain that eventually succeeded: `experiments/oracle_ceiling_diagnostic/`,
> `experiments/crop_uniqueness_ceiling/`, `experiments/psf_gated_selection/`.

Consolidated findings from the candidate-generation-focused research campaign (Experiments A-D).
Detail lives in each `experiments/<name>/REPORT.md`; this ties them together. Supersedes nothing —
`reports/ACCURACY_FORENSICS.md` remains the authoritative forensics document; this report is about
what was tried in response to it.

> **Update (2026-08-15): `finer_hypothesis_grid` is now integrated into production.** This report's
> original conclusion below ("do not integrate") reflected the evidence available at the time
> (fails the general-purpose `validation` split's literal-improvement criterion). A dedicated
> follow-up validation campaign (`experiments/finer_grid_validation/`) tested the hypothesis on data
> deliberately constructed to contain the conditions it targets and found a decisive, clean,
> twice-independently-reproduced effect (net rescue +16/+6, 0-1 breaks). `pipeline/candidate_generation.py`'s
> default hypothesis grid was widened accordingly (5x5 -> 9x9, same span) — the sections below are
> kept as the historical record of the decision process, not the current state.

## 1. What is currently limiting accuracy?

Per `reports/ACCURACY_FORENSICS.md`, confirmed again directly on the frozen benchmark this round:
candidate-generation failure (the true location never becomes a competitive classical match) is the
single largest failure category — 31/156 pairs, vs. 11 ranking failures and 7 genuine-ambiguity
cases. It concentrates specifically where no structural boundary is present (41.2% failure rate
without one vs. 3.4% with one), and the worst (catastrophic) failures are disproportionately
high-periodicity, zero-rotation/scale cases. Rotation/scale hypothesis-grid misalignment is a real,
separate, well-evidenced secondary mechanism.

## 2. Did the finer hypothesis grid actually help?

**Yes, robustly.** Re-validated this round with full instrumentation on both the frozen benchmark
and a genuinely fresh, independently-seeded dataset (seed `424242`, zero signature overlap): net
rescue **+4 on both**, candidate-generation failures reduced by **exactly 6 on both**. Mean error
dropped 56.9px -> 43.1px on the frozen benchmark; max error unchanged (the single worst case is
unaffected either way); >50px catastrophic rate 17.4% -> 15.2%. It still formally fails the
integration gate's criterion 1 (validation must improve, not tie) on both datasets — confirmed to be
a real, reproducible property of validation's boundary-heavy family composition, not noise.
**PROMISING, not integrated.**

## 3. Can a learned candidate generator recover candidate-generation failures?

**No, not at this data scale.** A dense-grid embedding-based proposer, trained on an expanded,
leakage-checked 184-pair/19-family development set (~5.9x more training triplets than the previously
rejected reranker), achieves only 3.0% candidate recall@5px (vs. classical's 82.6%) — far too low to
help. The classical-union-learned hybrid produced **bit-for-bit identical predictions to classical
alone** on all 132 evaluated pairs: zero rescues, zero breaks. **REJECT.**

## 4. Does larger spatial context distinguish true sites from periodic decoys?

**No reliable signal found**, at context windows up to 3x the reference footprint, using a
pixel-only periodicity/autocorrelation measure (no metadata, no ground truth fed into scoring).
Mean separation between true and decoy sites stayed within +/-0.008 of zero at every window size
tested, on 45 of the worst known failures. This is evidence of an information ceiling *for this
specific classical measure* — not proof no learned representation could ever do better, since a CNN
might exploit per-mat jitter/collapse signatures a hand-crafted autocorrelation statistic can't see
(this is exactly what Experiment B attempted, and it also fell short — consistent with, though not
conclusive proof of, a genuine information-scarcity problem at the current training-data scale).

## 5. Which approach performs best?

`finer_hypothesis_grid` (Experiment A), by a clear margin — the only approach with a positive,
robust, twice-independently-confirmed net rescue and no other approach came close. See the table
below.

## 6. What happened on fresh/external data?

`finer_hypothesis_grid` replicated almost exactly on the fresh independent dataset (net +4 both
places, candidate-generation failures -6 both places, validation ties both places). `cross_generator`
(the existing external dataset) also tied for the finer grid (65.0% -> 65.0%), consistent with it not
containing much rotation/scale drift (the external reference generator has no such mechanism, per
`reports/V2_BASELINE_REPORT.md`). No other experiment showed a positive result to check for
generalization.

## 7. What happened to mean/max error?

Only `finer_hypothesis_grid` improved mean error (56.9px -> 43.1px on the frozen benchmark). Max
error is unchanged by every experiment tested (900.8px on the frozen benchmark in every case) — the
single worst catastrophic pair (`cross_generator_00006`, a `candidate_generation` failure with a
0.0025 score margin between winner and runner-up) was not fixed by any approach tried this round.
`periodicity` (both variants) and `learned_candidate_generator` (learned-only) made mean error worse.

## 8. What happened to catastrophic failures (>50px)?

`finer_hypothesis_grid`: 17.4% -> 15.2% (improved). `rotation_scale` coarse-to-fine: 17.4% -> 15.9%
(mildly improved, weaker). `periodicity` gradient: 17.4% -> 23.5% (worse). `periodicity` ensemble:
17.4% -> 20.5% (worse). `learned_candidate_generator` learned-only: 87.9% (far worse, expected since
it's not meant to run standalone). Hybrid and `wider_candidate_pool`: unchanged (17.4%, identical to
baseline in both cases).

## 9. What should be integrated?

**Nothing, this round.** No experiment passed every integration-gate criterion.

## 10. What should NOT be integrated?

All six experiments tested (this round's three plus the prior round's three) — see the table and
per-experiment reports for why. None modified production; `pipeline/candidate_generation.py`,
`refinement.py`, and `feature_extraction.py` remain byte-for-byte unchanged (confirmed by direct
diff), and `localize.py`/`matching.py`/`ranking.py` carry only pre-existing docstring path edits from
an earlier terminology cleanup pass, verified line-by-line.

## 11. Final recommended production architecture

**Unchanged**: the classical multi-scale/multi-rotation ZNCC pipeline as it exists today
(`pipeline/candidate_generation.py`'s default 25-hypothesis grid, `rank_classical`, parabolic
subpixel refinement). This is a defensible, fully-understood, honestly-evaluated baseline. The one
credible path to a future integration is revisiting `finer_hypothesis_grid` with a validation set
that isn't already ceiling-limited (see below) — not a new architecture.

## Final comparison table

| Approach | @5px | Mean | Max | >50px | Candidate Recall@5px | Rescue | Break | Net | Runtime | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Classical (production)** | 70.5% | 56.9px | 900.8px | 17.4% | 82.6% | - | - | - | 1.0x | *(baseline)* |
| `finer_hypothesis_grid` | 73.5% | 43.1px | 900.8px | 15.2% | 87.1% | 6 | 2 | **+4** | 3.31x | **PROMISING — needs more validation** |
| `rotation_scale` (coarse-to-fine) | 72.0% | 50.0px | 900.8px | 15.9% | n/a | 4 | 2 | +2 | ~2.3-3.3x | REJECT (weaker than finer grid, no cost advantage) |
| `wider_candidate_pool` | 70.5% | 56.9px | 900.8px | 17.4% | 82.6% (unchanged) | 0 | 0 | 0 | ~3.2x | REJECT (structural no-op) |
| `periodicity` (gradient) | 63.6% | 87.8px | 900.8px | 23.5% | n/a | 1 | 10 | -9 | ~1.9x | REJECT |
| `periodicity` (ensemble) | 67.4% | 65.5px | 900.8px | 20.5% | n/a | 0 | 4 | -4 | ~3.5-5.2x | REJECT (also over runtime budget) |
| `learned_candidate_generator` (learned-only) | 0.8% | 349.5px | 977.5px | 87.9% | 3.0% | 0 | 92 | -92 | n/a (dense-grid, not comparable) | REJECT |
| `learned_candidate_generator` (hybrid) | 70.5% | 56.9px | 900.8px | 17.4% | 83.3% | 0 | 0 | 0 | classical + dense-grid overhead | REJECT (zero measurable effect) |
| `embedding_reranker_v1` (prior round) | ~30-55%\* | - | - | - | - | - | - | - | ~1.5x | REJECT (all 3 seeds) |

\* per-split range from the prior evaluation; failed every criterion, all splits regressed.

## Is anything safe enough to integrate into production right now?

**No.** `finer_hypothesis_grid` is the closest — a genuinely reproducible improvement, twice
confirmed independently, that specifically fixes the diagnosed grid-misalignment mechanism without
regressing candidate-generation-failure counts or catastrophic-failure rates. But it fails the
letter of the integration gate (validation must improve, not tie) on both datasets tested, and per
this project's own stated rule, that means it does not get integrated. The honest path forward, if
this is revisited, is either (a) accepting this as a known, documented limitation of the current
`validation` split's composition and deciding — as a deliberate policy choice, not a unilateral
override — whether "does not regress, ties at a pre-existing ceiling" should count differently than
"must literally improve" for a split that doesn't contain the target failure mode, or (b) expanding
`validation` with more of the conditions this fix targets before re-testing. Production remains the
classical baseline, unchanged, with every experiment preserved under `experiments/` for future
reference.
