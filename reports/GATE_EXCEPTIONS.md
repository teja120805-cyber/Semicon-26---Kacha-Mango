# Documented gate exceptions

`reports/V2_ARCHITECTURE_PLAN.md` section 8 requires all 7 integration-gate criteria to pass
before a candidate change is wired into `pipeline/`/`generator/`. Three changes have been integrated
despite failing that literal bar: two on 2026-08-15, one on 2026-08-16. This file exists so that fact is never
buried — every future reader of the gate should be able to find exactly which changes are
exceptions, why, and what evidence backed the decision, rather than assuming "in production"
means "passed all 7 criteria."

## Why an exception, not a rewrite of the gate itself

The gate's criteria 1/2 ("must improve validation" / "must improve held_out", pooled, strictly)
were designed to catch a specific risk: a general ranking-algorithm or model change that looks
good on one number but is actually net-neutral or harmful once you check broadly. That is
exactly what they caught earlier in this project (`experiments/embedding_reranker_v1/`,
`experiments/center_tiebreak_v2/` — both correctly rejected, the latter for a documented 497px
new catastrophic failure).

Exceptions 1 and 2 below are a different shape: a **narrowly-scoped, spec-compliance-driven fix**
whose effect surface is, by construction, a handful of specific families or a rare tie
condition — not a general algorithm change expected to move pooled metrics broadly. Applying
criteria 1/2 literally to this shape of change fails them almost by definition (the affected
families don't dominate `validation`/`held_out`'s pooled count), regardless of whether the
change is safe. Rewriting the gate to special-case this would risk quietly weakening it for the
next, more consequential change; leaving it as-is and logging the exception here keeps the gate
meaningful while still shipping compliance work the Applied Materials materials state as hard
requirements.

## Exception 1 — A2: scale hypothesis grid + dataset scale range widened to literal 9:1–11:1

- **What changed**: `pipeline/candidate_generation.py::DEFAULT_SCALE_HYPOTHESES` (9 → 11 points,
  9.0–11.0); `generator/dataset_generator.py`'s `ho_scale_drift`, `ch_combined_acquisition`,
  `ch_worst_case` families' `_scale_range` ((0.93, 1.07) → (0.90, 1.10)).
- **Why it's safe despite failing criteria 1/2**: `validation` contains none of the 3 affected
  families — it structurally cannot show improvement from this change, so criterion 1 fails by
  construction, not because of any harm. `held_out`'s one affected family (`ho_scale_drift`)
  improved on catastrophic rate (0.225 → 0.200) but didn't flip enough pairs across the 5px line
  to satisfy criterion 2's strict inequality on this particular draw.
- **Evidence**: zero regressions across 2 independent datasets, every split, every family
  (frozen: exact tie everywhere; fresh, seed 913442: pooled 70.5% → 71.4%@5px,
  `ch_worst_case` 50.0% → 62.5%). Runtime 1.22–1.28x (well under the 5x ceiling in criterion 6).
  Full derivation: `experiments/scale_range_v1/REPORT.md`.
- **Criteria that did pass**: 3 (ties on `cross_generator`, which the change doesn't touch),
  4 (no catastrophic increase), 5 (no per-family regression), 6 (runtime), 7 (same conclusion
  reproduced independently on the fresh dataset).

## Exception 2 — A6: multiway-gated centre tie-break

- **What changed**: `pipeline/ranking.py::apply_center_tiebreak` gained a second tier
  (`MULTIWAY_TIE_SCORE_EPSILON=0.005`, `MULTIWAY_MIN_GROUP_SIZE=3`,
  `MULTIWAY_MAX_SPREAD_PX=200.0`) alongside the original, unweakened tight tier
  (`TIE_SCORE_EPSILON=1e-6`).
- **Why it's safe despite failing criteria 1/2**: the one confirmed rescue
  (`ch_worst_case_006`) is in `challenge`, not `validation` or `held_out` — same structural
  reason as A2. Criterion 3 also reads False on the fresh-dataset run only because that dataset
  has no `cross_generator` split at all (external/fixed data, no fresh analogue — the gate has
  no "not applicable" state for a missing split); not a real finding.
- **Evidence**: zero regressions across 2 independent datasets (frozen n=132, fresh n=112,
  seed 502187), across all 13 families present. One confirmed catastrophic rescue
  (`ch_worst_case_006`, 118.5px → 4.6px). The fresh dataset shows the mechanism firing safely
  with no harm but no analogous case to rescue there — honestly, this is a narrower result than
  A2's: real and safe, not proven to generalize broadly. Full derivation, including the 72-config
  sweep that ruled out `min_group_size=2` at every epsilon tried:
  `experiments/multiway_tiebreak_v1/REPORT.md`.
- **Criteria that did pass**: 4 (no catastrophic increase), 5 (no per-family regression),
  6 (runtime — 1.01x, the change never touches candidate generation), 7 (same safe-but-narrow
  conclusion reproduced independently on the fresh dataset).

## Exception 3 — PSF-matched dual-arm candidate generation (2026-08-16)

- **What changed**: `pipeline/matching.py::build_template` gained an optional `psf_sigma`
  (default 0.0, bit-identical to before); `pipeline/candidate_generation.py::build_candidate_pool`
  and `pipeline/refinement.py::refine` forward it; `pipeline/localize.py` now builds the candidate
  pool under **both** `psf_sigma=0.0` and `psf_sigma=PSF_MATCH_SIGMA` (1.6) and keeps whichever
  arm is more decisive, measured by the gap between the top candidate and the best candidate at a
  location >10px away. `LocalizationResult` gained `psf_sigma` / `psf_decisiveness` so the
  per-pair choice is visible rather than hidden.
- **Why the change exists**: the Reference and Search images travel different optical/resampling
  paths. The Reference is blurred at Reference resolution and then shrunk 10x, so the template
  carries ~0.06 Search-pixels of blur; the Search is blurred at fine-canvas resolution and then
  area-averaged, carrying ~1.0. The template is therefore ~16x sharper than the image it is
  correlated against, and measured only ~0.85 ZNCC fidelity even at the TRUE location — leaving no
  separation from decoys sitting at the same level, despite the underlying Search content at those
  two places differing at ZNCC 0.73 (`experiments/crop_uniqueness_ceiling/REPORT.md` §3).
- **Why dual-arm rather than always-on**: applying the blur unconditionally is a real win on
  clean-optics and periodic-ambiguity families and a real **loss** on acquisitions carrying
  non-stationary, non-Gaussian corruption (`ch_barrel_charging`, `ch_speckle_saltpepper`,
  `ho_vignette_gamma`, `ch_worst_case`). Detecting those acquisitions was tried and **provably
  cannot work**: families that gain span estimated blur 0.36–1.06 and families that lose span
  0.34–1.03, almost completely overlapping (`experiments/psf_matched_adaptive/REPORT.md`).
  Selecting by decisiveness sidesteps detection entirely — the three worst-affected families
  revert to baseline **byte-for-byte**, because the unblurred arm simply wins on them.
- **Why it's safe despite failing criteria 4/5**: both failures are 1–2 pair effects on small
  samples. Criterion 4 fails on `held_out` alone, whose catastrophic (>50px) rate goes 0.200 →
  0.225 — **8 → 9 pairs of 40** — while the *total* catastrophic count across the benchmark
  **falls, 26 → 22**. Criterion 5 fails on one family, `ch_worst_case`, 0.750 → 0.500 — **2 pairs
  of 8**. Across both seeds that family is 1 rescued / 2 broken (net −1 of 16 pairs), and its own
  baseline accuracy swings 0.750 → 0.125 between seeds, so conclusions about it from 16 pairs are
  weak in either direction. It is the only net-negative family of 16.
- **Evidence**: replicated on two independently-seeded datasets — frozen benchmark
  0.7436 → **0.7756** (n=156) and seed 618234 0.6618 → **0.6985** (n=136). Across both,
  **14 rescued / 4 broken, sign test p = 0.031** — the first statistically significant result in
  this project. That is a stronger evidential base than exceptions 1 or 2, which rested on "safe,
  no regressions" rather than a significant measured gain. Full derivation, including the four
  rejected predecessors and a negative-control selection rule:
  `experiments/psf_gated_selection/REPORT.md`.
- **Criteria that did pass**: 1 (validation 0.900 → 0.925), 2 (held_out 0.650 → 0.700),
  3 (cross_generator tied at 0.800), 6 (runtime 2.0x against a 5.0x budget — the change doubles
  candidate generation by construction), 7 (reproduced on an independent seed).
  **This is the first change in the project to pass criteria 1, 2 and 3 together.**
- **Known cost**: candidate generation runs twice per pair. Callers needing the old single-arm
  cost can pass `psf_selection=False` to `localize()`, which is bit-identical to the
  pre-2026-08-16 pipeline (verified per-pair, not assumed).

## Exception 4 — `AMBIGUITY_THRESHOLD` recalibrated 0.92 → 0.990 (2026-08-17)

**This exception is a different KIND from 1–3, and the distinction matters.** Exceptions 1–3 all
*fail* gate criteria that genuinely applied to them. This change is **not evaluable by the gate at
all**: criteria 1–6 every one measure prediction quality, and this change provably cannot alter a
prediction. A reader who finds a fourth exception here should be able to tell immediately that it
carries no accuracy risk, rather than assuming the bar was lowered again.

- **What changed**: one constant in `pipeline/localize.py`. No logic, no control flow.
- **Why**: `reports/PROJECT_STATUS.md` already recorded the flag as miscalibrated ("fires on
  128/156 pairs at 31% precision"). That is reproduced independently here — 85–91% flagged at
  ~32% precision across two surfaces. The recorded *diagnosis* was incomplete, though: the
  **statistic** is sound (`ambiguity_ratio` separates correct from wrong at AUC 0.933–0.949,
  comparable to the pool gap's 0.941–0.964), and only the **constant** was wrong. It spans
  0.816–0.999 with median 0.985, so a 0.92 cut sits far below the distribution.
- **Why not PSR instead**: `reports/RESEARCH_SURVEY_SCORING.md` §P4 proposed PSR as the principled
  replacement. It was built and tested, and is **worse** — AUC 0.577–0.765, near chance on the
  degraded surface, and as a dual-arm selector it costs accuracy (0.708 → 0.583). See
  `experiments/psr_confidence/REPORT.md` §§1–2. The survey's prediction is refuted.
- **Evidence**: fitted on `development` + a freshly generated degraded surface (n=64), then
  evaluated **once** on a held-back independently-seeded surface (n=40, seed 271828) after the
  selection rule was fixed in code. Flag precision **0.324 → 0.750**; the pipeline answers 70.0%
  of pairs at 92.9% accuracy. This independently reproduces the "69% at 92.5%" operating point in
  `experiments/crop_uniqueness_ceiling/` §4 — via a single constant on a statistic already in
  production, rather than a separate gating mechanism.
- **Cost, stated plainly**: failure recall falls 1.000 → 0.818 on the held-back surface, so roughly
  one failure in five is no longer flagged. On `development` it stays at 1.000. At 0.92 the flag
  caught every failure but fired on nine pairs in ten, which is not a usable trade.
- **Safety verification, run rather than argued**: both thresholds were executed in the same
  process and compared in memory — first on all 24 `development` pairs, then widened to **100
  pairs** (`development` 24, `validation` 40, `held_out` 36). `x`, `y`, `confidence`,
  `ambiguity_ratio` and `error_px` are **bit-identical on every pair**; pooled accuracy@5px
  (0.7900) and catastrophic rate (0.1200) unchanged; per split 0.7083 / 0.9250 / 0.6944, all
  identical before and after. Reproduce with
  `python -m experiments.psr_confidence.verify_integration` and `...verify_wide`. Grep also
  confirms `ambiguous` is never read to make a decision anywhere in `pipeline/`, `evaluation/`,
  `scripts/` or `app/`. (`challenge` and `cross_generator` were not available locally, so this is
  100 of the frozen 156 — but since every prediction is bit-identical by construction and on 100
  measured pairs, a full frozen run was not spent to confirm a provable no-op.)
- **Precision depends on the failure base rate — the 0.750 figure is not universal.** Measured
  across three populations, the *relative* gain is stable at roughly 2x while the absolute number
  tracks how many failures exist to catch:

  | population | n | accuracy | flag precision before → after | failure recall after |
  |---|---:|---:|---:|---:|
  | held-back degraded (seed 271828) | 40 | 0.725 | 0.324 → **0.750** | 0.818 |
  | `development` | 24 | 0.708 | 0.333 → **0.778** | 1.000 |
  | `development`+`validation`+`held_out` | 100 | 0.790 | 0.259 → **0.514** | 0.857 |

  The 100-pair set is 79% accurate, so only 21 failures exist among 100 pairs and precision is
  bounded accordingly. **The defensible claim is the ~2x precision improvement at roughly a third
  the flag rate, not a fixed 0.75.** Quote the matching population when citing a number.
- **Frozen-benchmark figures (2026-08-17, full `scripts/evaluate_model.py` re-run, n=156).** These
  are the numbers the README quotes, and the only ones that should be cited against the benchmark:

  | threshold | flagged | flag rate | precision | failure recall | answered | acc on answered |
  |---|---:|---:|---:|---:|---:|---:|
  | 0.92 (previous) | **128** | 0.821 | 0.273 | 1.000 | 0.179 | 1.0000 |
  | **0.990 (shipped)** | 55 | 0.353 | **0.545** | 0.857 | **0.647** | **0.9505** |

  The 128/156 figure is an exact reproduction of the "fires on 128/156 pairs" already recorded in
  `reports/PROJECT_STATUS.md`, which is a useful independent confirmation that the constant, not
  the statistic, was the defect.

  **Pooled accuracy@5px came back 0.7756 and catastrophic rate 0.1410 — both unchanged**, which was
  the pass condition for that run. It was treated as a tripwire, not a measurement: any deviation
  would have falsified the no-op claim rather than confirmed it.

  This operating point also sits neatly between the two reported in
  `experiments/crop_uniqueness_ceiling/` §4 ("51% of pairs at 97.5%, or 69% at 92.5%") — 64.7% at
  95.1% — reached with a single constant on a statistic already in production rather than a
  separate gating mechanism. Two independent routes landing on the same curve is meaningful
  evidence the curve is real.

  Note the cross-platform caveat already documented in README's Reproducibility section: predictions
  computed on Linux differ from Windows by up to ~2e-3 px in these runs (OpenCV build variation in
  `warpAffine`/`GaussianBlur`/`matchTemplate`). That is immaterial against a 5px tolerance and
  identical accuracy resulted, but it means a cross-machine comparison cannot serve as the
  bit-identity check — the same-process A/B in `verify_integration.py` / `verify_wide.py` is what
  establishes that.
- **Not adopted**: threshold `0.984` gives 43.8% answered at 100% accuracy with zero missed
  failures — but on the fit surfaces only. It was not validated, because only the 0.990 rule was
  fixed before the held-back read. If a zero-error operating point is wanted it needs its own
  held-back evaluation first.
- **Practical effect on the demo**: at 0.92 the Streamlit app flagged 85–91% of results as
  ambiguous and `app/app.py`'s "not flagged ambiguous" success path almost never fired — a
  77.6%-accurate system presenting as having no confidence in itself. At 0.990 it flags ~30%.

## What would revoke any of these exceptions

New evidence of a regression on either change — a new dataset, a new seed, or real submission
data showing harm — should be treated as grounds to re-open this decision, exactly as if it were
a normal gate failure. No change gets a permanent pass; the exception is for the specific
evidence cited above, not a blanket allowance for future modification of the same code.

For exception 4 specifically, the revocation condition is different in kind, because accuracy
cannot regress from it: it should be re-opened if the flag's *precision* is measured below the
0.75 held-back figure on new data, or if any future code change starts **reading** `ambiguous` to
make a decision — at which point this stops being a reporting constant and must be re-evaluated
under the full gate.

## Decision log

- **2026-08-16: exceptions 1 (A2) and 2 (A6) explicitly confirmed by the user** to keep
  integrated as-is, after an independent verification pass (24/24 tests; full dataset
  regeneration from seed 777001; full 156-pair benchmark re-run, the first time A2 and A6 were
  evaluated running together rather than each in isolation). Pooled result at that point: 74.4%@5px,
  16.7% catastrophic rate.
- **2026-08-16: exception 3 (PSF-matched dual-arm candidate generation) explicitly confirmed by
  the user** to keep integrated as-is, on the evidence in this file and
  `experiments/psf_gated_selection/REPORT.md` — statistically significant across two independently-
  seeded datasets (p = 0.031), first change in the project to pass gate criteria 1/2/3 together,
  with the two failing criteria's margins (1 pair in `held_out`, 2 pairs in `ch_worst_case`)
  disclosed explicitly at the time of confirmation, not discovered after. As with exceptions 1/2,
  this was already live in the working tree pending the decision rather than staged behind it —
  the same pattern, noted again rather than treated as settled precedent.

- **2026-08-17: exception 4 (`AMBIGUITY_THRESHOLD` 0.92 → 0.990) integrated at the user's
  direction**, following the six-experiment campaign consolidated in
  `experiments/REACHABILITY_CAMPAIGN.md`. It was the only one of six investigations to produce a
  positive, validated result; the other five are rejections and nothing from them was integrated.
  In particular `experiments/anisotropic_psf/` was **deliberately not** integrated despite showing
  2 rescued / 0 broken and zero measured harm across 64 pairs — it had exactly zero effect on the
  independent 40-pair surface (sign test p = 0.25, against the p = 0.031 that exception 3 cleared),
  and integrating on a development-only gain is the benchmark-mining this project's conventions
  exist to prevent. Recorded here so the decision not to ship it is as visible as the decision to
  ship this one.

The revocation condition above applies to all four exceptions going forward from their
respective confirmations.
