# Dataset Audit

Scope: is the current V2 dataset (`data/development`, `data/validation`, `data/held_out`,
`data/challenge`, `data/cross_generator`) sufficient to support `reports/ACCURACY_FORENSICS.md` and
genuine held-out/generalization claims, and is it sufficient for training a learned component?
Findings below are from directly loading and cross-referencing every `ground_truth.json` in the
repo, not from inspection alone. **Updated** after the Phase 4 seeding fix (see section 2) — this
replaces the pre-fix version of this report; numbers below reflect the corrected
`driftsensev2.1.0` dataset currently on disk.

## 1. Current counts

| Split | Pairs | Families | Per-family n |
|---|---:|---:|---:|
| development | 24 | 3 | 8 |
| validation | 40 | 4 | 10 |
| held_out | 40 | 4 | 10 |
| challenge | 32 | 4 | 8 |
| cross_generator | 20 | 6 architectures | 3-4 |

Total internal pairs: 136. Plus 20 cross-generator = 156 (matches the pooled benchmark n).

**Statistical fragility**: at n=8-10 per family, a single flipped pair swings a family's
accuracy@5px by 10-12.5 percentage points. Enough to establish which families are *bad*, not enough
to confidently rank *which factor dominates* — the direct motivation for the larger, single-factor
sweep samples (n=20-40/level) used in `reports/ACCURACY_FORENSICS.md`, which exist specifically
because the frozen benchmark alone is too thin for that purpose.

## 2. RNG seeding fix — resolved

Previously, every pair's RNG derived from `default_rng(seed * 1_000_003 + pair_index)` — a function
of `pair_index` alone, causing real cross-split canvas/GT reuse (e.g. `validation` and `held_out`
shared 10/40 exact signatures). **Fixed** in `generator/dataset_generator.py`: the RNG now derives
from `default_rng([seed, family_salt(split, family_name), pair_index])`, verified to eliminate the
leakage (0 duplicate canvas+crop+GT signatures across all 136 pairs post-fix, confirmed directly
against the regenerated on-disk metadata — see section 3 below for exact current numbers).

## 3. Factor coverage (post-fix, current `data/`)

- **Distinct scenes**: 115/136 (84.6%) distinct canvas signatures (`mat_ids`+`presets`) — up from
  74/136 (54.4%) pre-fix. The remaining 21 coincidental repeats are normal statistical recurrence
  given a combinatorial space of 16 mat-grid positions x 6 presets, not a seeding artifact (verified:
  0 duplicate *full* signatures including crop location and ground truth).
- **Rotation** (`rotation_deg`): 26/136 pairs (19.1%) nonzero, range -3.99&deg; to +3.84&deg;.
- **Scale** (`extra_scale`): 26/136 pairs (19.1%) != 1.0, range 0.9014-1.0980 (widened by gate
  exception 1, `reports/GATE_EXCEPTIONS.md`; this row previously read 0.931-1.069) (the same 26 pairs sample
  both together per family design). Both remain thin for a factor identified as important — this is
  exactly why the dedicated forensics sweeps (n=40/level for rotation and scale independently) exist,
  rather than trying to stretch the frozen benchmark's 26 pairs into confident controlled
  conclusions.
- **Boundary crossing**: `crosses_mat_boundary` 74/136 (54.4%), `crosses_strip_boundary` 88/136
  (64.7%), neither (deep single-mat) 48/136 (35.3%) — workable balance on all sides.
- **Preset/architecture touch counts**: `mat_compact` 31, `mat_legacy` 32, `mat_dense` 47,
  `mat_relaxed` 32, `mat_nominal` 38, `mat_narrow` 31 (a crop can touch >1 preset at a boundary).
  Reasonably balanced; `mat_dense` is highest because `dev_dense_periodic` deliberately forces it.
- **Degradation combination coverage**: of 136 pairs, 64 have zero non-default acquisition/
  distortion parameters active, 30 have exactly one, 34 have two, and 8 (`ch_worst_case`) have four
  simultaneously (rotation + scale + barrel + speckle). No pair combines more than four — reasonable,
  since `reports/ACCURACY_FORENSICS.md`'s interaction sweeps already cover 2-3-way combinations of
  the factors that matter (rotation x scale x boundary) on dedicated, larger samples.
- **Seeds**: a single nominal seed (`777001`) for all internal splits, `900001` (external) for
  `cross_generator`. This is not a diversity concern post-fix: the per-pair RNG stream is now a
  function of `(seed, split, family_name, pair_index)`, so a single nominal seed value still yields
  136 statistically-independent draws, not 136 correlated ones.
- **Architecture**: 6 DRAM mat presets internally, 6 DRAM architectures in `cross_generator`.
  DRAM-only by explicit decision — not a coverage gap.
- **`cross_generator` schema mismatch**: its `ground_truth.json` has no `rotation_deg`/
  `extra_scale`/`structural_family`/`crosses_*_boundary` fields, so it cannot participate in the same
  per-factor breakdowns as the internal splits — reported separately by design.

## 4. Sufficiency for the classical pipeline (evaluation, not training)

**Sufficient**, with the seeding fix in place: large, robust effects (boundary presence 86.4% vs.
45.6%@5px; periodicity by preset 25-55%@5px) are visible even at n=8-10/family, and the dedicated
forensics sweeps supply the sample sizes needed for the finer-grained rotation/scale/interaction
conclusions the frozen benchmark itself cannot support. No additional benchmark data is needed for
this purpose.

## 5. Sufficiency for training a learned component

**Not sufficient, and this is unchanged by the seeding fix** (the fix addresses scene *diversity*
per pair, not pair *count*). `model/dataset.py::TripletPatchDataset` builds training triplets from
`development` only (24 pairs, 3 of the 15 structural families), yielding at most 72 triplets (3
hard negatives/pair) — the direct, diagnosed cause of `embedding_reranker_v1`'s catastrophic
overfitting (training loss to ~0 within 10-15 epochs). This is a training-data-scale problem, not a
seeding problem, and the current `development` split cannot supply meaningfully more without
changing its family composition.

**If a learned-model experiment is pursued** (contingent — see `reports/ACCURACY_FORENSICS.md`'s
"what would be worth doing next"), the concrete, bounded recommendation is:

- Expand `development` from 3 families (`dev_strip_anchor`, `dev_single_mat`, `dev_dense_periodic`)
  to cover representative pairs from **all major structural conditions already validated
  elsewhere in the benchmark** — boundary-crossing, same-preset-boundary, multi-mat, rotation drift,
  scale drift, and at least one heavy-noise condition — roughly 10-12 pairs each across ~12-15
  family variants, i.e. **~150-180 development pairs** (vs. the current 24), yielding ~450-540 hard
  negative triplets (roughly 6-7x more than the 72 that caused the prior failure), while still being
  a modest, bounded addition rather than "huge amounts of unnecessary data."
- Each new development family must use a genuinely new seed/pair range (not reuse `validation`/
  `held_out`/`challenge`'s own pairs) to avoid the exact cross-split contamination section 2 fixed.
- This expansion is **not** justified by wanting a better benchmark number — it is justified
  independently by the diagnosed triplet-count/family-diversity cause of the one learned-model
  attempt made so far. Do not execute it unless/until a learned-model experiment is actually
  attempted; generating it speculatively would violate the "don't generate data just in case" rule.

## 6. Verdict

> **Update 2026-08-16 — the "more training data" hypothesis is falsified.** The remediation this
> section proposes (roughly 6-7x more triplets) was built and tested in
> `experiments/learned_reranker_v2/`: 537 triplets vs. the original 72, all 3 seeds. It still
> regressed catastrophically (matched-splits 0.7727 -> 0.386-0.409). Triplet count and family
> diversity were therefore **not** the cause of the learned model's failure, and more data of this
> kind should not be expected to fix it.

Dataset is sufficient for classical-pipeline evaluation post-fix. It remains insufficient for
learned-component training, for a training-data-scale reason (bounded, specified above), not a
seeding-correctness reason (already fixed). FinFET remains out of scope (DRAM-only, by explicit
decision) — not a gap.
