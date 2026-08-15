# Accuracy Forensics (Phase 3)

**Question this report answers**: *why* is the classical pipeline wrong when it's wrong — not just
which structural family scores worst. `reports/V2_BASELINE_REPORT.md`'s family-level breakdown
correlates several hard conditions (periodicity, rotation, scale, boundary) but confounds them,
because every family combines several axes at once. This report isolates one variable at a time.

**Method**: `experiments/accuracy_forensics/` generates controlled pairs varying exactly one
parameter across a set of levels (holding everything else at the generator's defaults —
`rotation_deg=0`, `extra_scale=1.0`, no boundary bias, no extra noise, `crop_mode="random"` unless
the factor under test is boundary/periodicity), runs the real classical pipeline
(`pipeline/candidate_generation.py`, `pipeline/ranking.py::rank_classical`,
`pipeline/refinement.py` — unmodified, called directly), and additionally checks the resulting
candidate pool against ground truth for diagnostic purposes only (never fed back into the
pipeline's own decisions — the same principle `evaluation/evaluate.py` already uses). Uses a
dedicated seed (`830001`), fully independent of the frozen benchmark under `data/`.

Every level of a given factor's sweep uses the **same underlying scene** at a given `pair_index`
(direct-value overrides consume no extra RNG draws — see `sweep_config.py`'s module docstring),
so a factor's effect is measured on the same 40 (or 30/25/20) scenes across every level — a paired,
repeated-measures design, not independent random draws per level.

## Failure taxonomy

For every pair, the candidate whose center is nearest ground truth is located in the deduplicated
candidate pool and classified:

| Category | Meaning |
|---|---|
| `success` | final error ≤ 5px |
| `candidate_generation` | no candidate within 5px of GT survived generation + deduplication — the classical search never proposed the right answer |
| `candidate_ranking` | GT's candidate exists but a different, higher-scoring candidate won |
| `genuine_ambiguity` | same as above, but the winning score beats GT's candidate by less than 0.02 ZNCC — a near-tie, not a confident wrong pick |
| `refinement` | the correct candidate won at the coarse stage, but subpixel refinement pushed the final prediction past 5px |
| `unexplained` | none of the above — flagged for manual review (none observed in this run) |

## Finding 1 — Periodicity is the strongest single factor tested, and it fails at candidate generation, not ranking

| Preset (word pitch) | n | Acc@5px | P90 error | >10px failure | Dominant failure mode |
|---|---:|---:|---:|---:|---|
| `mat_compact` (56nm) | 40 | **25.0%** | 652.5px | 75% | `candidate_generation` (24/40) |
| `mat_dense` (48nm) | 40 | 32.5% | 650.1px | 62% | `candidate_generation` (27/40) |
| `mat_narrow` (64nm) | 40 | 32.5% | 458.8px | 68% | `candidate_generation` (23/40) |
| `mat_nominal` (80nm) | 40 | 47.5% | 719.1px | 53% | `candidate_generation` (13/40) |
| `mat_relaxed` (100nm) | 40 | 50.0% | 697.5px | 50% | `candidate_generation` (11/40) |
| `mat_legacy` (136nm) | 40 | 55.0% | 583.0px | 45% | mixed (`candidate_generation` 8, `genuine_ambiguity` 8) |

All six levels use `crop_mode="single_mat"` (deep inside one mat, zero rotation/scale drift, no
boundary in view) — this isolates periodicity from every other axis. Accuracy tracks pitch
monotonically: the two densest presets (`mat_compact`, `mat_dense`) are the worst in the entire
forensics sweep, worse than any single rotation, scale, noise, or drift level tested.

**This is not a ranking problem — it's the classical search never finding the right answer.** For
every preset, the majority of failures are `candidate_generation`, not `candidate_ranking`. Checked
directly: **62-85% of `candidate_generation` failures land within a quarter-pitch of an exact
integer multiple of the mat's own word pitch** (85% for `mat_dense`, the densest preset; 62% for
`mat_legacy`, the loosest). This is the unambiguous signature of periodic aliasing: the ZNCC
correlation genuinely produces a strong peak at a wrong periodic repeat, and that peak — not a
near-miss of the true location — is what enters the candidate pool. The true location's peak either
doesn't survive the top-2-per-hypothesis extraction or gets deduplicated away in favor of a
higher-scoring wrong repeat.

## Finding 2 — Rotation/scale damage tracks hypothesis-grid alignment, not magnitude

`pipeline/candidate_generation.py` tests a fixed grid: scale ratios {9.2, 9.6, 10.0, 10.4, 10.8}
(i.e. `extra_scale` {0.92, 0.96, 1.00, 1.04, 1.08}) and rotations {-5°, -2.5°, 0°, 2.5°, 5°}.

| `extra_scale` | Distance to nearest tested hypothesis | Acc@5px | >10px failure |
|---:|---:|---:|---:|
| 1.00 | 0.00 (exact hit) | **70.0%** | 30% |
| 1.02 | 0.02 (midpoint) | **42.5%** | 57% |
| 1.04 | 0.00 (exact hit) | **70.0%** | 30% |
| 1.06 | 0.02 (midpoint) | **45.0%** | 55% |
| 1.08 | 0.00 (exact hit) | 65.0% | 35% |

| `rotation_deg` | Distance to nearest tested hypothesis | Acc@5px | >10px failure |
|---:|---:|---:|---:|
| 0° | 0.0° (exact hit) | 70.0% | 30% |
| 1° | 1.5° (near midpoint of 0°/2.5°) | **50.0%** | 50% |
| 2° | 0.5° (close to 2.5°) | **52.5%** | 45% |
| 3° | 0.5° (close to 2.5°) | 67.5% | 30% |
| 4° | 1.0-1.5° (midpoint of 2.5°/5°) | **40.0%** | 57% |
| 5° | 0.0° (exact hit) | 65.0% | 35% |

Both tables show the same **sawtooth pattern lined up exactly with distance to the nearest tested
hypothesis**, not with the magnitude of rotation/scale itself: `extra_scale=1.02` and `1.06`
(equidistant from two grid points) collapse to 42-45%, while `1.00`/`1.04`/`1.08` (exact grid hits)
recover to 65-70% — despite `1.06` being *closer to nominal* than `1.08`. The same misalignment
signature appears for rotation. This directly confirms, with controlled evidence rather than
correlation, the mechanism `reports/V2_BASELINE_REPORT.md` could previously only speculate about:
**the coarse hypothesis grid, not rotation/scale drift per se, is what fails** — a real value
between two tested hypotheses scores worse under its nearest neighbor than a competing wrong
location does under a well-matched hypothesis, so the ranking stage picks the wrong place *for the
right reason* (best available score). This is a fixable algorithmic property (finer grid, or a
continuous scale/rotation refinement stage after the coarse hypothesis search), not a fundamental
information limit.

## Finding 3 — Boundary presence is strongly protective, exactly as the correlational data suggested

| Crop condition | n | Acc@5px | P90 error |
|---|---:|---:|---:|
| `multi_mat_3plus` (touches 3+ mats) | 40 | **100.0%** | 0.3px |
| `strip_crossing` | 40 | 97.5% | 0.4px |
| `mat_boundary` (2+ mats) | 40 | 92.5% | 0.4px |
| `same_preset_boundary` (hardest boundary case) | 40 | 77.5% | 23.6px |
| `no_boundary_single_mat` (deep in one mat, any preset) | 40 | **57.5%** | 276.0px |

Confirms the correlational finding on controlled, single-variable evidence: any real macro structure
in view (a boundary or a strip) makes the problem nearly trivial for the classical matcher; the
absence of one is the single largest source of remaining difficulty short of a densest-preset
periodic interior.

## Finding 4 — Noise, raster drift, and row jitter are minor factors across their full tested range

| Factor | Range tested | Acc@5px range | Pattern |
|---|---|---|---|
| `search_dose` (noise) | 220 (baseline) down to 25 (severe) | 62.5-70.0% | Flat; no clear monotonic trend |
| `raster_drift_shear` | 0px to 4px | 60.0-70.0% | Flat; no clear monotonic trend |
| `row_jitter` | 0px to 2.5px | 53.3-70.0% | Mild downward trend only at the top of the range |

None of these three approaches the damage periodicity or grid-misaligned rotation/scale cause. This
matches the original correlational finding (`ho_heavy_noise` scored 70-80%@5px in both the
correlational and this controlled setting) and rules out noise as a primary target for improvement.

## Failure decomposition on the frozen benchmark itself (not just the synthetic sweeps)

Everything above uses `experiments/accuracy_forensics/`'s dedicated controlled-sweep data. This
section applies the same instrumented pipeline wrapper directly to the actual 156-pair frozen
benchmark (`scripts/decompose_baseline_failures.py`, output:
`outputs/reports/baseline_failure_decomposition.csv`), to confirm the sweep findings aren't an
artifact of the synthetic sweep design.

Pooled failure composition (n=156): **success 107 (68.6%), `candidate_generation` 31 (19.9%),
`candidate_ranking` 11 (7.1%), `genuine_ambiguity` 7 (4.5%)**. Candidate-generation failures alone
outnumber ranking + ambiguity combined — confirming, on the real benchmark, that the dominant
failure mode is the search never proposing the true location, not picking the wrong one from a pool
that contains it.

**Boundary presence correlates specifically with which failure type occurs, not just overall
accuracy**: with a boundary in view, only 3/88 pairs (3.4%) fail at candidate generation; without
one, 28/68 (41.2%) do. This sharpens Finding 1: boundary absence doesn't just make localization
harder in general, it specifically causes the classical search to never find the true location as a
competitive peak.

Rotation/scale buckets on the real benchmark are too thin to draw independent conclusions (only
26/156 pairs have any drift at all, split further into low/medium/high buckets leaves single digits
per cell) — this is exactly why `reports/DATASET_AUDIT.md` flagged this as insufficient for
controlled conclusions and why the dedicated sweeps (n=40/level) exist. One consistent signal
despite the thin data: the medium-rotation bucket (1.5-3.5°, spanning the grid-hypothesis midpoint)
shows the highest candidate-generation-failure rate (6/14, 42.9%) of any rotation bucket — consistent
with Finding 3, and a reminder that grid misalignment can manifest as a generation failure, not only
a ranking one.

## Visual analysis of the top-10 catastrophic failures

`scripts/visualize_catastrophic_failures.py` renders Reference + Search (with ground-truth diamond
and predicted-location cross) for the 10 largest-error pairs on the frozen benchmark, alongside each
one's full diagnostics (candidate rank/score, score margin, rotation/scale hypothesis, periodicity
score) — `outputs/visualizations/catastrophic_failures/`.

**9 of the top 10 have `periodicity_score` >= 0.64, and 7 of 10 have it at the maximum (1.0,
`mat_dense` preset). 7 of 10 have essentially zero rotation/scale drift.** This is a sharper finding
than the pooled averages alone suggest: rotation/scale grid-misalignment (Finding 2) is real and
measurable, but the *single worst* failures — the ones dragging mean/P95/max error up — are
overwhelmingly pure-periodicity cases, not rotation/scale cases. The mechanism why: a periodicity
failure can jump to a wrong repeat arbitrarily far away (hundreds of px, bounded only by the search
image size), while a grid-misalignment failure is typically a wrong-but-nearby location within the
same search window — still wrong, but rarely as catastrophic in magnitude unless it also coincides
with periodicity (as in `ch_worst_case_006`, which combines maximum periodicity, a boundary, *and*
severe rotation+scale grid misalignment simultaneously — the single worst combined case in the
benchmark, 583px error).

Concrete examples from the top 10 (all in `outputs/reports/baseline_failure_decomposition.csv`):

- Four of the ten (`dev_dense_periodic_{000,003,004,005}`) are `candidate_generation` failures with
  `periodicity_score=1.0`, zero rotation/scale, and a **score margin of only 0.002-0.007** between
  the winning (wrong) candidate and the runner-up — i.e. many nearly-identical-scoring peaks, the
  textbook aliasing signature, not a close call between two options but a genuine multi-way tie.
- `ho_scale_drift_006` and `ho_vignette_gamma_006` are `candidate_ranking` failures where ground
  truth *was* in the pool (rank 9-10) but scored far below the winner (0.30-0.40 ZNCC-point gap) —
  not a near-miss; the wrong location genuinely looked much better to the classical matcher.
- `val_mat_boundary_004` (725px error) is a `genuine_ambiguity` case *despite* a boundary being
  present (both mat and strip) — GT ranked 2nd, 0.009 behind the winner. A reminder that boundary
  presence is strongly protective on average but not a guarantee.

## Interactions

Six 2x2 (or 2x2x2) factorial cells, each holding every other factor at its default, isolating
exactly the named interaction. n=25 per cell (n=15 for the three-way interaction).

| Cells (rotation=0/4°, extra_scale=1.00/1.07) | Acc@5px |
|---|---:|
| r=0°, s=1.00 (both grid-aligned) | **84%** |
| r=0°, s=1.07 (scale off-grid only) | 36% |
| r=4°, s=1.00 (rotation off-grid only) | 52% |
| r=4°, s=1.07 (both off-grid) | 32% |

**Scale misalignment alone is more damaging than rotation misalignment alone** (36% vs. 52%), and
combining both is not dramatically worse than scale misalignment by itself (32% vs. 36%) — the two
failure modes mostly share the same ceiling rather than compounding multiplicatively.

| rotation × boundary | Acc@5px | | scale × boundary | Acc@5px |
|---|---:|---|---|---:|
| boundary, r=0° | 96% | | boundary, s=1.00 | **100%** |
| boundary, r=4° | 64% | | boundary, s=1.07 | **92%** |
| no boundary, r=0° | 44% | | no boundary, s=1.00 | **24%** |
| no boundary, r=4° | 20% | | no boundary, s=1.07 | **4%** |

**Boundary presence doesn't just help on average — it substantially rescues grid-misalignment
damage.** With a mat boundary in view, a grid-misaligned scale (1.07x) barely costs anything (100%
→ 92%); without one, the *same* grid-aligned scale (1.00x, no misalignment penalty at all) still
only reaches 24%, because `single_mat` crops draw from the same all-preset pool as the periodicity
sweep and inherit its ambiguity regardless of scale alignment. The two mechanisms are not
independent nuisances — boundary structure is doing most of the disambiguation work in every case
that has it available, whether the competing problem is periodicity or grid misalignment.

| noise × rotation | Acc@5px | | noise × scale | Acc@5px |
|---|---:|---|---:|
| dose=220, r=0° | 76% | | dose=220, s=1.00 | 72% |
| dose=220, r=4° | 48% | | dose=220, s=1.07 | 40% |
| dose=60, r=0° | 72% | | dose=60, s=1.00 | 64% |
| dose=60, r=4° | 56% | | dose=60, s=1.07 | 32% |

Noise shifts accuracy by only a few points at either rotation/scale condition — confirming Finding 4
holds in combination, not just in isolation.

**Triple interaction (rotation × scale × boundary, n=15/cell):**

| Boundary | r=0°,s=1.00 | r=0°,s=1.07 | r=4°,s=1.00 | r=4°,s=1.07 (worst case) |
|---|---:|---:|---:|---:|
| `mat_boundary` | 100% | 87% | 67% | **40%** |
| `single_mat` (no boundary) | 60% | 13% | 40% | **0%** |

The worst combined case (misaligned rotation *and* scale, no boundary) is a complete failure (0/15)
— but the identical rotation/scale misalignment with a boundary in view still recovers 40%. Boundary
presence is the single strongest lever found anywhere in this forensics sweep, including under the
worst acquisition conditions tested.

## Secondary degradation factors

12 factors, n=20/level (smaller than the primary sweeps — at this size a single flipped pair swings
accuracy by 5pp, so only a clear, repeated trend across levels should be trusted, not any single
level-to-level difference).

| Factor | Range tested | Acc@5px range | Trend |
|---|---|---|---|
| `barrel_pincushion` | k=-0.006 to +0.006 | 40-75% | **Real, monotonic-ish**: k=0 (75%) is clearly best; magnitude in either direction degrades performance (k=-0.006 worst at 40%) — consistent with barrel/pincushion's known uncorrected-GT mechanism (Phase 4), where induced positional error grows with `|k|`. |
| `beam_astigmatism` | ratio 1.0-2.0 | 55-80% | No consistent trend. |
| `beam_spot_size` | 0.5-2.5px blur | 70-75% | Flat. |
| `charging_streaks` | prob 0-0.03 | 50-85% | No consistent trend (non-monotonic). |
| `corner_rounding` | 0-4px | 60-80% | No consistent trend. |
| `gamma` | 0.7-1.6 | 60-85% | No consistent trend. |
| `linewidth_cd_bias` | 0-8nm | 55-80% | No consistent trend. |
| `pattern_collapse_threshold` | off, 5-20nm | 75-85% | Mild: collapse `off` (85%) slightly better than any nonzero threshold (75-80%), but within noise at this n. |
| `reference_dose` | 400-3000 | 65-80% | No consistent trend. |
| `salt_pepper` | 0-0.02 | 60-85% | No consistent trend. |
| `speckle_sigma` | 0-0.12 | 55-85% | No consistent trend. |
| `vignette_strength` | 0-0.55 | 60-80% | No consistent trend. |

**Only `barrel_pincushion` shows a real, mechanistically-expected effect** at this sample size — and
even its worst level (40%@5px) is still well above the periodicity sweep's worst level (25%@5px) or
the no-boundary/grid-misaligned combination (0%@5px). Every other secondary factor is statistically
indistinguishable from flat across its full tested range. This confirms Finding 4's conclusion at
finer granularity: none of the twelve remaining acquisition/distortion mechanisms are a meaningful
standalone bottleneck — boundary presence, periodicity, and hypothesis-grid alignment account for
essentially all of the classical pipeline's systematic difficulty.

## What this means for where to look next

1. **Boundary presence is the single strongest lever found anywhere in this sweep**, and it doesn't
   just help on average — it substantially rescues *both* other failure modes (periodicity and
   grid-misalignment) even in combination. The worst tested condition (misaligned rotation *and*
   scale) is a complete failure (0%) without a boundary in view, but still recovers to 40% with one.
   This reframes the priority question from "fix periodicity vs. fix the grid" to "how much of the
   remaining difficulty is inherent to the no-boundary, non-trivial-scale/rotation case specifically"
   — since that is where every failure mode concentrates.
2. **Periodicity/aliasing is a stronger and more clearly diagnosed standalone bottleneck than
   rotation/scale drift, and it dominates the catastrophic tail specifically** (see the visual
   analysis above: 9/10 worst failures are high-periodicity, 7/10 have zero rotation/scale drift at
   all) — and it fails at candidate generation, before ranking or refinement ever get a chance.
   **Tested and ruled out**: two alternative classical correlation representations
   (`experiments/periodicity/` — gradient-domain and intensity+gradient ensemble scoring) both made
   things *worse* (net rescue -9 and -4 respectively), and wider candidate retention alone
   (`experiments/wider_candidate_pool/`) is a structural no-op. This is now reasonably strong
   evidence that the fix isn't a cheap classical scoring tweak — it needs either genuinely more
   image context per candidate (untested) or a properly-data-scaled learned representation (the
   prior `embedding_reranker_v1` rejection was a data-scale problem, not an architecture problem —
   see `reports/DATASET_AUDIT.md` section 5 for the bounded expansion that would be needed before
   trying that again).
3. **The rotation/scale hypothesis grid is a concrete, low-risk classical-pipeline lever, but two
   variants have now been tested and neither cleanly passes the gate**: a globally denser grid
   (`experiments/finer_hypothesis_grid/`, net rescue +5, near-miss on one criterion) and a cheaper
   coarse-to-fine local refinement (`experiments/rotation_scale/`, net rescue +2, weaker and no
   real cost advantage). The global-grid design remains the stronger of the two if this is revisited.
   Scale misalignment is consistently more damaging than rotation misalignment across every test.
4. **Noise, drift, and jitter are not worth spending an experiment budget on** — they are not the
   bottleneck at any level tested, alone or in combination with rotation/scale.
