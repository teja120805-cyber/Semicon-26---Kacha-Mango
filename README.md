# DriftSense V2

DriftSense V2 is a navigation error recovery system for SEM (scanning electron microscope) imaging
of semiconductor DRAM structure: given a small, high-resolution **Reference** crop and a larger,
lower-resolution, acquisition-degraded **Search** image, find where the Reference sits inside the
Search image. It includes its own synthetic dataset generator, a classical localization pipeline,
an evaluation harness with a mandatory integration gate for candidate improvements, and a Streamlit
application.

**Input**: a Reference image (1000x1000, grayscale) and a Search image (1000x1000, grayscale) at a
nominal 10:1 magnification difference (tested 9:1-11:1), with up to a few degrees of rotation and
realistic SEM acquisition degradation on the Search side.
**Output**: predicted target-centre coordinates `(x, y)` in Search-image pixels — origin `(0, 0)` at
the top-left, `x` increasing right, `y` increasing down — plus a confidence score and an ambiguity
flag. If several candidate locations score comparably, the pipeline flags the prediction as
`ambiguous` (see `LocalizationResult.ambiguous`/`ambiguity_ratio` in `pipeline/localize.py`) rather
than silently guessing; see section 10 for how ties are currently broken.

## Evaluation criteria alignment (official Section 6 weighting)

Quoted directly from the Applied Materials Problem Statement help document, Section 6 ("Evaluation
Parameters" — "the sponsor presentation states the following provisional framework"), with a fourth
column added mapping each parameter to where this repo addresses it:

| Parameter | Weight | What evaluators will examine | Where this repo addresses it |
|---|---|---|---|
| Localization / inference | 50% | Coordinate accuracy on sponsor test data and computation time | `pipeline/localize.py` + section 9 below (pooled accuracy@1/2/4/5px, runtime/pair) + the app's Live Localization/Benchmark Dashboard pages |
| Synthetic augmentation code | 30% | Realism, diversity, reproducibility, literature-based justification | `generator/` + citations below and in `reports/` (Foi et al. 2008, Orji et al. 2018 *Nature Electronics*, the AM/HuggingFace reference generator) |
| Failure analysis / explainability | 10% | Understanding of failure causes, especially **repeated-pattern ambiguity** | `reports/ACCURACY_FORENSICS.md` (periodicity/aliasing found to be a stronger *standalone* bottleneck than rotation/scale drift) + the app's Failure Analysis page |
| RGB optical-image extension | Bonus | Optional generalization after completing the grayscale SEM task | Not yet implemented — grayscale SEM task is the current, required scope |
| Remaining core weight | 10% (pending) | Not defined in the supplied presentation | N/A — awaiting the detailed rubric; help doc Section 4 "IMPORTANT" box states the final evaluation utility, exact sub-pixel cutoff, official test dataset, and runtime environment "will take precedence when released" |

## 1. Why a macro-structured generator

Real DRAM dies are composed of discrete sub-array **mats** tiled with **strip/peripheral routing**
regions — not one continuous periodic field. DriftSense V2's generator renders each mat
independently (its own child RNG, its own word/bit-line jitter and structural pattern-collapse
behavior) and tiles them with strip regions on a shared fine canvas, so that real structural
disambiguation (mat boundaries, strip crossings) is available to the localization pipeline instead
of relying on an artificial injected landmark. See `reports/V2_ARCHITECTURE_PLAN.md` for the full
design rationale.

## 2. Generator architecture (`generator/`)

```
generate_macro_canvas()  ->  10000x10000 px, 1 nm/px, alternating mat/strip spans on both axes
   |-- each mat: pick_mat_preset() -> generate_mat() -> render_dram_cell_array()
   |     (independent child RNG per mat; word/bit line jitter, width jitter,
   |      structural pattern-collapse, optional corner rounding/linewidth bias)
   `-- strip cells: routing texture (flat fill + sparse orthogonal lines)

pick_crop_origin(mode, ...)  ->  (x0, y0), by structural family's crop mode
gt = (x0/10 + 50, y0/10 + 50)          # computed BEFORE any imaging call
reference = image_reference(crop)      # mild degradation
search    = image_search(full_canvas)  # blur -> exact 10x downsample -> optional
                                        # rotation/scale drift -> shear/drift -> noise
                                        # -> optional barrel/speckle/salt-pepper/charging/vignette/gamma
```

Six DRAM mat presets (`mat_dense`, `mat_narrow`, `mat_nominal`, `mat_relaxed`, `mat_compact`,
`mat_legacy`) with distinct feature size / word-pitch / bit-pitch, tiled via 15 structural families
across 4 splits (development/validation/held_out/challenge), plus a bonus "acquisition variant"
generator (one Reference + 5 differently-degraded Search re-acquisitions of the same location).
Full parameter/family tables, and every degradation's physical motivation, in
`reports/DEGRADATION_COVERAGE.md`.

**Ground-truth safety**: `generator/test_gt_safety.py` statically confirms no rendering function
accepts a GT-shaped parameter and no rendering module imports the localization pipeline, then
dynamically confirms the Search image's GT box actually matches a (correspondingly transformed)
downsampled Reference crop. `generator/test_dataset_validation.py` re-checks all of this against
already-written dataset files on disk (dimensions, grayscale, GT correctness, reproducibility,
metadata completeness).

## 3. Dataset structure (`data/`)

| Split | Pairs | Purpose |
|---|---:|---|
| `development` | 24 | Model training only |
| `validation` | 40 | Integration-gate criterion 1 |
| `held_out` | 40 | Integration-gate criterion 2 |
| `challenge` | 32 | Hardest combined conditions |
| `cross_generator` | 20 | External evaluation surface (independently-generated reference-implementation output, imported as data files only — never its code, never used for training) |

Each split directory holds `{pair_id}_reference.png`, `{pair_id}_search.png`, and
`ground_truth.{json,csv}`. `data/` is gitignored (regenerate via `scripts/generate_dataset.py` — fully
reproducible from the documented seed). See `reports/DATASET_AUDIT.md` for split independence and
sufficiency analysis.

## 4. Localization pipeline (`pipeline/`)

```
Reference -> preprocessing -> candidate_generation (multi-scale x multi-rotation ZNCC,
             spatially deduplicated) -> ranking (classical by default; learned only if
             explicitly requested) -> refinement (subpixel parabolic fit) -> final
             coordinate -> confidence / ambiguity score
```

Never reads ground truth. Used identically by `evaluation/evaluate.py` (offline benchmarking) and
`app/app.py`'s Live Localization page (interactive use) — there is exactly one localization code path.

## 5. Experimentation (`experiments/`)

Candidate localization improvements (learned re-rankers, alternative matching strategies, etc.) are
developed in isolation under `experiments/<name>/`, each with its own code, config, results, and a
concise `REPORT.md`. A candidate is only integrated into `pipeline/`/`model/` if it clears the
mandatory integration gate (`reports/V2_ARCHITECTURE_PLAN.md` section 8) against the frozen
classical baseline — improvement must be reproducible across seeds, hold on held-out and
cross-generator data, and not regress any structural family. Production code is untouched by
experiments that don't clear the gate. Current experiment status: `experiments/README.md`.

## 6. Evaluation methodology (`evaluation/`)

`evaluate.py` runs the pipeline over a split and writes per-pair predictions/errors. `metrics.py`
computes accuracy@{1,2,3,4,5}px, median/mean/P90/P95/max error, >10px/>50px failure rates, and
breakdowns by structural family, noise level, scale condition, rotation condition, boundary condition,
periodicity, and uniqueness. `plots.py` generates the 8 required plots. `benchmark.py` applies the
mandatory 7-criterion integration gate comparing a candidate model against the frozen classical
baseline.

## 7. Streamlit application (`app/app.py`)

Executive Summary, **Generate Sample** (every generator parameter — architecture preset, feature
size scale, crop placement, SEM imaging physics, acquisition noise/drift, distortion, die layout —
exposed as a live sidebar control that regenerates the sample on change, then runs the real
pipeline on it), Live Localization (uploads + runs the real pipeline), Visualization (browse any
evaluated pair), Benchmark Dashboard (all 8 plots), Failure Analysis (representative cases selected
from `outputs/reports/failure_analysis_cases.json`), **Experiment Results** (accuracy-forensics
findings plus every candidate change's integration-gate verdict), and System Information (versions,
seeds, hardware, model/gate status). Run `python scripts/run_demo.py`.

## 8. Reproducibility

- Every generated pair is `default_rng([seed, family_salt(split, family_name), pair_index])` —
  reproducible per-pair, not dependent on how much randomness earlier pairs consumed, and
  statistically independent across splits/families even at the same `pair_index` (fixed in
  generator v2.1.0 — see `reports/DATASET_AUDIT.md` section 2 for the cross-split leakage this
  closes).
- Every mat gets its own spawned child RNG.
- All generation/training/evaluation scripts take an explicit `--seed`.
- Model candidates are trained and gated across 3 independent seeds (`20260101`, `20260102`,
  `20260103`).
- Generator/dataset/model version strings are recorded in every metadata file and shown in the app's
  System Information page.
- `opencv-python-headless` is pinned to an **exact** version (`==5.0.0.93`) in `requirements.txt`,
  not a floor (`>=`). A cross-machine benchmark comparison on 2026-08-15 found that opencv's 4.x→5.x
  major version bump changes the internal numerics of `warpAffine`/`remap`/`GaussianBlur`/
  `sepFilter2D` (all used in `generator/degradation_models.py`) enough to produce materially
  different Search-image pixels from an identical seed — up to 115/255 gray-level differences across
  98% of pixels on affected pairs, which moved pooled accuracy@5px by several percentage points
  between two otherwise-identical runs. This was root-caused by regenerating the same pairs with each
  opencv major version and diffing pixels directly; numpy's `Generator`/`default_rng` API held up
  fine across the versions compared alongside it (2.4.4 vs 2.5.1 — negligible residual once opencv
  was matched). Anyone reproducing the benchmark numbers in section 9 should install exactly the
  pinned opencv version — an unpinned `pip install` can silently resolve to a newer major version and
  shift results.

## 9. Current benchmark

Classical baseline (as established in `reports/V2_BASELINE_REPORT.md`, prior to the 2026-08-15
compliance changes below): **71.2%@5px pooled** across 156 pairs (median error 0.34px, mean 66.8px,
catastrophic (>50px) rate 19.9%). `pipeline/candidate_generation.py`'s default scale/rotation
hypothesis grid was widened from 5x5 (25 combinations) to 9x9 (81 combinations), same span, half the
step size, after `experiments/finer_hypothesis_grid/` and `experiments/finer_grid_validation/` showed
this corrects hypothesis-grid misalignment failures on independent datasets without affecting any
case that doesn't involve rotation/scale drift (net rescue positive on 4 independent datasets;
runtime ~3.1x). Full breakdown: `reports/V2_BASELINE_REPORT.md`; controlled single-factor forensics:
`reports/ACCURACY_FORENSICS.md`; integration evidence: `reports/ACCURACY_IMPROVEMENT_PHASE.md`.

**2026-08-15 update — two compliance-driven changes integrated as documented gate exceptions**
(`reports/GATE_EXCEPTIONS.md`; both technically fail the automated gate's strict pooled
validation/held_out criteria for a structural reason unrelated to harm — see that file):

- The scale hypothesis grid widened again, 9x9→11x9 (99 combinations), to literally cover the
  9:1–11:1 range the Applied Materials help doc/pptx state ("robustness tests may span ~9:1–11:1"),
  up from the previous ±8%. The 3 dataset families that exercise scale drift (`ho_scale_drift`,
  `ch_combined_acquisition`, `ch_worst_case`) were regenerated to actually sample that literal edge.
  Evidence: `experiments/scale_range_v1/REPORT.md`.
- `pipeline/ranking.py::apply_center_tiebreak` gained a second, structurally-gated tier for the
  "if more than one matching region is found, return the one closest to the centre of the Search
  image" rule (stated 4 times across every source document) — see section 10 below for why the
  original tier alone essentially never fired. Evidence: `experiments/multiway_tiebreak_v1/REPORT.md`.

**Current production benchmark** (both changes above, full pipeline re-run end-to-end on the
regenerated dataset, verified on a clean install with the pinned `opencv-python-headless==5.0.0.93`
— see section 8): **74.36%@5px pooled** across 156 pairs (median error 0.334px, mean error 55.65px,
catastrophic (>50px) rate 16.67%, runtime 2.90s/pair). Per-split: `validation` 90.0%, `held_out`
65.0%, `challenge` 75.0%, `cross_generator` 80.0%, `development` 58.3%.

Because 3 families' dataset content changed as part of the scale-range bullet above, a strict
per-pair before/after comparison on those specific families reflects genuinely harder test images,
not just a pipeline difference — full isolated pipeline-only comparisons (same data, only the code
changed) are in each experiment's `REPORT.md`. The 74.36%@5px figure above is the number to cite as
the current, reproducible production benchmark: it was cross-checked against an independent run and
a real (root-caused, not assumed) ~3.16 percentage-point discrepancy against an earlier
differently-opencv-versioned run was found and resolved by pinning the exact opencv version (see
section 8) — it was not residual floating-point/tie-break noise. Prefer the relative (before/after,
same-environment) comparisons in the experiment reports when judging whether a specific change
helped, and reproduce this pooled number yourself with the pinned dependency versions before citing
it elsewhere.

A candidate learned re-ranker (`experiments/embedding_reranker_v1/`) failed the integration gate on
every criterion, across all 3 seeds — not integrated. Full analysis:
`reports/V2_MODEL_EVALUATION_REPORT.md`.

## 10. Known limitations

- Rotation/scale-drift and dense periodic structure are the classical pipeline's weakest points —
  see `reports/ACCURACY_FORENSICS.md` for the controlled analysis of which factor actually dominates
  and why.
- Barrel distortion's effect on ground truth is deliberately kept small (bounded to roughly a couple
  dozen px at most, for a crop near the image corner farthest from the distortion center) rather than
  analytically corrected, unlike rotation/scale drift, which IS corrected — see
  `reports/DEGRADATION_COVERAGE.md`.
- `pipeline/ranking.py::rank_with_model` exists and is tested, but is never the default — it must be
  passed explicitly (`ranking_mode="learned"`).
- **Tie-breaking**: `pipeline/ranking.py::apply_center_tiebreak` (called from `localize()` after
  ranking, before refinement) implements the "closest to the Search-image centre" rule as two
  tiers — a tight, near-exact numeric-equality check (unconditional on group size, essentially
  never fires alone: 0/156 on the frozen benchmark) plus a wider, structurally-gated multiway tier
  (requires ≥3 genuinely score-tied candidates within a bounded spatial spread, to distinguish a
  real multi-way periodicity tie from a coincidental pairwise near-tie between two unrelated,
  one-of-them-wrong candidates — two score-threshold-only attempts at this were tried and rejected
  first). Integrated as a documented gate exception, not a clean pass — see
  `reports/GATE_EXCEPTIONS.md` and `reports/TIE_BREAK_IMPLEMENTATION.md` section 12 for the full
  evidence, including why its confirmed benefit (one catastrophic rescue) is real but narrower than
  the scale-range change above.

## 11. How to run

```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv\Scripts\activate on native Windows shells
pip install -r requirements.txt

# 1. Generate the dataset (all splits + acquisition-variant demo + cross_generator import)
python scripts/generate_dataset.py

# 2. Validate it
python -m generator.test_gt_safety
python -c "from generator import test_dataset_validation as v; v.validate_dataset('data')"

# 3. Run the classical baseline evaluation + generate all plots
python scripts/evaluate_model.py

# 4. Localize a single Reference/Search pair (no dataset/manifest needed) -
#    or a whole batch via a CSV with reference_path,search_path columns
python scripts/localize_pair.py --reference path/to/reference.png --search path/to/search.png
python scripts/localize_pair.py --batch-csv pairs.csv --out predictions.csv

# 5. (Optional) Train a candidate learned re-ranker and run the integration gate
python scripts/train_model.py --seeds 20260101,20260102,20260103
python scripts/evaluate_model.py --learned-checkpoint experiments/embedding_reranker_v1/checkpoints/embedding_net_seed20260101.pt

# 6. Launch the Streamlit app
python scripts/run_demo.py
```

## 12. Repository structure

```
driftsensev2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── generator/    # synthetic DRAM macro-structure dataset generator
├── pipeline/     # localize.py + candidate_generation/ranking/refinement/feature_extraction
├── model/        # candidate learned re-ranker (EmbeddingNet) - not the production default
├── evaluation/   # evaluate.py, metrics.py, plots.py, benchmark.py (integration gate)
├── app/          # app.py - Streamlit application
├── scripts/      # CLI entry points (generate/evaluate/train/localize_pair/run_demo)
├── experiments/  # isolated candidate improvements, each with its own REPORT.md
├── reports/      # audit/benchmark/decision reports referenced throughout this README
├── data/         # generated dataset (gitignored - regenerate via scripts/generate_dataset.py)
└── outputs/      # generated plots/metrics (gitignored - regenerate via scripts/evaluate_model.py)
```
