# Drift-Sense

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-5.0.0-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-active-17A673)
![Hackathon](https://img.shields.io/badge/SEMICON%20India-Hackathon%202026-0B5FA8)

**AI-powered navigation-error recovery for wafer inspection tools.** Given a high-resolution
Reference crop and a lower-resolution, acquisition-degraded Search image of the same semiconductor
structure, Drift-Sense locates the Reference inside the Search image and returns its centre
coordinates — recovering the intended inspection site after stage drift, vibration, or thermal error.

*Applied Materials Problem Statement · SEMICON India Hackathon 2026 · Team Kaccha Mango*

---

## Table of Contents

- [Key Results](#key-results)
- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Evaluation Methodology](#evaluation-methodology)
- [Reproducibility](#reproducibility)
- [Known Limitations](#known-limitations)
- [Documentation](#documentation)
- [References](#references)
- [Team](#team)

---

## Key Results

**77.6% accuracy @5px, pooled across 156 evaluated pairs** — classical multi-scale, multi-rotation
ZNCC matching, no deep learning in the production path.

| Metric | Value |
|---|---:|
| Accuracy @0.5px (sub-pixel) | 66.7% |
| Accuracy @1px | 72.4% |
| Accuracy @2px | 77.6% |
| Accuracy @3px | 77.6% |
| Accuracy @4px | 77.6% |
| **Accuracy @5px (pooled)** | **77.6%** |
| Median error | 0.32 px |
| Mean error | 47.7 px |
| P90 / P95 error | 69.8 px / 433.8 px |
| Failure rate >10px | 21.2% |
| Catastrophic (>50px) rate | 14.1% |
| Mean / median runtime | 3.72 s/pair / 3.62 s/pair (machine-dependent) |
| Pairs evaluated | 156, across 5 independent splits |

| Split | Pairs | Accuracy @5px |
|---|---:|---:|
| `validation` | 40 | 92.5% |
| `cross_generator` | 20 | 80.0% |
| `challenge` | 32 | 71.9% |
| `development` | 24 | 70.8% |
| `held_out` | 40 | 70.0% |

**Read the four threshold rows together — they are the bimodality result.** Accuracy is identical
at 2, 3, 4 and 5 px (121/156 in every case): not one pair in the benchmark lands between 2 px and
5 px of truth. Loosening the tolerance from 2 px to 5 px buys nothing, and tightening it to 1 px
costs only 8 pairs. A prediction is either essentially exact or it has locked onto the wrong
lattice cell. Improving accuracy is therefore a question of candidate disambiguation, not of
sub-pixel precision.

**Sub-pixel performance: 66.7% of all pairs land inside half a pixel** (104/156), and the median
error is 0.32 px. `pipeline/refinement.py` fits a parabola to the correlation peak, so the returned
coordinates are continuous rather than quantised to the search grid.

**Runtime, hardware and timing method.** Runtime is measured with `time.perf_counter()` wrapped
around the `localize()` call only (`pipeline/localize.py`). It covers candidate generation,
scoring, ranking and sub-pixel refinement, and **excludes** image file I/O and dataset generation.
Execution is single-process CPU — no GPU is used anywhere in the production path, and no code in
`pipeline/` spawns threads or subprocesses.

The 3.72 s/pair figure was measured on:

| | |
|---|---|
| OS | Windows 11 (10.0.26200), AMD64 / 64-bit |
| CPU | Intel64 Family 6 Model 183 Stepping 1 (Raptor Lake), 24 logical cores |
| Python | 3.14.6 (CPython) |
| OpenCV | 5.0.0 (`opencv-python-headless==5.0.0.93`, pinned — see Reproducibility) |
| NumPy / SciPy / pandas | 2.5.1 / 1.18.0 / 3.0.5 |
| GPU | none used; `torch 2.13.0+cpu`, CUDA unavailable |

Only one core is used, so the 24-core count sets no expectation of parallel speed-up. The identical
benchmark takes 6.08 s/pair on a 2-core Linux container, so treat the ratio rather than the
absolute as portable. `python scripts/report_environment.py` prints this block for any machine, as
does the Streamlit app's **System Information** screen.

**Selective prediction.** Every result carries an `ambiguous` flag derived from the candidate
pool's own score distribution. Withholding flagged results trades coverage for reliability:

| Operating point | Coverage | Accuracy |
|---|---:|---:|
| All predictions | 100% (156/156) | 77.6% |
| **Unflagged predictions only** | **64.7%** (101/156) | **95.1%** |

The flag captures 85.7% of all failures. Both rows are measured on the same frozen 156-pair
benchmark as the tables above; [Known Limitations](#known-limitations) records what it does not
capture.

Full breakdown, all 8 required plots, and per-family/per-condition results: `outputs/reports/`,
`outputs/plots/`, or the Streamlit app's Benchmark Dashboard. Numbers regenerate deterministically —
see [Reproducibility](#reproducibility).

---

## Problem Statement

| | Specification |
|---|---|
| Reference image | 1000×1000 px, grayscale, 100x close-up view of the exact target location |
| Search image | 1000×1000 px, grayscale, 10x wider field of view containing the target |
| Scale relationship | Nominal 10:1, robustness-tested across 9:1–11:1 |
| Rotation | Up to a few degrees of residual drift |
| Output | Predicted target-centre coordinates `(x, y)` in Search-image pixels |
| Coordinate convention | Origin `(0, 0)` top-left; `x` increases right, `y` increases down |
| Multiple matches | If several valid matches exist, return the one closest to the Search-image centre |

If several candidate locations score comparably, the pipeline also flags the prediction as
`ambiguous` (see `LocalizationResult.ambiguous` / `.ambiguity_ratio` in `pipeline/localize.py`)
rather than silently guessing. The threshold governing that flag is calibrated rather than assumed:
it was fitted on tuning surfaces, then evaluated once on a held-back, independently-seeded set
before adoption. At the shipped value it fires on 35.3% of pairs and captures 85.7% of all
failures. Derivation and evidence: `reports/GATE_EXCEPTIONS.md`, exception 4.

---

## How It Works

**Dataset generator** — real DRAM dies are discrete sub-array **mats** tiled with **strip/peripheral
routing** regions, not one continuous periodic field. The generator renders each mat independently
and tiles them on a shared fine canvas, so real structural disambiguation (mat boundaries, strip
crossings) is available to the localization pipeline instead of an artificial injected landmark.

```
generate_macro_canvas()  ->  10000x10000 px, 1 nm/px, alternating mat/strip spans on both axes
   |-- each mat: pick_mat_preset() -> generate_mat() -> render_dram_cell_array()
   `-- strip cells: routing texture (flat fill + sparse orthogonal lines)

pick_crop_origin(mode, ...)  ->  (x0, y0), by structural family's crop mode
gt = (x0/10 + 50, y0/10 + 50)          # computed BEFORE any imaging call
reference = image_reference(crop)      # mild degradation
search    = image_search(full_canvas)  # blur -> exact 10x downsample -> optional
                                        # rotation/scale drift -> shear/drift -> noise
                                        # -> optional barrel/speckle/salt-pepper/charging/vignette/gamma
```

**Localization pipeline** — classical, explainable, and the same code path for every benchmark
number and every interactive use:

```
Reference -> preprocessing -> candidate_generation (multi-scale x multi-rotation ZNCC,
             spatially deduplicated, run under two template passbands) -> arm selection
             (keep the more decisive pool) -> ranking (classical by default; learned only
             if explicitly requested) -> refinement (subpixel parabolic fit) -> final
             coordinate -> confidence / ambiguity score
```

**Template passband matching.** The Reference and Search images reach the matcher through different
optical and resampling paths: the Reference is blurred at Reference resolution and then reduced 10x,
while the Search image is blurred at fine-canvas resolution and area-averaged. A template built
naively is therefore far sharper than the image it is correlated against, which costs correlation
fidelity at the true location and leaves too little margin over self-similar decoys.

The pipeline builds the candidate pool twice — once with the direct template, once with a template
convolved into the Search image's passband — and keeps whichever pool yields a more decisive winner,
measured as the score gap between the best candidate and the best candidate at a different location.
No threshold or tuned constant is involved, and acquisitions that are degraded in ways the blur
would compound simply select the direct template. `LocalizationResult.psf_sigma` reports which
passband was used for each pair. Passing `psf_selection=False` to `localize()` disables the second
arm.

The pipeline never reads ground truth. Candidate learned improvements are developed in isolation
under `experiments/<name>/` and only integrated into production if they clear a mandatory,
7-criterion integration gate against the frozen classical baseline (see
[Evaluation Methodology](#evaluation-methodology)).

---

## Project Structure

```
Drift-Sense/
├── README.md
├── requirements.txt
├── generate_dataset.py   # root entry point -> scripts/generate_dataset.py
├── localize.py           # root entry point -> scripts/localize_pair.py
├── Kaccha Mango_PS02.pptx
├── .gitignore
│
├── generator/    # synthetic DRAM macro-structure dataset generator
├── pipeline/     # localize.py + candidate_generation / ranking / refinement / feature_extraction
├── model/        # candidate learned re-ranker (EmbeddingNet) - not the production default
├── evaluation/   # evaluate.py, metrics.py, plots.py, benchmark.py (integration gate)
├── app/          # app.py - Streamlit application
├── scripts/      # CLI entry points (generate / evaluate / train / localize_pair / run_demo)
├── experiments/  # isolated candidate improvements, each with its own REPORT.md
├── reports/      # audit, benchmark, and design-decision reports (see Documentation)
├── references/   # bibliography + mechanism-to-code-to-source map (BibTeX included)
├── external/     # imported cross-generator evaluation surface (see section 3)
├── data/         # generated dataset (gitignored - regenerate via generate_dataset.py)
└── outputs/      # RESULTS: the 8 plots, metrics JSONs and the per-pair manifest (tracked)
```

**Mapping to the help document's recommended layout.** That document suggests
`generate_dataset.py`, `localize.py`, `configs/`, `src/`, `model/`, `results/` and `references/`.
The equivalents here: `generate_dataset.py` and `localize.py` are at the root as recommended;
`src/` is split into the four purposeful packages `generator/`, `pipeline/`, `evaluation/` and
`app/` rather than one bag; `model/` and `references/` match; **`results/` is `outputs/`** — the
eight required plots, `baseline_metrics.json` and the per-pair CSV/JSON manifest are committed
there, so a fresh clone has the results without regenerating anything. There is no `configs/`:
every tunable is a documented module-level constant next to the code that reads it (for example
`pipeline/localize.py::AMBIGUITY_THRESHOLD`, `generator/dataset_generator.py::DEFAULT_PARAMS`),
which keeps a value and its justification in one place instead of two.

---

## Getting Started

```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv\Scripts\activate on native Windows shells
pip install -r requirements.txt
```

```bash
# 1. Generate the dataset (all splits + acquisition-variant demo + cross_generator import)
python scripts/generate_dataset.py

# 2. Validate it
python -m generator.test_gt_safety
python -c "from generator import test_dataset_validation as v; v.validate_dataset('data')"

# 3. Run the classical baseline evaluation + generate all plots
python scripts/evaluate_model.py

# 4. Launch the Streamlit app
python scripts/run_demo.py
```

---

## Usage

```bash
# Localize a single Reference/Search pair (no dataset/manifest needed)
python scripts/localize_pair.py --reference path/to/reference.png --search path/to/search.png

# Or a whole batch via a CSV with reference_path,search_path columns
python scripts/localize_pair.py --batch-csv pairs.csv --out predictions.csv

# (Optional) Train a candidate learned re-ranker and run the integration gate
python scripts/train_model.py --seeds 20260101,20260102,20260103
python scripts/evaluate_model.py --learned-checkpoint experiments/embedding_reranker_v1/checkpoints/embedding_net_seed20260101.pt
```

The Streamlit app (`python scripts/run_demo.py`) additionally provides: Executive Summary, Generate
Sample (every generator parameter exposed as a live control), Live Localization (upload and run),
Visualization (browse any evaluated pair), Benchmark Dashboard (all 8 required plots), Failure
Analysis, Experiment Results, and System Information.

---

## Dataset

Six DRAM mat presets, tiled via 15 structural families across 5 splits:

| Split | Pairs | Purpose |
|---|---:|---|
| `development` | 24 | Model training only |
| `validation` | 40 | Integration-gate criterion 1 |
| `held_out` | 40 | Integration-gate criterion 2 |
| `challenge` | 32 | Hardest combined conditions |
| `cross_generator` | 20 | External evaluation surface (independently-generated reference-implementation output, imported as data files only — never its code, never used for training) |

Every pair stores the random seed, architecture, transforms, noise settings, scale, rotation, and
ground truth. `generator/test_gt_safety.py` statically confirms no rendering function ever sees a
GT-shaped parameter, then dynamically confirms the Search image's GT box actually matches a
(correspondingly transformed) downsampled Reference crop.

Only public structural knowledge and self-generated synthetic data are used — no confidential or
proprietary fab data.

---

## Evaluation Methodology

`evaluation/evaluate.py` runs the pipeline over a split and writes per-pair predictions/errors.
`metrics.py` computes accuracy@{1,2,3,4,5}px, median/mean/P90/P95/max error, >10px/>50px failure
rates, and breakdowns by structural family, noise level, scale/rotation condition, boundary
condition, periodicity, and uniqueness. `plots.py` generates the 8 required plots. `benchmark.py`
applies a mandatory 7-criterion integration gate comparing any candidate change against the frozen
classical baseline before it can be merged into production.

Four production behaviours are integrated as **documented gate exceptions**, each logged with the
criteria it did not clear and the evidence that justified it:

| # | Change | Why it is an exception |
|---|---|---|
| 1 | Literal 9:1–11:1 scale range | Narrowly-scoped spec-compliance fix; `validation` contains none of the affected families, so criterion 1 cannot register it |
| 2 | "Closest to Search-image centre" tie-break | Effect surface is a rare tie condition in specific families, too small for the gate's pooled-improvement bar |
| 3 | Template passband matching | Broad change validated across two independently-seeded datasets; both failing criteria are single-pair margins |
| 4 | Ambiguity-threshold calibration | **Not evaluable by the gate** rather than failing it — criteria 1–6 all measure prediction quality, and this change provably cannot alter a prediction |

Exception 4 is a different kind from the first three and is labelled as such: predictions,
accuracy@5px, catastrophic rate and runtime were verified bit-identical before and after, per pair.
Full rationale and evidence: `reports/GATE_EXCEPTIONS.md`. "In production" does not by itself mean
"passed all seven"; that file is the authoritative list.

A candidate learned re-ranker (`experiments/embedding_reranker_v1/`) failed the integration gate on
every criterion, across all 3 training seeds, and was not integrated — production ranking remains
classical. Full analysis: `reports/V2_MODEL_EVALUATION_REPORT.md`.

---

## Reproducibility

- Every generated pair is seeded `default_rng([seed, family_salt(split, family_name), pair_index])`
  — reproducible per-pair and statistically independent across splits/families.
- Every mat gets its own spawned child RNG; every generation/training/evaluation script takes an
  explicit `--seed`.
- Model candidates are trained and gated across 3 independent seeds.
- Generator/dataset/model version strings are recorded in every metadata file and shown in the
  app's System Information page.
- `opencv-python-headless` is pinned to an **exact** version (`==5.0.0.93`) rather than a floor,
  because OpenCV's 4.x→5.x bump changes the internal numerics of `warpAffine` / `remap` /
  `GaussianBlur`. The pin is load-bearing for **dataset generation**, and the mechanism is sharper
  than a rounding difference: `image_search` applies Poisson shot noise after those transforms, and
  because Poisson sampling uses rejection, a last-ulp difference in a single pixel's rate parameter
  desynchronises the entire subsequent noise stream. The result is an image with identical
  structure but a completely different noise realisation —
  `generator/test_dataset_validation.py`'s byte-equality check is designed to catch exactly this and
  does. Regenerate the dataset only with the pinned version.

- **Localization itself is far more robust than generation.** Verified directly: the full 156-pair
  benchmark was re-run on a different OS (Linux vs Windows) and a different OpenCV major version
  (4.13.0 vs the pinned 5.0.0.93), against identical input images.

  | Quantity | Result |
  |---|---|
  | accuracy @1px / @2px / @5px | **identical to the last digit** |
  | Pairs changing side of the 5px line | **0 of 156** |
  | `ambiguous` flag agreement | **100%** |
  | Max coordinate difference | 2.4 × 10⁻³ px |

  So the deliverable — predicted coordinates — reproduces across environments even when the
  dataset bytes do not. Runtime is the one figure that is genuinely machine-dependent: 3.72 s/pair
  on the reference machine, 6.08 s/pair on a 2-core container.

---

## Known Limitations

- **Crop uniqueness, not periodicity, governs accuracy.** Reference crops containing no
  distinguishing macro structure (`uniqueness_score = 0`) score 50.0% (24/48), against 89.8%
  (97/108) for all others; crops crossing a mat or strip boundary score 92.0% (81/88) against
  58.8% (40/68) for those crossing neither. Periodicity correlates with failure because
  non-unique crops tend to be periodic — with
  uniqueness held fixed it moves accuracy by well under one point. Analysis:
  `experiments/crop_uniqueness_ceiling/REPORT.md` and `reports/ACCURACY_FORENSICS.md`.
- Remaining failures are near-ties rather than blowouts: the correct location is scored within 0.05
  ZNCC of the chosen one in every failing pair measured. Correlation fidelity at the true location,
  not candidate ranking, is the active constraint.
- **Failures split roughly one-third discovery, two-thirds selection.** Measured across all 35
  failures on the frozen benchmark: in **37%** the true location is not within 5px of any pooled
  candidate, so no re-scoring or re-ranking stage can reach them; in the remaining **63%** the true
  location *is* in the pool — at median rank 3, losing to the winner by a median of only 0.029
  ZNCC. Pool recall is 0.917 and selector efficiency is 0.846, so the binding constraint is
  choosing correctly among near-ties rather than discovering the location. That said, twelve
  independent attempts at exactly that selection problem have all been rejected — nine in
  `experiments/ACCURACY_90_CAMPAIGN.md`, three in `experiments/REACHABILITY_CAMPAIGN.md` — so the
  near-tie is real but nothing tried so far can break it.
- The `ambiguous` flag is a reporting output only. Nothing in the pipeline reads it to make a
  decision, and it captures 85.7% of failures rather than all of them — a triage aid, not a
  correctness guarantee. Its precision depends on the failure base rate of the population measured,
  so it should be quoted against a stated population rather than as a single fixed figure.
- Running two template passbands doubles candidate generation. `psf_selection=False` restores
  single-arm cost where throughput matters more than the accuracy it provides.
- Barrel distortion's effect on ground truth is deliberately kept small rather than analytically
  corrected, unlike rotation/scale drift, which is corrected.
- `pipeline/ranking.py::rank_with_model` exists and is tested but is never the default; it must be
  requested explicitly (`ranking_mode="learned"`).
- The centre tie-break rule is implemented as two tiers — an exact-equality check plus a
  structurally-gated multiway tier for genuine multi-way periodicity ties — integrated as a
  documented gate exception rather than a clean pass. Full detail: `reports/TIE_BREAK_IMPLEMENTATION.md`.

---

## Documentation

Deep dives live in `reports/`; the table below is a map, not a substitute for reading them.
Every candidate change ever evaluated — integrated or rejected — keeps its own written report,
including the ones that failed and why. **42 experiment directories, 5 of which reached
production.** All but two carry a `REPORT.md` in the directory itself: `accuracy_forensics/`
predates that convention and is written up as `reports/ACCURACY_FORENSICS.md`, and
`finer_grid_validation/` splits into `FORWARD_HYPOTHESIS.md` and `INTEGRATION_RECOMMENDATION.md`
because the pre-registration and the verdict were written at different times. The Streamlit app's
**Experiment Results** screen renders the whole ledger live from disk, so it cannot drift from
what is actually here.

| Report | What it covers |
|---|---|
| `V2_ARCHITECTURE_PLAN.md` | System design rationale and the 7-criterion integration gate definition |
| `DATASET_AUDIT.md` | Split independence and sufficiency analysis |
| `DEGRADATION_COVERAGE.md` | Every degradation mechanism, its physical motivation, and its literature source |
| `ACCURACY_FORENSICS.md` | Controlled single-factor analysis of what actually drives failure |
| `V2_BASELINE_REPORT.md` | Full classical baseline benchmark breakdown |
| `V2_MODEL_EVALUATION_REPORT.md` | Learned re-ranker evaluation (rejected by the integration gate) |
| `GATE_EXCEPTIONS.md` | Every change shipped without a clean gate pass, with the criteria it failed and the evidence behind it |
| `TIE_BREAK_IMPLEMENTATION.md` | Full technical derivation of the two-tier centre tie-break mechanism |
| `ACCURACY_IMPROVEMENT_PHASE.md` | Integration evidence for the finer hypothesis-grid change |
| `FINAL_RESULTS.md` | Consolidated results summary |
| `PROJECT_STATUS.md` | Running record of phases completed, and the prioritised list of what remains |
| `RESEARCH_SURVEY_SCORING.md` | External literature/patent survey of scoring approaches, and the ranked proposals drawn from it |
| `HACKATHON_COMPLIANCE_CHECKLIST.md` | Point-by-point mapping of every stated problem-statement requirement to where it is satisfied |

Two consolidated experiment campaigns sit alongside these, in `experiments/`:

| Campaign | What it covers |
|---|---|
| `ACCURACY_90_CAMPAIGN.md` | Nine independent attempts at a 90% target, all rejected, and the shared failure mode |
| `REACHABILITY_CAMPAIGN.md` | Eight further experiments (weighted ZNCC, wider pool, PSR, anisotropic PSF, spectral/spatial filtering, aperiodic anchoring, NMS spatial diversity, DDIS) — seven rejected, one calibration change integrated; establishes the reachability ceiling that bounds the whole re-scoring class |
| `BATCH_2026-08-17_REPORT.md` | Final batch: alternative correlation scores (OTSDF/MACE, phase), axis decomposition, low-frequency 1-D cues, cross-hypothesis aggregation and gated escalation — all rejected or interim; closes the consensus/voting family with a mechanism |

---

## References

**Full bibliography: [`references/`](references/README.md)** — every entry carries a DOI, stable URL
or ISBN, and is mapped to the file and function that implements it, so a claim can be traced from
the README to the code to the paper. [`references/BIBLIOGRAPHY.bib`](references/BIBLIOGRAPHY.bib)
has the same entries as BibTeX.

The load-bearing ones:

- **Keeth, Baker, Johnson & Lin (2007)**, *DRAM Circuit Design: Fundamental and High-Speed Topics*,
  2nd ed., Wiley-IEEE Press, ISBN 978-0-470-18475-2 — the mat/periphery split this generator tiles.
  Corroborated by **Vogelsang (2010)**, MICRO-43, [10.1109/MICRO.2010.42](https://doi.org/10.1109/MICRO.2010.42),
  which states the geometry directly and is freely readable: each sub-array has bitline sense
  amplifiers and local wordline drivers surrounding it.
- **Reimer (1998)**, *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*,
  2nd ed., [10.1007/978-3-540-38967-5](https://doi.org/10.1007/978-3-540-38967-5) — probe-forming
  optics, spot size and astigmatism, behind the PSF blur model.
- **Foi, Trimeche, Katkovnik & Egiazarian (2008)**, *IEEE TIP* 17(10),
  [10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399) — the Poissonian-Gaussian
  composite behind the shot-noise / read-noise pair. SEM-specific counterpart: **Timischl, Date &
  Nemoto (2012)**, *Scanning* 34(3), [10.1002/sca.20282](https://doi.org/10.1002/sca.20282).
- **Sutton et al. (2006)**, *Meas. Sci. Technol.* 17(10),
  [10.1088/0957-0233/17/10/012](https://doi.org/10.1088/0957-0233/17/10/012) — separates fixed
  *spatial distortion* from time-varying *drift distortion* in a scanned image. That separation is
  why barrel/pincushion and shear/jitter are modelled as distinct terms rather than one blur.
- **Tanaka, Morigami & Atoda (1993)**, *Jpn. J. Appl. Phys.* 32(12S),
  [10.1143/JJAP.32.6059](https://doi.org/10.1143/JJAP.32.6059) — capillary-force resist collapse,
  the reason pattern collapse is a *spacing threshold* and bridges only interior gaps.
- **Orji et al. (2018)**, *Nature Electronics* 1(10),
  [10.1038/s41928-018-0150-9](https://doi.org/10.1038/s41928-018-0150-9) — CD, linewidth roughness
  and pattern-placement metrology, behind the per-line width and position jitter.
- **Lewis (1995)**, *Fast Normalized Cross-Correlation*, Vision Interface — the production score
  itself. DOI-bearing companion: **Briechle & Hanebeck (2001)**,
  [10.1117/12.421129](https://doi.org/10.1117/12.421129).

*Not a citation, listed for provenance:* Applied Materials' official starter resource,
*Drift-Sense Synthetic Data* (Hugging Face) — the hackathon's own starter kit, read as a coverage
reference only. Its code is never imported or executed; only its already-generated output images
are used, as an external evaluation surface.

**Honest coverage statement.** `reports/DEGRADATION_COVERAGE.md` audits 22 implemented mechanisms.
14 have a source that treats that exact mechanism; 6 are standard engineering models with no single
canonical citation, anchored instead to the textbook chapter covering their class; 2 are not
physical degradations at all. That breakdown is stated per-row rather than papered over — a
"citation" reading *Standard radial falloff model* is labelled as such.

---

## Team

**Kaccha Mango** — Applied Materials Problem Statement, SEMICON India Hackathon 2026.
