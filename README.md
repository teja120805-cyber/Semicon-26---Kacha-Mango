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
| Accuracy @1px | 72.4% |
| Accuracy @2px | 77.6% |
| **Accuracy @5px (pooled)** | **77.6%** |
| Median error | 0.32 px |
| Mean error | 47.7 px |
| P90 / P95 error | 69.8 px / 433.8 px |
| Failure rate >10px | 21.2% |
| Catastrophic (>50px) rate | 14.1% |
| Mean runtime | 6.3 s/pair |
| Pairs evaluated | 156, across 5 independent splits |

| Split | Pairs | Accuracy @5px |
|---|---:|---:|
| `validation` | 40 | 92.5% |
| `cross_generator` | 20 | 80.0% |
| `challenge` | 32 | 71.9% |
| `development` | 24 | 70.8% |
| `held_out` | 40 | 70.0% |

Error is strongly bimodal: the median prediction lands within a third of a pixel, while the
remaining failures are wrong-location misses rather than imprecise ones. Improving accuracy is
therefore a question of candidate disambiguation, not subpixel precision.

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
rather than silently guessing.

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
├── data/         # generated dataset (gitignored - regenerate via scripts/generate_dataset.py)
└── outputs/      # generated plots/metrics (gitignored - regenerate via scripts/evaluate_model.py)
```

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

Three production behaviours are integrated as **documented gate exceptions** — changes shipped
despite not clearing all seven criteria, each logged with the criteria it failed and the evidence
that justified it: the literal 9:1–11:1 scale range, the "closest to Search-image centre" tie-break
rule, and template passband matching. The first two are narrowly-scoped spec-compliance fixes whose
effect surface is too small for the gate's blanket "must broadly improve pooled validation/held_out"
bar; the third is a broad change validated across two independently-seeded datasets whose two
failing criteria are single-pair margins. Full rationale and evidence: `reports/GATE_EXCEPTIONS.md`.
"In production" does not by itself mean "passed all seven"; that file is the authoritative list.

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
- `opencv-python-headless` is pinned to an **exact** version (`==5.0.0.93`) rather than a floor. A
  cross-machine comparison found OpenCV's 4.x→5.x major version bump changes the internal numerics
  of `warpAffine` / `remap` / `GaussianBlur` enough to shift pooled accuracy@5px by several
  percentage points from an identical seed. Reproduce the numbers above with the pinned version
  installed exactly.

---

## Known Limitations

- **Crop uniqueness, not periodicity, governs accuracy.** Reference crops containing no
  distinguishing macro structure (`uniqueness_score = 0`) score 43.8%, against 88.0% for all
  others; crops crossing a mat or strip boundary score 89.8% against 54.4% for those crossing
  neither. Periodicity correlates with failure because non-unique crops tend to be periodic — with
  uniqueness held fixed it moves accuracy by well under one point. Analysis:
  `experiments/crop_uniqueness_ceiling/REPORT.md` and `reports/ACCURACY_FORENSICS.md`.
- Remaining failures are near-ties rather than blowouts: the correct location is scored within 0.05
  ZNCC of the chosen one in every failing pair measured. Correlation fidelity at the true location,
  not candidate ranking, is the active constraint.
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
Every candidate change ever evaluated — integrated or rejected — keeps its own `REPORT.md` under
`experiments/<name>/`, including the ones that failed and why.

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

---

## References

- Foi, Trimeche, Katkovnik & Egiazarian (2008), *"Practical Poisson-Gaussian Noise Modeling and
  Fitting for Single-Image Raw-Data"* — basis for the Poisson shot-noise / detector read-noise model.
- Orji et al. (2018), *"Metrology for the next generation of semiconductor devices"*, **Nature
  Electronics** — basis for the per-line position-jitter (cumulative random walk) model.
- Applied Materials' official starter resource, *Drift-Sense Synthetic Data* (Hugging Face) — read
  as a reference for degradation coverage only; its code is never imported or executed, only its
  already-generated output images are used as an external evaluation surface.

Full 22-mechanism citation table, mapping every implemented degradation to its physical motivation
and source: `reports/DEGRADATION_COVERAGE.md`.

---

## Team

**Kaccha Mango** — Applied Materials Problem Statement, SEMICON India Hackathon 2026.
