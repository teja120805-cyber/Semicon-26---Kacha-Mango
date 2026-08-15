# DriftSense V2 — Architecture Plan

Status: **written before implementation began**. This document is the design contract for the
generator, pipeline, and evaluation methodology, and the rationale behind each major decision. The
generator design was informed by a read-only study of the Applied Materials/Hugging Face reference
generator (`src/`, `baseline_solution/` — the hackathon's official starter resource); its code is
never imported, copied, or executed by this project, only studied for design ideas and, separately,
used as a frozen external evaluation surface via its already-generated output images (section 8).

---

## 1. Why a macro-structured generator

Real DRAM dies are composed of discrete sub-array **mats** tiled with **strip/peripheral routing**
regions — not one continuous periodic field. A generator that only produces a uniform repeating
pattern makes the localization problem artificially easy (an injected landmark or brightness cue can
disambiguate an otherwise uniform field) or artificially hard in a way that doesn't match real
acquisition (no real structural boundaries ever appear). DriftSense V2's generator instead renders
each mat independently — its own child RNG, its own word/bit-line jitter and structural
pattern-collapse behavior — and tiles them with strip regions on a shared fine canvas, so that real
structural disambiguation (mat boundaries, strip crossings) is available to the localization pipeline
instead of relying on an artificial signal. This macro-structured composition is the central design
choice V2 is built around.

---

## 2. Generator design decisions

| Decision | Rationale |
|---|---|
| Shared fine canvas → Reference crop + downsampled Search | Only way to guarantee exact, leak-free ground truth |
| Macro layout = alternating mat/strip spans on both axes | Matches how real DRAM dies are physically composed |
| Independently-seeded per-mat RNG | Real chips don't have correlated noise between physically separate mats; this is also what makes mats non-fingerprinted |
| Named DRAM presets (pitch/width families) as the source of mat-to-mat distinguishability, not brightness | Keeps "uniqueness" physically motivated instead of an injected signal |
| Crop-placement **modes** (`random`, `single_mat`, `strip_center`, `mat_boundary`, `same_preset_boundary`, `multi_mat`) as the difficulty-family axis | More legible and auditable than a single continuous probability knob when defining named, reproducible difficulty families |
| GT = center of the crop's footprint in Search-pixel space, computed from `(x0, y0)` before any imaging call | Verified leak-free by an automated test (`generator/test_gt_safety.py`) |
| Per-pair-index-independent seeding (`seed, pair_index → RNG`, not one sequential stream) | Reproducing pair #47 shouldn't require regenerating #0-46 first |
| Exactly-10x base magnification (`SCALE_FACTOR = 10`, fixed) | A microscope's magnification ratio is a hardware constant, not a random variable; scale *drift* is a separate, smaller acquisition-stage effect layered on top (section 4) |
| Structural pattern-collapse (adjacent-line bridging below a gap threshold), implemented directly in the line-mask renderer and covered by a unit test | Physically real (etch/capillary bridging) degradation on the shared canvas |
| Sample families: one Reference, N differently-degraded Search re-acquisitions of the same location | Tests "does the model find the same place under different acquisition conditions" — complementary to "does it find the right place in one acquisition" |
| Charging streaks, speckle, salt-and-pepper, barrel distortion, corner rounding | Present in the reference generator; added here and audited in `reports/DEGRADATION_COVERAGE.md` |
| A multi-scale, multi-rotation-hypothesis candidate stage in the localization pipeline (section 5) | Real acquisitions carry small stage rotation and calibration-scale drift, so the search must cover a hypothesis grid rather than assume a single fixed scale/rotation |
| A formal integration gate (section 8) | Any pipeline/model change must pass every criterion in the gate before being merged into production, so no change ships on the strength of a single favorable number |

---

## 3. Design choices deliberately not modeled

- **A single continuous periodic field as the whole Search image** — superseded entirely by the
  macro mat/strip generator; realistic DRAM localization has to contend with genuine structural
  boundaries, not just a uniform repeating texture.
- **An injected artificial disambiguation signal** (e.g. a probabilistically-placed "landmark" in an
  otherwise uniform field) — this was considered and rejected. V2's uniqueness comes only from real
  structure (mats/strips/positional variation), never an injected marker.
- **Sampling the nm-per-px magnification ratio itself as a generative random variable** — V2 uses a
  fixed exact 10x base with a small *residual* scale-drift degradation layered on top post-downsample
  (section 4), a more physically legible model: hardware magnification is fixed, calibration drift is
  a separate, smaller effect.
- **Arbitrary-orientation astigmatism** — V2 keeps astigmatism axis-locked to the scan axes, since
  arbitrary orientation has no clear physical motivation for a raster-scanned instrument.

---

## 4. Generator architecture

```
generate_macro_canvas(seed)                       10000x10000, 1 nm/px
   |-- row spans, col spans (alternating mat/strip, "regular" tiling)
   |-- per-mat: pick_mat_preset() -> generate_mat() -> render_dram_cell_array()
   |     (independent child RNG per mat: word/bit line jitter, width jitter,
   |      structural pattern-collapse)
   `-- strip cells: routing texture (flat fill + sparse orthogonal lines)
        -> returns canvas + mat_rects + strip_rects

pick_crop_origin(mode, mat_rects, strip_rects, rng)
   `-- mode in {random, single_mat, strip_center, mat_boundary,
               same_preset_boundary, multi_mat}
        retries up to N times against a structural predicate, then falls back
        to random with the fallback COUNTED and recorded in metadata

gt = (x0/10 + 50, y0/10 + 50)                     computed BEFORE any imaging

reference = image_reference(canvas[y0:y0+1000, x0:x0+1000])   mild degradation
search    = image_search(canvas)                              full pipeline:
   blur -> structural collapse (pre-baked into canvas, not here) ->
   10x area-average downsample -> rotation/scale drift (optional) ->
   raster shear/drift -> Poisson+Gaussian noise -> vignette/gamma ->
   barrel distortion -> charging streaks -> speckle -> salt & pepper
   (each stage independently toggleable; off-by-default stages are OFF,
   not silently no-op'd — see reports/DEGRADATION_COVERAGE.md)

metadata = build_metadata(...)                    -> ground_truth.json/.csv
```

Locked conventions, stated explicitly so nothing downstream has to guess:

| Property | Value |
|---|---|
| Reference / Search size | 1000x1000, grayscale `uint8` |
| Fine canvas | 10000x10000, 1 nm/px |
| Base scale factor | exactly 10 (fixed constant, not sampled) |
| Coordinate convention | `x` = column, `y` = row, origin top-left; GT = **center** of the crop's 100x100-search-px footprint |
| Determinism | `rng = default_rng([seed, family_salt(split, family_name), pair_index])`; per-mat child RNGs spawned from the layout RNG |
| GT computed | strictly before any degradation/imaging call; verified by an automated leak-safety test (section 6) |

Residual scale/rotation drift (acquisition-stage, not generative-stage) is applied **after** the exact
10x downsample, only in families that request it, sampled from small ranges (`extra_scale ∈
[0.93, 1.07]`, `rotation ∈ [-4°, 4°]`) — this correctly separates "what the objective magnifies by"
(fixed) from "what stage/calibration drift adds on top" (variable).

---

## 5. Data flow (dataset → pipeline → model → evaluation → app)

```
generator/dataset_generator.py
        |  writes PNGs + ground_truth.json/.csv per split
        v
data/{development,validation,held_out,challenge}/   (V2's own generator)
data/cross_generator/                                (external eval-only,
                                                        see section 8 - real
                                                        reference-generator
                                                        OUTPUTS copied in,
                                                        never its code, never
                                                        used for training)
        |
        v
pipeline/localize.py   (candidate_generation -> feature_extraction ->
                         matching -> ranking -> refinement -> confidence)
        |  never reads ground_truth.json
        v
evaluation/evaluate.py -> per-pair predictions + errors
evaluation/metrics.py  -> @kpx, median/mean/P90/P95/max, runtime, breakdowns
evaluation/benchmark.py -> compares baseline vs. candidate, integration
                           gate decision
        |
        v
outputs/{plots,reports}/, reports/V2_BASELINE_REPORT.md, reports/V2_MODEL_EVALUATION_REPORT.md
        |
        v
app/app.py  (Streamlit - reads the JSON/CSV/PNG artifacts above; can also run
             the live pipeline on user-uploaded images)
```

The pipeline package is used identically by `evaluation/evaluate.py` (offline benchmarking) and
`app/app.py` (interactive/live use) — there is exactly one localization code path, never a
Streamlit-only reimplementation.

---

## 6. Reproducibility strategy

- Every generated pair is traceable to a single seed/family/pair-index triple; regenerating any split
  with the same seed reproduces it byte-for-byte (verified by an automated test that generates the
  same pair twice and diffs the PNGs and metadata).
- `generator/test_gt_safety.py` statically scans every rendering module for GT-shaped parameter names
  and for any import of `pipeline/`, then dynamically verifies the Search-image patch at the GT box
  matches a downsampled Reference crop within a small tolerance — both a "the answer can't leak
  forward" check and a "the answer is actually right" check.
- All dataset-generation, training, and evaluation scripts take an explicit `--seed`; no script reads
  system entropy for anything that affects a reported number.
- `model/TRAINING_PROTOCOL.md` fixes the training seed(s) in advance; the benchmark reports whether
  results are stable across at least 2 additional seeds before any integration decision is made.
- Software versions (numpy/opencv/torch/streamlit/etc.) and generator/dataset version strings are
  recorded in every metadata file and surfaced in the Streamlit "System Information" panel.

---

## 7. Model architecture strategy (as planned; see outcome below)

**Baseline first (mandatory):** a pure classical pipeline — multi-scale, multi-rotation normalized
cross-correlation candidate generation, subpixel parabolic refinement, and a confidence score derived
from the peak-to-second-peak ratio. This is evaluated in full (all metrics x all breakdowns) before
any learned component is written.

**Candidate learned model (as planned):** a small CNN Siamese embedding network (reference crop and
candidate Search crop -> shared-weight conv tower -> L2-normalized embedding, trained with triplet
loss), with hard negatives drawn from the classical candidate pool itself (periodic-repeat peaks,
nearby-mat peaks, strip-boundary peaks) rather than random crops, since the anticipated hard cases
(periodic repeats, same-preset mat boundaries, strip-adjacent decoys) are fundamentally a
hard-negative embedding problem. Training was scoped to the `development` split only, with
validation/held_out/challenge/cross_generator never touched during training or model selection.

**Actual outcome:** this candidate re-ranker (`experiments/embedding_reranker_v1/`) was trained and
evaluated against the integration gate below and did not pass — see
`reports/V2_MODEL_EVALUATION_REPORT.md` for the full result. Production remains the classical
baseline. `reports/FINAL_RESULTS.md` summarizes every candidate approach tested and its outcome.

---

## 8. Integration gate

A candidate model or pipeline change is only wired into production if **all** of the following hold;
otherwise it stays under `experiments/<name>/` permanently, with its own honest report, and production
keeps using the classical baseline:

1. Improves accuracy@5px on `validation` vs. the classical baseline.
2. Improves accuracy@5px on `held_out` vs. the classical baseline.
3. Improves (or is not worse, given small-n) on `cross_generator` vs. the classical baseline.
4. No increase in >50px catastrophic-failure rate on any split.
5. No regression on any currently-correct case class (checked per structural family, not just pooled).
6. Runtime per pair stays within the same order of magnitude as the classical baseline.
7. The improvement holds across at least 2 additional random seeds, not just the first one.

This gate is evaluated in `evaluation/benchmark.py` and its outcome (pass/fail per criterion) is
written into the relevant report — including a fail, if that's what happens. See
`reports/FINAL_RESULTS.md` for how this was applied across every candidate approach tested.

---

## 9. Cross-generator testing strategy

The reference generator's **code** is never imported, copied, or adapted into V2. Its already-generated
**output images** (20 Reference/Search pairs with ground truth, produced by running the real
Applied Materials/Hugging Face reference generator) are copied — as data files only — into
`data/cross_generator/`, strictly as a frozen, external evaluation surface. This split:

- Is never used for training or model/checkpoint selection.
- Is evaluated exactly like every other split (same metrics, same pipeline code).
- Is reported **separately** from V2's own generator splits everywhere (baseline report, model
  report, Streamlit dashboard) — never pooled into a single blended number.

---

## 10. Experiment isolation strategy

`experiments/<experiment_name>/` holds a **copy** of whatever module(s) an experiment modifies, plus
its own config and its own output directory. Experiments import shared, unmodified code from
`generator/`, `pipeline/`, `model/` where unchanged, but never edit those packages in place. Only a
model/pipeline change that has passed every criterion in section 8 may be merged into `pipeline/` or
`model/` proper, via a normal code change with the gate results cited in its commit message.

---

## 11. Streamlit architecture

See README section 7 for the current, up-to-date section list — this design contract predates several
sections (Generate Sample, Experiment Results) added after the accuracy-improvement research phase.

## 12. Vocabulary lock

- **Structural family** — a named difficulty/condition category (crop mode + degradation overrides).
  Defined in `generator/dataset_generator.py::FAMILIES`.
- **Acquisition variant** — one of several differently-degraded Search re-acquisitions of the *same*
  Reference/location. Defined in `generator/dataset_generator.py::generate_acquisition_variants`.

These terms are used consistently in code, metadata field names, and every doc in this repository.
