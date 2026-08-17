# DriftSense V2 — project brief

Navigation-error recovery for semiconductor wafer inspection. Applied Materials problem statement,
SEMICON India Hackathon 2026, team Kaccha Mango. Repository root is the checkout directory; no absolute paths are assumed anywhere in the code.

## The task

Given a 1000×1000 high-resolution grayscale **Reference** crop of a DRAM structure and a 1000×1000
**Search** image covering a 10× wider field of view containing it, predict the Reference's centre
coordinates in the Search image. Nominal 10:1 scale (robustness-tested 9:1–11:1), up to ~5° of
rotation drift, plus acquisition degradations (shot/read noise, raster shear, barrel, vignette,
gamma, speckle, charging). A prediction is correct if it lands within **5px**.

## Current state

**77.6% accuracy @5px, pooled over 156 pairs.** Classical multi-scale × multi-rotation ZNCC template
matching; no deep learning in the production path. Runtime 3.72 s/pair. Splits: validation 92.5%,
cross_generator 80.0%, challenge 71.9%, development 70.8%, held_out 70.0%.

Layout: `generator/` (synthetic DRAM dataset), `pipeline/` (production localizer), `model/`
(candidate learned re-ranker, not default), `evaluation/` (metrics + 7-criterion integration gate),
`app/` (Streamlit), `scripts/` (CLI entry points), `experiments/` (isolated candidate changes, each
with its own REPORT.md), `reports/` (design and analysis docs).

## Working conventions — please follow these

- **Never modify `pipeline/`, `generator/`, or `model/` without explicit approval.** All research
  happens in a self-contained `experiments/<name>/` folder that imports production code unmodified.
- **Every experiment gets an honest `REPORT.md`, including rejections** — the reasoning behind a
  failure is the deliverable, not just the verdict.
- **Dev-only tuning**: sweep hyperparameters on the 24-pair `development` split *only*, choose one
  configuration, then run the frozen 156-pair benchmark *once*. Never tune on the reported set.
- **Null controls**: every change needs a parameter setting that provably reproduces production
  bit-for-bit, verified per pair rather than assumed.
- **Second-seed validation** before proposing anything for integration — this project has caught a
  6/7-gate-passing result as a single-seed artifact that way.
- **Never fabricate or estimate a result.** If something wasn't measured, say so.
- Watch for benchmark-mining: repeated frozen-benchmark runs across a line of work can select on
  noise even when each individual step is disciplined. Disclose the count.

## Established findings — do not re-derive

- **Every remaining failure is a near-tie.** The true location scores within 0.05 ZNCC of the chosen
  one, max ratio 1.048× — not the "4× gap" an earlier report claimed (that claim is falsified and
  corrected in place).
- **Crop uniqueness governs accuracy, not periodicity.** `uniqueness_score = 0` → 50.0% (n=48);
  above 0 → 89.8% (n=108). Holding uniqueness fixed, periodicity moves accuracy by ~0.4pp.
  Crops crossing a mat or strip boundary score 92.0% (n=88) vs 58.8% (n=68) for those crossing
  neither. Periodicity is a **confound**.
- **The task is well-posed.** 154/156 crops have a unique origin at 0.95 content identity; zero of
  the failures picked a location matching ground truth above 0.95 (identity check run on the
  superseded 40-failure set; production now has 35 failures).
- **Template fidelity is the bottleneck.** The two Search locations differ at ZNCC 0.732 while the
  template separates them by 0.0098 — the discriminating information is destroyed in template
  construction, upstream of ranking.
- Failure decomposition, measured across all **35** failures: **37% discovery** (the true location
  is not within 5px of any pooled candidate, so no re-scoring or re-ranking stage can reach it) /
  **63% selection** (the true location *is* in the pool but loses to a near-tie).

## Closed by evidence — do not retry as-is

- Periodicity-targeted re-ranking (nine independent rejections, `experiments/ACCURACY_90_CAMPAIGN.md`).
- Learned re-rankers trained from scratch on this dataset (three rejections, including at 7.5× data —
  the "more data" hypothesis in `reports/DATASET_AUDIT.md` is falsified).
- Blur estimation from image spectra as a family discriminator (`experiments/psf_matched_adaptive/`).
- Broadband spectral prewhitening and lattice notching (`experiments/parallel_pipeline/`) — real
  signal, but amplifies the noise floor and breaks as many pairs as it fixes.
- General-purpose learned matchers and sparse keypoint pipelines — negative evidence in
  `reports/RESEARCH_SURVEY_SCORING.md`, plus our own `experiments/keypoint_candidate_fusion/`.

## Known issues in the methodology itself

- `development` contains **no degraded-acquisition family**, so every dev-only sweep is structurally
  blind to over-smoothing damage. This affects past conclusions, not just future ones.
- `validation` is only 40 pairs at 92.5% — a ceiling that has blocked two near-miss results from
  passing the gate on that criterion alone.
- `AMBIGUITY_THRESHOLD` was miscalibrated at `0.92` and has been recalibrated to **`0.990`**
  (gate exception 4). It now fires on 55/156 pairs (35.3%) at 54.5% precision and 85.7% failure
  recall — a usable triage flag rather than one that fires on nearly everything. A pool-internal
  score-gap statistic still reaches 95% failure recall at 49% coverage and remains the stronger
  basis for selective escalation.
- Four production changes are in as **documented gate exceptions** — "in production" does not mean
  "passed all 7 criteria". See `reports/GATE_EXCEPTIONS.md`.

## Next planned work

**P3 — discriminability-weighted ZNCC**, from `reports/RESEARCH_SURVEY_SCORING.md`. Weight template
pixels by local distinctiveness so the ~10% of aperiodic boundary structure isn't diluted by the
~90% periodic array. Closed form given in the survey; stays FFT-fast; `w = uniform` is the null
control. Preferred over the band-limited spectral fix because it reweights pixels rather than
reshaping the spectrum, so it cannot amplify the noise floor — which is precisely how P1 failed.

## Read these first

`reports/PROJECT_STATUS.md` (the arc and what remains) · `reports/RESEARCH_SURVEY_SCORING.md`
(industry practice, literature, ranked proposals) · `experiments/parallel_pipeline/REPORT.md` (why
P1 failed) · `reports/GATE_EXCEPTIONS.md` (what shipped without a clean gate pass).
