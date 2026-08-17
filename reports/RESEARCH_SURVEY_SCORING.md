# Research survey — how this problem is solved elsewhere, and what to try next

**Date:** 2026-08-16. **Status:** survey and proposal. Nothing here has been implemented or measured
on our data; every claim below is sourced from external literature, and every expectation is a
hypothesis until an experiment says otherwise.

**Why this survey exists.** Production is at 77.6%@5px. Our own diagnostics
(`experiments/oracle_ceiling_diagnostic/`, `experiments/crop_uniqueness_ceiling/`) established
three things: every remaining failure is a **near-tie** (the true location scores within 0.05 ZNCC
of the chosen one, never worse than a 1.048x ratio); the information needed to separate them **is
present in the data** (154/156 crops have a unique origin at 0.95 content identity); and the
bottleneck is that **the correlation score itself cannot see it**. That is a scoring-function
problem, and it is worth checking what the rest of the world does about scoring before inventing
more of our own.

---

## 1. The single most important finding

**Three independent lines of evidence say ZNCC is the wrong similarity measure for periodic
structure**, and they agree for the same underlying reason.

**From signal processing.** ZNCC *is* the matched filter, and the matched filter is provably
optimal **only when the competing background is white noise**. Ours is the opposite extreme: a DRAM
cell array concentrates almost all of its energy into a handful of lattice harmonics. Under
coloured interference the optimal detector is the **prewhitened** matched filter — divide by the
interference power spectrum before correlating. This is not a heuristic; it is the textbook
correction for exactly the situation we measured.

**From commercial practice.** The two dominant commercial matching engines — Cognex PatMax/PatQuick
and MVTec HALCON shape-based matching — abandoned grey-level correlation decades ago in favour of
**gradient-direction** similarity. Steger's published formulation scores the mean normalized dot
product of gradient-direction vectors. Because every vector is unit-normalized, the score is
invariant to arbitrary illumination change and missing features contribute ~0 on average, so the
score reads directly as "fraction of the model visible."

**From the semiconductor industry itself.** Hitachi's design-based fine matching extracts edges from
the SEM image and aligns those against design polygons. Notably, a recent semiconductor alignment
patent (US20240095935) uses a learned model to *translate design into predicted SEM appearance* and
then **still measures the final offset with NCC** — industry uses learning for representation, not
for matching. That is a useful signal that our architecture is not the problem; our score is.

---

## 2. What the industry does about periodicity specifically

The patent literature is far more revealing than any vendor documentation, and four distinct
production strategies appear:

- **Anchor on the non-repeating structure, deliberately ignore the array.** KLA's US9830421B2
  explicitly rejects device features that are "repeating" or "lacking uniqueness" and targets
  **array corners and boundaries** instead. This is our `crosses_mat_boundary` / `crosses_strip_boundary`
  finding stated as production doctrine — we measured 89.8% accuracy on boundary-crossing crops
  against 54.4% for crops crossing neither.
- **Directional uniqueness.** US8073242B2 accepts targets with no uniqueness in X but uniqueness in
  Y, requiring uniqueness only along the axis that matters. Our two axes may be separately solvable.
- **Detect periodicity by FFT, then clamp the search.** US20090103799A1 classifies patterns as
  repeating via 2D FFT energy concentration and restricts the search range accordingly.
- **Break half-pitch ambiguity with region statistics, not edges.** Hitachi's JP5639797B2 states the
  failure mode outright: matching on edge information alone means "a position shifted by a half
  cycle is erroneously recognized," and adds an interior luminance-statistics term. **This is an
  important caution against going purely gradient-based** — see §5.

---

## 3. Ranked proposals

Ordered by (expected value × cheapness × alignment with what we actually measured). Each is scored
against the same discipline as prior work: a null-control setting that provably reproduces current
production, dev-only tuning, then one frozen-benchmark run, then second-seed validation.

### P1 — Spectral prewhitening of the correlation *(strongest candidate)*

The direct implementation of §1's signal-processing argument. Instead of correlating raw template
against raw image, divide by a power of the spectrum magnitude first:

```
H(f) = F_template(f)* / (|F_image(f)|² + λ)^ρ
```

`ρ` interpolates continuously between plain correlation (`ρ = 0`) and phase correlation (`ρ = 1`),
with partial whitening (`ρ ≈ 0.3–0.7`) the usual practical choice — full whitening amplifies SEM
shot noise, which is severe at our `dose_search = 220`.

**Why this targets our measured failure exactly.** The lattice's energy sits at discrete harmonics.
Dividing by `|F|^2ρ` suppresses precisely those frequencies and relatively amplifies the broadband,
low-energy aperiodic content — which is the mat boundary, the 10% of template area that carries all
the discriminating information and is currently diluted to a 0.02–0.05 contribution.

**Why it fits our process:** `ρ = 0` is a built-in null control that must reproduce production
bit-for-bit, exactly like `psf_sigma = 0` in the integrated change. It is ~10 lines of numpy, FFT
based, and composes with the PSF matching already in production rather than replacing it.

### P2 — Lattice-notch filtering

More surgical than general whitening: read the lattice harmonics directly off the template's own
autocorrelation/FFT (we already have pitch-detection code from `experiments/pitch_aware_prominence/`)
and notch exactly those frequencies out of both template and image. This deletes the uninformative
90% outright rather than attenuating it smoothly. Cheap, and a good A/B against P1 to learn whether
broad whitening or targeted notching is doing the work.

### P3 — Discriminability-weighted ZNCC

Implements our own measured diagnosis directly. With per-pixel weights `w`, `Σw = 1`:

```
μ_w(I) = Σ w·I
σ_w(I) = sqrt(Σ w·(I − μ_w)²)
ZNCC_w = Σ w·(T − μ_w(T))·(I − μ_w(I)) / (σ_w(T)·σ_w(I))
```

This stays FFT-fast — it needs correlations of `I` with `w`, `I` with `w·T`, and `I²` with `w`, so
four FFTs instead of two. Two ways to set the weights, both attacking confusers directly:

- **Lattice-shift dissimilarity:** `w(x) ∝ mean_δ |T(x) − T(x+δ)|²` over lattice vectors δ. Array
  pixels get ≈0 weight, boundary pixels get large weight.
- **Confuser-variance (no lattice estimation needed):** run plain ZNCC, take the top-K sidelobe
  windows, set `w(x) ∝ Var_k[I_k(x)]`. Pixels identical across all confusers get zero weight. This
  is Fisher-style discriminant weighting and is fully data-driven.

`w = uniform` reproduces current production exactly — again a built-in null control.

### P4 — PSR as the confidence metric

`PSR = (peak − μ_sidelobe) / σ_sidelobe`, excluding a small window around the peak. This is what our
own gap statistic is a crude version of, and it is better founded. We already showed the gap
statistic separates correct from wrong pairs by ~7x in the median and drives the integrated dual-arm
selection; PSR should do the same job better, and would also be the principled replacement for
`AMBIGUITY_THRESHOLD`, now recalibrated to `0.990` (it fires on 55/156 pairs, 35.3%, at 54.5%
precision; at the superseded `0.92` it fired on 128/156 at 31% precision). Cheap, and
worth doing regardless of which scoring change wins.

### P5 — DDIS as a re-ranker over the top-K candidates

Deformable Diversity Similarity was designed for the case where much of the template is
uninformative. Its diversity term counts how many template patches share a nearest neighbour and
penalizes concentration — at a true match most patches have unique nearest neighbours, while at a
wrong match they collapse onto a few points. That collapse **is** the periodic signature. Run it
only over our existing top-K peaks, never as a dense scan, and it is cheap.

### P6 — Frozen pretrained features as the score, no training

Our three from-scratch learned attempts all failed catastrophically (0.77 → ~0.39), including at
7.5x data, so training on our ~500 synthetic pairs is falsified. **Frozen** features sidestep that
entirely: score candidates by cosine similarity on DINOv3 features over the existing hypothesis
grid, with no training at all. Honest caveats: DINOv3 tokens are 16px, far too coarse for 5px
accuracy, so this can only ever be a coarse re-ranker over existing candidates, never the locator.

---

## 4. What the survey says *not* to do

- **General-purpose learned matchers (RoMa v2, LoMa, LoFTR family).** The out-of-distribution
  evidence is strongly negative. XoFTR measured off-the-shelf transfer to a new modality at
  2.8–7.3% AUC, and fine-tuning *without* domain-specific augmentation gave essentially no gain
  (2.92% vs 2.77%). The 2025 "Deep Learning Reforms Image Matching" survey lists repetitive patterns
  as an explicitly **unsolved** failure mode for learned methods. There is no benchmark evaluating
  learned matchers on periodic industrial texture — we would be the experiment.
- **Sparse learned keypoint pipelines (SuperPoint/LightGlue, DISK, XFeat).** Wrong family: keypoint
  detection on a periodic lattice produces thousands of mutually indistinguishable keypoints, which
  is our exact failure mode amplified. Our own `keypoint_candidate_fusion` experiment already found
  ORB proposals never once outscored the classical winner.
- **Relying on any matcher's scale invariance at 10x.** PRISM's scale-stratified benchmark shows the
  best method reaching only 27.4% AUC at ratios ≥4x, with LoFTR at 10.2%. Nobody tests 10x. We know
  our ratio to within 9:1–11:1 — we should keep solving scale ourselves rather than delegating it.
- **Phase correlation for the final translation.** Useful as a pre-stage for rotation/scale off the
  lattice peaks, but translation via phase correlation on a periodic image is ambiguous modulo the
  lattice by construction.
- **Local self-similarity descriptors.** Actively counterproductive here: a DRAM array has a strong,
  consistent self-similarity signature, so LSS would make the periodic region look *more* matchable.
- **Mutual information, ASIFT.** No evidence of relevance; ASIFT solves affine viewpoint invariance,
  which SEM (orthographic) does not have.

---

## 5. Two honest cautions

**Do not go purely gradient-based.** The Steger/PatMax argument is strong, but Hitachi's JP5639797B2
documents that edge-only matching on line/space arrays is itself vulnerable to **half-pitch**
errors — a shift by half a period looks identical in edge space. Any gradient-direction score we try
should carry an interior grey-level term alongside it. This is a specific, named failure mode we
would otherwise walk straight into.

**One external claim conflicts with our own data, and our data wins.** The survey suggests crops
with no unique structure may be information-theoretically under-determined. Our
`crop_uniqueness_ceiling` measurement says otherwise for almost all pairs: 154/156 crops have a
unique origin at a 0.95 content-identity threshold, and **zero** of our 40 failures chose a location
matching ground truth above 0.95. Only two pairs are genuine near-duplicates. So the recoverable
fraction here is much larger than the external framing assumes — which is good news, and a reminder
to trust our own measurements over general guidance.

---

## 6. Recommended sequence

1. **P1 (spectral prewhitening)** first — best theory-to-effort ratio, directly targets the measured
   mechanism, contains current production as `ρ = 0`.
2. **P4 (PSR)** alongside it — nearly free, improves confidence reporting regardless of outcome, and
   would replace a miscalibrated production threshold.
3. **P3 (weighted ZNCC)** if P1 shows the mechanism is real but under-exploited; P2 as the A/B to
   isolate whether targeted notching beats broad whitening.
4. **P5 (DDIS re-rank)** as a cheap independent check on the top-K.
5. **P6** only if P1–P5 stall, and only ever as a coarse re-ranker.

The important structural point: **P1–P3 and P5 all change the score itself, not the ranking of
existing ZNCC scores.** The nine rejected experiments in `experiments/ACCURACY_90_CAMPAIGN.md` all
re-ranked existing scores and all failed, and our own diagnosis predicted they must. These proposals
are in the class that diagnosis pointed to.

---

## Sources

Cognex PatMax patent [US7016539B1](https://patents.google.com/patent/US7016539B1/en) ·
Steger, *Similarity Measures for Occlusion, Clutter, and Illumination Invariant Object Recognition*
([DAGM 2001](https://mv.in.tum.de/_media/members/steger/publications/2001/dagm-2001-steger.pdf),
[ISPRS 2002](https://mv.in.tum.de/_media/members/steger/publications/2002/isprs-comm-iii-steger.pdf)) ·
KLA [US9830421B2](https://patents.google.com/patent/US9830421) ·
SEM addressing points [US8073242B2](https://patents.google.com/patent/US8073242B2/en) ·
Memory pattern matching [US20090103799A1](https://patents.google.com/patent/US20090103799A1/en) ·
Hitachi half-pitch [JP5639797B2](https://patents.google.com/patent/JP5639797B2/en) ·
Cognex sub-models [US6324299B1](https://patents.google.com/patent/US6324299B1/en) ·
Design-to-SEM alignment [US20240095935](https://patents.justia.com/patent/20240095935) ·
[MOSSE (Bolme et al., CVPR 2010)](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf) ·
[Kumar, correlation-filter survey](https://www.cis.rit.edu/~rlepci/Erho/Derek/Useful_References/Correlation%20Filtering/Kumar_COR_Survey99.pdf) ·
[MACE](https://pubmed.ncbi.nlm.nih.gov/20490115/) · [UMACE](https://opg.optica.org/ao/abstract.cfm?uri=ao-33-17-3751) ·
[DDIS](https://arxiv.org/abs/1612.02190) ([code](https://github.com/roimehrez/DDIS)) ·
[BBS](https://arxiv.org/abs/1609.01571) ·
[CoTM](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kat_Matching_Pixels_Using_CVPR_2018_paper.pdf) ·
[RoMa v2](https://arxiv.org/abs/2511.15706) · [LoMa](https://github.com/davnords/LoMa) ·
[XoFTR](https://arxiv.org/html/2404.09692v1) ·
[Deep Learning Reforms Image Matching (survey, 2025)](https://arxiv.org/html/2506.04619v1) ·
[PRISM (large scale ratios)](https://arxiv.org/html/2408.03598v1) ·
[DINOSim (DINOv2 zero-shot on EM)](https://www.biorxiv.org/content/10.1101/2025.03.09.642092v2.full)
