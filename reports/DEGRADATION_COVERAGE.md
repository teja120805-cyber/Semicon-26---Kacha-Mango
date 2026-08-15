# DriftSense V2 — Degradation Coverage Audit

This audits every degradation mechanism V2's generator implements against the Applied Materials /
Hugging Face reference generator (the hackathon's official starter resource) and the published SEM
imaging literature, and states the physical motivation and source for each.

Methodology: read the reference generator's imaging/degradation code (`sem_imaging.py`,
`structural_defects.py`, `pipeline.py`/`GenerationParams`) directly as a read-only reference — its
code is never imported, copied, or executed by this project; only its already-generated output
images are used, and only as an external evaluation surface (see README section 3).

## Coverage table

| # | Degradation | In reference generator? | V2 implementation | Parameters (V2) | Physical motivation | Citation/source |
|---|---|---|---|---|---|---|
| 1 | Gaussian PSF blur (beam spot) | Yes | Reimplemented independently | `blur_sigma_ref_px`, `blur_search_effective_px` | Finite electron-beam spot size | Standard SEM image-formation model |
| 2 | Astigmatism (axis-locked elliptical blur) | Yes | Reimplemented independently | `astigmatism_ratio` (off by default) | Beam not perfectly round — sharper on one scan axis | Kept axis-locked to the scan axes; arbitrary-orientation astigmatism has no clear physical motivation for a raster-scanned instrument |
| 3 | Poisson shot noise | Yes | Reimplemented independently | `dose_reference`, `dose_search` | Electron-counting statistics dominate SEM noise | Foi, Trimeche, Katkovnik & Egiazarian (2008), "Practical Poisson-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data" |
| 4 | Gaussian read/detector noise | Yes | Reimplemented independently | `read_noise_sigma_ref`, `read_noise_sigma_search` | Amplifier/detector electronic noise floor | Same source as #3 (Poisson-Gaussian composite model) |
| 5 | Vignette | Yes | Implemented and exercised (`ho_vignette_gamma`) | `vignette_strength` | Radial illumination/collection-efficiency falloff | Standard optical/detector falloff model |
| 6 | Gamma | Yes | Implemented and exercised (`ho_vignette_gamma`) | `gamma` | Detector-gain nonlinearity | Standard nonlinear detector response model |
| 7 | Raster shear + drift/jitter | Yes | Reimplemented independently | `shear_amplitude_px`, `jitter_std_px` | Progressive row-to-row scan drift + finite scan-stabilization bandwidth | Standard raster-scan acquisition artifact |
| 8 | Residual rotation drift | No | Implemented, applied post-downsample | `rotation_deg`, sampled per-family from `_rotation_range` | Stage/scan rotation calibration drift, layered on a fixed base magnification | Modeled as a small residual effect on top of a fixed 10x base, not part of the reference generator |
| 9 | Residual scale drift | No | Implemented, applied post-downsample | `extra_scale`, sampled per-family from `_scale_range` | Residual magnification-calibration error, layered on a fixed base magnification (the magnification itself is not randomized — see `reports/V2_ARCHITECTURE_PLAN.md` section 4) | Deliberate design choice: hardware magnification is fixed, calibration drift is a separate, smaller effect |
| 10 | Exact 10x area-average downsample | Yes | Reimplemented independently | `SCALE_FACTOR = 10` (fixed constant) | Physical 10:1 magnification ratio between Reference and Search fields of view | Matches the reference generator's convention |
| 11 | Structural pattern-collapse | Yes | Implemented directly in the line-mask renderer (`generator/pattern_renderer.py::bridge_narrow_gaps`), called unconditionally wherever collapse is enabled, covered by a directional unit test (interior gaps bridge, edge gaps never do) | `collapse_enabled`, `collapse_threshold_nm`, `collapse_prob` (on by default) | Capillary/etch-induced bridging between adjacent lines below a process-dependent spacing threshold | Same physical basis as the reference generator's `structural_defects.py`; V2's implementation is an independent, from-scratch re-derivation (run-length-encoding based), not a copy |
| 12 | Charging streaks | Yes | Added | `charging_prob`, `charging_intensity` | Local sample charging on insulating regions → transient bright streaks along the slow scan axis | Present in the reference generator (`sem_imaging.py`) |
| 13 | Speckle (multiplicative noise) | Yes | Added | `speckle_sigma` | Detector-gain variation, multiplicative rather than additive | Present in the reference generator; standard multiplicative-noise model |
| 14 | Salt-and-pepper (impulse noise) | Yes | Added | `salt_pepper_amount` | Dead/hot detector pixels, discrete discharge events | Present in the reference generator; standard impulse-noise model |
| 15 | Barrel/pincushion distortion | Yes | Added | `barrel_k` | Imperfect beam-scan linearity / lens calibration, radial | Present in the reference generator; a 10x-larger Search FOV is exactly where this effect should be most visible |
| 16 | Corner rounding | Yes | Added | `corner_rounding_px` | Lithography/etch never produces perfectly sharp corners | Present in the reference generator (`dram.py` morphological open+close); implemented in V2 as a structural (generation-time), not acquisition-time, effect — matches where it physically belongs |
| 17 | Linewidth/CD bias | Yes | Implemented and exercised, with a regression test guarding against an unexercised parameter | `linewidth_bias_nm` (exercised by `val_linewidth_bias`) | Deterministic global over/under-exposure or etch bias, on top of per-line random jitter | Present in the reference generator |
| 18 | Per-line position jitter | Yes | Reimplemented independently | `POSITION_JITTER_NM` (cumulative random walk) | Real overlay/placement drift accumulates along a scan/exposure field | Orji et al. (2018), "Metrology for the next generation of semiconductor devices", *Nature Electronics* |
| 19 | Per-line/contact width jitter | Yes | Reimplemented independently | `WIDTH_JITTER_FRACTION` | Critical-dimension variation / line-edge roughness | Standard lithography/etch CD-variation model |
| 20 | Macro mat/strip zone composition | Yes | Reimplemented from scratch | `mat_size_nm`, `strip_width_nm` | Real chips are discrete sub-array mats + peripheral/routing strips, not one continuous field | The core structural design choice V2 is built around — see `reports/V2_ARCHITECTURE_PLAN.md` section 1 |
| 21 | Acquisition variants (1 Reference + N Search re-acquisitions) | Yes | Added (bonus, not part of the main 4-split benchmark) | `generate_acquisition_variant_set`, 5 named variants | Tests "does the model find the same place under different acquisition conditions" — a different, complementary axis to single-acquisition accuracy | Present in the reference generator (`generate_sample_family`) |
| 22 | Feature-size continuous scaling | N/A (structural control, not a degradation) | Added 2026-08-15 | `feature_size_scale` (`generator/mat_generator.py::generate_mat`, multiplies word/bit pitch and feature width proportionally; propagates into `periodicity_score` via `dataset_generator._touched_word_pitches`) | Lets one preset stand in for a continuum of process nodes instead of only the 6 discrete presets | Present in the reference/demo generator as its "Feature size scale" slider (0.5-2.0x); found during the reference-UI comparison in `reports/DATASET_AUDIT.md` |

## Deliberately not added

None. Every degradation mechanism identified in the reference generator is implemented in V2, at
minimum as an available, off-by-default, seed-controlled, individually-toggleable mechanism. Every
implemented mechanism is exercised by at least one shipped structural family, enforced by an automated
test (`generator/test_gt_safety.py::test_every_optional_degradation_is_exercised`) so nothing in this
table can quietly become dead weight.

## GT-leakage safety

None of the above mechanisms take a ground-truth-shaped parameter, and ground truth is computed from
the crop origin strictly before any of them run (`generator/dataset_generator.py::generate_pair`).
Verified automatically by `generator/test_gt_safety.py` (static parameter-name scan + dynamic
GT-box-matches-downsampled-Reference check) — see `reports/V2_ARCHITECTURE_PLAN.md` section 6.
