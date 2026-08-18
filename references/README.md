# References

Public sources supporting the semiconductor structures, image formation, noise and transformations
this project models. Deliverable 7 of the Applied Materials "Drift-Sense" problem statement.

Every entry below was checked against a publisher, indexing service or the author's own hosted copy
before being listed, and carries a DOI, a stable URL, or an ISBN. Nothing is listed that could not
be verified. `BIBLIOGRAPHY.bib` in this directory holds the same entries as BibTeX, for reuse in the
slide deck or any write-up.

## Honest statement of coverage

`reports/DEGRADATION_COVERAGE.md` audits 22 rows. Against that table:

| Tier | Meaning | Rows | Which |
|---|---|---:|---|
| A | A specific verified source (peer-reviewed paper, or a named textbook chapter) treats this exact mechanism | 14 | 1, 2, 3, 4, 7, 8, 9, 11, 12, 15, 17, 18, 19, 20 |
| B | Standard engineering model with no single canonical citation; anchored to the textbook chapter covering its general class | 6 | 5, 6, 10, 13, 14, 16 |
| C | Not a physical degradation — an acquisition protocol or a structural control knob, so no physics citation applies | 2 | 21, 22 |

So 14 of the 22 rows have a mechanism-specific citation. The remaining 8 are labelled as such in the
coverage table rather than dressed up with a loosely-related paper. Tier B rows are genuinely
standard: radial illumination falloff, display gamma, area-average decimation, multiplicative gain
noise, impulse noise and lithographic corner rounding are all textbook constructions that no single
paper owns. They are pointed at the chapter that covers them.

The matching method itself (normalised cross-correlation over a scale/rotation hypothesis grid, with
parabolic sub-pixel peak refinement) is separately cited in the "Matching method" section — that is
production code, not generator code, and the rubric's literature-justification weight applies to the
approach as much as to the synthetic data.

---

## 1. Semiconductor structure — DRAM array, mat and strip organisation

This is the project's central structural claim: a DRAM die is not one continuous periodic field. It
is an array of sub-arrays ("mats"), each surrounded by bands of peripheral circuitry — bitline sense
amplifiers along one axis, local/sub wordline drivers along the other — so a field of view looks
like tiled blocks separated by structurally different strips. `generator/macro_layout.py` renders
exactly that.

- **Keeth, B., Baker, R. J., Johnson, B., & Lin, F.** *DRAM Circuit Design: Fundamental and
  High-Speed Topics*, 2nd edition. Wiley-IEEE Press, IEEE Press Series on Microelectronic Systems,
  2007. ISBN 978-0-470-18475-2.
  <https://www.wiley.com/en-us/DRAM+Circuit+Design:+Fundamental+and+High+Speed+Topics+,+2nd+Edition-p-9780470184752>
  The standard text. Chapter 2 "The DRAM Array", Chapter 3 "Array Architectures" (open vs. folded
  bitline, cell area in F² units), Chapter 4 "The Peripheral Circuitry" (sense amplifiers, wordline
  drivers). Primary source for the folded-bitline cell array `generator/pattern_renderer.py` draws
  and for the mat/periphery split `generator/macro_layout.py` tiles.

- **Vogelsang, T.** "Understanding the Energy Consumption of Dynamic Random Access Memories."
  *2010 43rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-43)*, pp. 363–374,
  2010. DOI [10.1109/MICRO.2010.42](https://doi.org/10.1109/MICRO.2010.42).
  Open copy: <https://www.engineering.upenn.edu/~leebcc/teachdir/ece299_fall10/Vogelsang10_dram.pdf>
  States the tiling directly: each sub-array has "bitline sense-amplifiers and local wordline
  drivers surrounding it"; local wordlines and bitlines are "between 256 cells and 512 cells long",
  and master lines span "16 to 32 sub-arrays". Also gives the bank-level floorplan with row logic
  placed *between* banks and a centre stripe — the macro-scale reason a die image reads as blocks
  separated by bands. This is the single best free source to hand a judge for the mat/strip claim.

- **Kim, Y., Seshadri, V., Lee, D., Liu, J., & Mutlu, O.** "A case for exploiting subarray-level
  parallelism (SALP) in DRAM." *Proc. 39th International Symposium on Computer Architecture (ISCA)*
  / *ACM SIGARCH Computer Architecture News* 40(3), pp. 368–379, 2012.
  DOI [10.1145/2366231.2337202](https://doi.org/10.1145/2366231.2337202).
  Documents the bank → subarray → mat hierarchy and the local sense-amplifier rows between
  subarrays, from the architecture side. Useful as an independent corroboration of the same
  geometry described by Keeth & Baker from the circuit side.

- **Itoh, K.** "Trends in megabit DRAM circuit design." *IEEE Journal of Solid-State Circuits*
  25(3), pp. 778–789, 1990. DOI [10.1109/4.102676](https://doi.org/10.1109/4.102676).
  Classic JSSC survey of DRAM array partitioning, divided/hierarchical wordline schemes and
  sense-amplifier placement — the design pressures that produce the mat-and-strip floorplan in the
  first place.

- **Itoh, K.** *VLSI Memory Chip Design*. Springer Series in Advanced Microelectronics, vol. 5,
  Springer, 2001. DOI [10.1007/978-3-662-04478-0](https://doi.org/10.1007/978-3-662-04478-0),
  ISBN 978-3-540-67820-5. Chapter 3 "DRAM Circuits".
  Book-length treatment of the same material, for the array-partitioning and signal-to-noise
  reasoning behind sub-array sizing.

**Scope note.** These sources support the *organisation* being modelled — discrete mats separated by
peripheral bands, periodic wordline/bitline arrays inside a mat, contacts on a sub-lattice of their
intersections. They do not support the six specific pitch/CD presets in
`generator/mat_generator.py::DRAM_MAT_PRESETS`, nor the 2:3 word:bit pitch ratio; those are this
project's own parameterisation, chosen to span a plausible range, and are documented as a design
choice in `reports/V2_ARCHITECTURE_PLAN.md` rather than claimed as a measured value.

---

## 2. SEM image formation

- **Reimer, L.** *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*,
  2nd edition. Springer Series in Optical Sciences, vol. 45, Springer, 1998.
  DOI [10.1007/978-3-540-38967-5](https://doi.org/10.1007/978-3-540-38967-5),
  ISBN 978-3-540-63976-3.
  The standard text. Chapter 2 "Electron Optics of a Scanning Electron Microscope" (probe-forming
  optics, spot size, aberrations including astigmatism); Chapter 4 "Emission of Backscattered and
  Secondary Electrons" (secondary-electron yield and its dependence on surface tilt — the origin of
  edge brightening); Chapter 5 "Electron Detectors and Spectrometers"; Chapter 6 "Image Contrast and
  Signal Processing".

- **Goldstein, J. I., Newbury, D. E., Michael, J. R., Ritchie, N. W. M., Scott, J. H. J., & Joy,
  D. C.** *Scanning Electron Microscopy and X-Ray Microanalysis*, 4th edition. Springer, 2018.
  DOI [10.1007/978-1-4939-6676-9](https://doi.org/10.1007/978-1-4939-6676-9),
  ISBN 978-1-4939-6674-5.
  Chapter 3 "Secondary Electrons", Chapter 6 "Image Formation", Chapter 7 "SEM Image
  Interpretation", Chapter 8 "The Visibility of Features in SEM Images" (dose, signal-to-noise and
  feature detectability), Chapter 9 "Image Defects" (charging, drift, contamination, scan artifacts).
  Chapters 8 and 9 are the anchor for several of the Tier-B degradation rows.

- **Postek, M. T., & Vladár, A. E.** "Modeling for accurate dimensional scanning electron microscope
  metrology: then and now." *Scanning* 33(3), pp. 111–125, 2011.
  DOI [10.1002/sca.20238](https://doi.org/10.1002/sca.20238).
  Full text (US Government work, public domain, open):
  <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=908153>
  NIST review of why an SEM edge signal is not the edge: beam-sample interaction volume, edge
  effects, and why modelling the image-formation chain is required for dimensional accuracy. Also
  covers magnification and scan calibration, which is what the residual scale-drift term models.

- **Villarrubia, J. S., Vladár, A. E., Ming, B., Kline, R. J., Sunday, D. F., Chawla, J. S., &
  List, S.** "Scanning electron microscope measurement of width and shape of 10 nm patterned lines
  using a JMONSEL-modeled library." *Ultramicroscopy* 154, pp. 15–28, 2015.
  DOI [10.1016/j.ultramic.2015.01.004](https://doi.org/10.1016/j.ultramic.2015.01.004).
  Physics-based forward model of the CD-SEM signal from a patterned line, matched against measured
  images. The concrete demonstration that "structure → simulated SEM image" is a legitimate,
  validated modelling approach — which is what this project's generator is doing, at lower fidelity.

---

## 3. Noise

- **Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K.** "Practical Poissonian-Gaussian Noise
  Modeling and Fitting for Single-Image Raw-Data." *IEEE Transactions on Image Processing* 17(10),
  pp. 1737–1754, 2008.
  DOI [10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399).
  The signal-dependent Poisson (counting) component plus a signal-independent Gaussian (read)
  component, as a single composite model. This is exactly the two-term structure used in
  `generator/degradation_models.py`. Note the published title uses "Poissonian-Gaussian"; the repo
  previously cited it as "Poisson-Gaussian".

- **Timischl, F., Date, M., & Nemoto, S.** "A statistical model of signal–noise in scanning electron
  microscopy." *Scanning* 34(3), pp. 137–144, 2012 (published online 2011).
  DOI [10.1002/sca.20282](https://doi.org/10.1002/sca.20282).
  The SEM-specific counterpart: signal and noise statistics for a scanned electron probe and its
  detection chain. Cited alongside Foi et al. so the noise model is anchored to SEM specifically,
  not only to camera raw data.

- Feature visibility as a function of dose and signal-to-noise: **Goldstein et al. (2018),
  Chapter 8**, above.

---

## 4. Scan artifacts — drift, jitter, shear and scan nonlinearity

- **Sutton, M. A., Li, N., Garcia, D., Cornille, N., Orteu, J. J., McNeill, S. R., Schreier, H. W.,
  & Li, X.** "Metrology in a scanning electron microscope: theoretical developments and experimental
  validation." *Measurement Science and Technology* 17(10), pp. 2613–2622, 2006.
  DOI [10.1088/0957-0233/17/10/012](https://doi.org/10.1088/0957-0233/17/10/012).
  Separates the two error classes in a scanned image: a fixed **spatial distortion** (the scan raster
  is not a perfect Cartesian grid) and a time-varying **drift distortion**. That separation is the
  direct justification for modelling barrel/pincushion and shear/drift as distinct terms.

- **Sutton, M. A., Li, N., Joy, D. C., Reynolds, A. P., & Li, X.** "Scanning Electron Microscopy for
  Quantitative Small and Large Deformation Measurements Part I: SEM Imaging at Magnifications from
  200 to 10,000." *Experimental Mechanics* 47(6), pp. 775–787, 2007.
  DOI [10.1007/s11340-007-9042-z](https://doi.org/10.1007/s11340-007-9042-z).
  Quantifies both distortion classes across a magnification range, including the progressive,
  time-correlated character of drift within a single slow raster acquisition — which is why
  `apply_raster_shear_drift` uses a row-index-dependent shear plus per-row jitter rather than a
  single global translation.

- **Jin, P., & Li, X.** "Correction of image drift and distortion in a scanning electron
  microscopy." *Journal of Microscopy* 260(3), pp. 268–280, 2015.
  DOI [10.1111/jmi.12293](https://doi.org/10.1111/jmi.12293).
  Drift corrected by inter-image correlation over time, and distortion derived from charged-particle
  imaging theory using images at several magnifications. Establishes that the radial scan
  nonlinearity is magnification-dependent — the reason `barrel_k` is applied at full strength to the
  wide-field Search image and at reduced strength to the Reference.

- **Cazaux, J.** "Charging in scanning electron microscopy 'from inside and outside'." *Scanning*
  26(4), pp. 181–203, 2004.
  DOI [10.1002/sca.4950260406](https://doi.org/10.1002/sca.4950260406).
  Charge build-up on insulating regions, its time dependence, and the resulting image artifacts.
  Supports charging being modelled as a transient, row-correlated brightness excursion along the
  slow scan axis, and being physically tied to insulating (strip/routing) material rather than
  conductive lines.

- Charging, drift and scan-related artifacts also appear as a catalogue in **Goldstein et al.
  (2018), Chapter 9 "Image Defects"**, above.

---

## 5. Lithography and etch effects

- **Orji, N. G., Badaroglu, M., Barnes, B. M., Beitia, C., Bunday, B. D., Celano, U., Kline, R. J.,
  Neisser, M., Obeng, Y., & Vladár, A. E.** "Metrology for the next generation of semiconductor
  devices." *Nature Electronics* 1(10), pp. 532–547, 2018.
  DOI [10.1038/s41928-018-0150-9](https://doi.org/10.1038/s41928-018-0150-9).
  NIST-led review covering critical-dimension measurement, linewidth/line-edge roughness, pattern
  roughness and local size variation, and overlay/placement metrology. Supports both the per-line
  width jitter (roughness around nominal CD) and the cumulative per-line placement drift.

- **Tanaka, T., Morigami, M., & Atoda, N.** "Mechanism of Resist Pattern Collapse during Development
  Process." *Japanese Journal of Applied Physics* 32(12S), p. 6059, 1993.
  DOI [10.1143/JJAP.32.6059](https://doi.org/10.1143/JJAP.32.6059).
  The original capillary-force analysis of pattern collapse: unbalanced surface tension of the rinse liquid
  in the gap between adjacent resist lines deflects them until they touch. This is the physical basis
  for `pattern_renderer.py::bridge_narrow_gaps` bridging only *interior* gaps below a threshold —
  an edge line has no neighbour to be pulled toward.

- **Mack, C. A.** "Pattern Collapse." *The Lithography Expert*, November 2006.
  <https://www.lithoguru.com/scientist/litho_tutor/Tutor55%20(Nov%2006).pdf>
  Short, freely readable derivation of the same effect: capillary pressure equals the liquid surface
  tension divided by the meniscus radius of curvature, and collapse tendency scales roughly with the
  cube of the line aspect ratio, with narrower spaces collapsing at lower aspect ratios. Gives
  worked numbers (a 100 nm space tolerating aspect ratio 4.3 versus 3.4 at a 50 nm space) that
  justify treating collapse as a *spacing-threshold* effect, which is how the generator parameterises
  it (`collapse_threshold_nm`).

- **Mack, C. A.** *Fundamental Principles of Optical Lithography: The Science of Microfabrication*.
  John Wiley & Sons, 2007. ISBN 978-0-470-01893-4.
  <https://www.wiley.com/en-us/Fundamental+Principles+of+Optical+Lithography:+The+Science+of+Microfabrication-p-9780470018934>
  General reference for the printed-feature effects the generator applies at render time: systematic
  CD bias from exposure dose and etch, corner rounding and line-end shortening from the finite
  bandwidth of the imaging system, and CD control in general.

---

## 6. Matching method (production pipeline)

- **Lewis, J. P.** "Fast Normalized Cross-Correlation." *Vision Interface*, pp. 120–123, 1995.
  Author's hosted copy: <https://scribblethink.org/Work/nvisionInterface/nip.pdf>
  The canonical reference for normalised cross-correlation as a template-matching score and for
  computing it efficiently (running sums for the local mean/energy, FFT for the numerator). No DOI —
  *Vision Interface* proceedings predate DOI assignment; the author's own copy is the stable link.
  `pipeline/matching.py::correlate` is this score.

- **Briechle, K., & Hanebeck, U. D.** "Template matching using fast normalized cross correlation."
  *Proc. SPIE 4387, Optical Pattern Recognition XII*, pp. 95–102, 2001.
  DOI [10.1117/12.421129](https://doi.org/10.1117/12.421129).
  Peer-reviewed, DOI-bearing companion to Lewis, for judges who want a citable venue.

- **Steger, C.** "Similarity Measures for Occlusion, Clutter, and Illumination Invariant Object
  Recognition." *Pattern Recognition (DAGM 2001)*, Lecture Notes in Computer Science vol. 2191,
  pp. 148–154, Springer, 2001.
  DOI [10.1007/3-540-45404-7_20](https://doi.org/10.1007/3-540-45404-7_20).
  The shape-based-matching similarity measure behind industrial template matching, and the standard
  argument for evaluating a template over an explicit grid of pose (here: scale and rotation)
  hypotheses rather than assuming an identity transform. `pipeline/candidate_generation.py` enumerates
  such a grid.

- **Gonzalez, R. C., & Woods, R. E.** *Digital Image Processing*, 4th edition. Pearson, 2018.
  ISBN 978-0-13-335672-4.
  <https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003224>
  Textbook anchor for the general-purpose image-processing constructions used in both the generator
  and the pipeline: sampling and anti-aliased decimation, intensity (gamma) transformations, and the
  impulse / additive / multiplicative noise models.

- **OpenCV**, `matchTemplate` and the `TM_CCOEFF_NORMED` method definition.
  <https://docs.opencv.org/5.x/df/dfb/group__imgproc__object.html>
  (5.x, matching the pinned `opencv-python-headless==5.0.0.93`.)
  Software reference, not a citation. Included because `pipeline/matching.py` relies on
  `TM_CCOEFF_NORMED` being exactly zero-mean normalised cross-correlation, and that identity is what
  the documented formula states.

- **Applied Materials, *Drift-Sense Synthetic Data* (Hugging Face)** — the hackathon's provided
  starter kit, not an independent public source. Listed here only for provenance: its degradation
  code was read as a coverage reference and never imported, copied or executed, and only its
  already-generated output images are used, as an external evaluation surface (see the README's
  "Dataset" section and `reports/DEGRADATION_COVERAGE.md`).

---

## Mechanism → code → source map

Row numbers match `reports/DEGRADATION_COVERAGE.md`. Tier as defined at the top of this file.

| # | Mechanism | Where implemented | Source | Tier |
|---|---|---|---|---|
| 1 | Gaussian PSF blur (finite beam spot) | `generator/degradation_models.py::gaussian_psf_blur` | Reimer (1998) ch. 2; Postek & Vladár (2011); Villarrubia et al. (2015) | A |
| 2 | Astigmatism (axis-locked elliptical blur) | `generator/degradation_models.py::gaussian_psf_blur` (`astigmatism_ratio`) | Reimer (1998) ch. 2 — probe-forming optics and aberrations | A |
| 3 | Poisson shot noise | `generator/degradation_models.py::poisson_shot_noise` | Foi et al. (2008); Timischl et al. (2012); Goldstein et al. (2018) ch. 8 | A |
| 4 | Gaussian read/detector noise | `generator/degradation_models.py::gaussian_read_noise` | Foi et al. (2008) — the Gaussian term of the same composite model; Timischl et al. (2012) | A |
| 5 | Vignette (radial falloff) | `generator/degradation_models.py::apply_vignette` | Standard radial collection-efficiency model. Anchor: Goldstein et al. (2018) ch. 9; Reimer (1998) ch. 5 | B |
| 6 | Gamma (detector nonlinearity) | `generator/degradation_models.py::apply_gamma` | Standard power-law intensity transform. Anchor: Reimer (1998) ch. 6; Gonzalez & Woods (2018) | B |
| 7 | Raster shear + drift/jitter | `generator/degradation_models.py::apply_raster_shear_drift` | Sutton et al. (2006); Sutton et al. (2007) — drift distortion, time-correlated within one raster | A |
| 8 | Residual rotation drift | `generator/degradation_models.py::apply_rotation_scale` | Sutton et al. (2007); Jin & Li (2015) | A |
| 9 | Residual scale drift | `generator/degradation_models.py::apply_rotation_scale` | Postek & Vladár (2011) — magnification/scan calibration; Sutton et al. (2006) | A |
| 10 | Exact 10x area-average downsample | `generator/degradation_models.py::downsample_area_average` | Standard anti-aliased decimation; also the problem statement's fixed 10:1 FOV ratio. Anchor: Gonzalez & Woods (2018) | B |
| 11 | Structural pattern collapse | `generator/pattern_renderer.py::bridge_narrow_gaps` | Tanaka et al. (1993); Mack (2006) | A |
| 12 | Charging streaks | `generator/degradation_models.py::apply_charging_streaks` | Cazaux (2004); Goldstein et al. (2018) ch. 9 | A |
| 13 | Speckle (multiplicative gain noise) | `generator/degradation_models.py::apply_speckle_noise` | Standard multiplicative-noise model. Anchor: Timischl et al. (2012) for SEM signal-noise statistics; Gonzalez & Woods (2018) | B |
| 14 | Salt-and-pepper (impulse noise) | `generator/degradation_models.py::apply_salt_and_pepper` | Standard impulse-noise model. Anchor: Gonzalez & Woods (2018) | B |
| 15 | Barrel/pincushion scan nonlinearity | `generator/degradation_models.py::apply_barrel_distortion` | Sutton et al. (2006) — spatial distortion; Jin & Li (2015) — magnification-dependent distortion | A |
| 16 | Corner rounding | `generator/pattern_renderer.py::_round_corners` | Standard litho/etch finite-bandwidth effect. Anchor: Mack (2007) | B |
| 17 | Linewidth / CD bias | `generator/pattern_renderer.py::_rasterize_lines` (`linewidth_bias_nm`) | Mack (2007) — dose and etch bias on printed CD; Orji et al. (2018) — CD metrology | A |
| 18 | Per-line position jitter (cumulative walk) | `generator/pattern_renderer.py::jittered_line_positions` | Orji et al. (2018) — overlay and pattern-placement metrology | A |
| 19 | Per-line width jitter (LER/LWR) | `generator/pattern_renderer.py::_rasterize_lines` (`WIDTH_JITTER_FRACTION`) | Orji et al. (2018) — linewidth roughness and local size variation; Villarrubia et al. (2015) | A |
| 20 | Macro mat/strip zone composition | `generator/macro_layout.py::generate_macro_canvas`, `generator/mat_generator.py` | Keeth et al. (2007) ch. 2–4; Vogelsang (2010); Kim et al. (2012); Itoh (1990); Itoh (2001) ch. 3 | A |
| 21 | Acquisition variants (1 Reference + N Search) | `generator/dataset_generator.py::generate_acquisition_variant_set` | Evaluation protocol, not a physical mechanism — no physics citation applies | C |
| 22 | Feature-size continuous scaling | `generator/mat_generator.py::generate_mat` (`feature_size_scale`) | Structural control knob spanning process nodes, not a degradation — no physics citation applies | C |

### Matching method

| Component | Where implemented | Source |
|---|---|---|
| Zero-mean normalised cross-correlation score | `pipeline/matching.py::correlate` (`cv2.TM_CCOEFF_NORMED`) | Lewis (1995); Briechle & Hanebeck (2001); OpenCV `matchTemplate` docs |
| Explicit scale × rotation hypothesis grid | `pipeline/candidate_generation.py::build_candidate_pool` | Steger (2001) — pose-hypothesis search for invariant matching |
| Greedy non-maximum suppression over the score map | `pipeline/matching.py::top_k_peaks` | Standard NMS; no single canonical citation |
| Template resampling into the Search passband | `pipeline/matching.py::build_template` (`psf_sigma`) | Reimer (1998) ch. 2 for the PSF being matched; the specific matched-blur choice is this project's own, evidenced in `experiments/psf_matched_template/` |
| Sub-pixel refinement by parabolic peak fit | `pipeline/refinement.py::refine` | Standard three-point quadratic peak interpolation; no single canonical citation |
| Centre tie-break when candidates are tied | `pipeline/ranking.py::apply_center_tiebreak` | Applied Materials problem-statement rule, not a literature result — see `reports/TIE_BREAK_IMPLEMENTATION.md` |

---

## Verification note

Each DOI above resolves through Crossref or the publisher; each ISBN was checked against the
publisher's own catalogue page; each bare URL was fetched. Sources that could not be verified were
dropped rather than listed, and the dropped sources were not enumerated anywhere — there is no
exclusion list to consult.
