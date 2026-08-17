# experiments/anisotropic_psf — anisotropic template PSF: NOT SUPPORTED (dev gain did not reproduce)

**Date:** 2026-08-16. **Not integrated. Production stays at 77.6%@5px.** Frozen benchmark: **0 runs**.
`validate_fresh`: **0 runs** — deliberately not spent, see §5.

## Summary

A mechanistically-derived change that looked strong on `development` (+8.3pp, 2 rescued / 0 broken)
and produced **exactly zero effect** on an independent 40-pair surface (0 rescued, 0 broken, at
three separate settings). Combined evidence is 2 rescued / 0 broken across 64 pairs, sign test
**p = 0.25** — nowhere near the bar the integrated PSF change cleared (p = 0.031).

The honest reading: the development result was noise from an 11-configuration sweep on 24 pairs.
It is reported anyway, with the sweep size disclosed, because the mechanism is sound and the
arithmetic in §1 is reusable.

## 1. The mechanism, read off the generator (this part stands regardless)

`generator/degradation_models.py::image_search` composes the Search image as blur → **10× area-average
downsample** → optional rotation/scale → `apply_raster_shear_drift` → noise → … . The raster drift
therefore runs at **Search resolution**, after downsampling, and displaces each row *horizontally*:

```
shift(row) = shear_amplitude_px * (row/(h-1) - 0.5) + N(0, jitter_std_px)
```

`DEFAULT_PARAMS` sets `shear_amplitude_px = 1.0`, `jitter_std_px = 0.4` — applied to **every pair**,
not just a special family.

Two conclusions follow, and the first is the more useful:

**The deterministic shear is negligible and a shear hypothesis grid would be chasing nothing.**
1.0px across 1000 rows is 0.1px across a 100px template. `reports/PROJECT_STATUS.md` lists "the
raster shear/jitter the template never models" as a remaining fidelity gap, and "add shear
hypotheses to the grid" is the obvious next move by analogy with `finer_hypothesis_grid` and
`scale_range_v1` (both integrated, both wins). **The arithmetic says don't** — the effect is an
order of magnitude below the 5px tolerance. This saves a plausible-looking experiment.

**The per-row jitter is not negligible, and is horizontal-only.** ~0.4px random displacement per
row, which no global affine hypothesis can model, but whose effect on correlation approximates an
extra *horizontal* blur. `pipeline/matching.build_template` blurs isotropically
(`cv2.GaussianBlur(t, (0,0), psf_sigma)` sets sigmaY = sigmaX), so the effective Search PSF is wider
horizontally than vertically and the template cannot match both axes at once.

Hence: blur the template with separate (σx, σy), σx > σy. Cost is identical to production — same
single blur, different kernel — so unlike a wider hypothesis grid this is **free at runtime**
(measured 4.87–5.37 s/pair against a 4.8–4.9 s/pair baseline, i.e. within noise).

## 2. Null control

σx = σy = `PSF_MATCH_SIGMA` (1.6) must reproduce `pipeline.localize.localize` exactly. Verified on
6 pairs per surface, on x, y and confidence: **0 mismatches**. The baseline row of each sweep also
reproduces the true production accuracy exactly — 0.7083 on `development`, 0.7000 on
`tune_degraded` — so the comparison is against real production behaviour.

## 3. Result on `development` (24 pairs, 11 configurations)

| σx | σy | ratio | acc@5px | Δ | rescued | broken | net |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **2.0** | **1.6** | 1.250 | **0.7917** | **+8.3pp** | 2 | 0 | **+2** |
| 1.8 | 1.6 | 1.125 | 0.7500 | +4.2pp | 1 | 0 | +1 |
| 2.2 | 1.6 | 1.375 | 0.7500 | +4.2pp | 2 | 1 | +1 |
| 1.6 | 1.4 | 1.143 | 0.7083 | 0.0pp | 1 | 1 | 0 |
| *all 7 configs with σy ≤ 1.4* | | | 0.6667 | −4.2pp | 1 | 2 | **−1** |

The structure looked *right*, which is exactly why it was worth checking rather than believing:
single-peaked in σx at fixed σy = 1.6, monotone up to the peak (net 0 → +1 → +2 → +1), and every
reduction of σy uniformly harmful. That matches the prediction — the fix should be *added
horizontal* blur, not *reduced vertical* blur.

## 4. Result on `tune_degraded` (40 pairs, the three σy = 1.6 settings)

| σx | σy | acc@5px | Δ | rescued | broken | net |
|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 1.6 | 0.7000 | 0.0pp | **0** | **0** | **0** |
| 2.0 | 1.6 | 0.7000 | 0.0pp | **0** | **0** | **0** |
| 2.2 | 1.6 | 0.7000 | 0.0pp | **0** | **0** | **0** |

**Not one pair changed outcome at any setting.** Baseline 0.7000 (28/40) is reproduced exactly.

A mechanistic note worth keeping: the anisotropic arm *wins* the decisiveness comparison on 24 of
40 pairs — it is being selected — and still changes no outcome. So the template blur's effect on
**how decisive the pool looks** is much stronger than its effect on **where the peak is**. Any
future template-shaping idea should be judged on peak location, not on score or decisiveness
improvements, which move easily and mean little.

## 5. Verdict, and a run not spent

**NOT SUPPORTED.** Combined: 2 rescued, 0 broken across 64 pairs — sign test p = 0.25.

`validate_fresh` was **not** run. There is nothing left to validate: `tune_degraded` already served
as the independent check and returned an exact tie at every setting. Spending the held-back surface
now would be drawing again at the same lottery until a favourable number appears, which is the
benchmark-mining failure the project's conventions exist to prevent. It stays unspent.

**Worth noting in the change's favour, honestly:** across 64 pairs it broke **zero** pairs while
rescuing two. It is not harmful — it is unproven. If a larger tuning surface is ever built (the
`validation`-expansion item in `PROJECT_STATUS.md`), σx = 2.0 / σy = 1.6 is cheap to re-test and
costs nothing at runtime.

**Reusable outputs regardless of the verdict:**
- Shear hypotheses are ruled out by arithmetic (§1) before anyone spends a week on them.
- The decisiveness-vs-peak-location asymmetry (§4) is a general caution for template work.

## 6. Run-count disclosure

- Frozen 156-pair benchmark: **0 runs.**
- `development` (24): 1 baseline + 11 configs. **This is the overfitting risk**, and the +8.3pp
  headline came from it. Disclosed rather than presented as a finding.
- `tune_degraded` (40): 1 baseline + 3 configs, chosen before the run from the σy = 1.6 row only.
- `validate_fresh` (40): **0 runs**, deliberately.

## Reproduce

```
python -m experiments.anisotropic_psf.run --surface development
python -m experiments.anisotropic_psf.run --surface tune_degraded --sigma-x 1.8,2.0,2.2 --sigma-y 1.6
```
