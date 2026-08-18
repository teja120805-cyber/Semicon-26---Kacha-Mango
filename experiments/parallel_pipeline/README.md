# experiments/parallel_pipeline

A sandbox for **alternative scoring functions**, kept separate from the incremental
`experiments/<one-idea>/` folders because these share a harness and are meant to be A/B'd against
each other on the same candidate pool.

Motivation: `reports/RESEARCH_SURVEY_SCORING.md`. Every one of the nine rejected experiments in
`experiments/ACCURACY_90_CAMPAIGN.md` re-ranked *existing ZNCC scores*, and our own diagnostics
predicted they had to fail — the discriminating information is destroyed before ranking. The
proposals here change **the score itself**.

## Design

`spectral.py` and friends provide interchangeable scoring backends. `harness.py` runs the
unmodified production pipeline (`build_candidate_pool` structure, `rank_classical`,
`apply_center_tiebreak`, parabolic refinement) with one of them substituted for raw ZNCC. Every
backend has a parameter setting that provably reduces to current production, so each sweep carries
its own null control.

Production code (`pipeline/`, `generator/`, `model/`) is imported unmodified and never written to.

## Backends

| id | idea | status |
|---|---|---|
| `zncc` | production baseline (null control) | reference |
| `prewhiten` | P1 — spectral prewhitening, `1/(\|F\|²+λ)^ρ` | implemented — **REJECTED** |
| `notch` | P2 — targeted lattice-harmonic notching | implemented — **REJECTED** |

`psr` (peak-to-sidelobe ratio) is computed alongside every backend as a candidate confidence
metric — P4 in the survey — rather than being a backend itself.

## Discipline

Same as every prior experiment in this project: sweep on the 24-pair `development` split only,
choose one configuration, run the frozen 156-pair benchmark once, then validate on the
independently-seeded dataset before proposing anything for integration. Null control must reproduce
production bit-for-bit, verified per-pair rather than assumed.
