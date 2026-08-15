# Experiment B: learned candidate generator

**Status: REJECT — not integrated.** The learned model's dense-grid proposals almost never land
near the true location (3.0% candidate recall@5px, vs. classical's 82.6%), and consequently the
classical-union-learned hybrid is **bit-for-bit identical** to classical alone on every metric — the
learned candidates never won a single ranking decision, on any of the 132 gate-relevant pairs.

## Methodology

This is a candidate **proposer**, not a re-ranker (unlike the already-rejected
`embedding_reranker_v1`, which only re-scored the classical pipeline's own top-12 shortlist): a
shared-weight embedding network scores a **dense grid** of ~3,600 locations spanning the whole
search image (every 15px, `experiments/learned_candidate_generator/propose_candidates.py`), so it
can in principle surface a location the classical ZNCC search never considered competitive at all —
directly targeting the candidate-generation failure mode (31/156 failures on the frozen benchmark,
`reports/ACCURACY_FORENSICS.md`).

**Model**: `model/architecture.py::EmbeddingNet`, reused unmodified (same 64,992-param CNN
`embedding_reranker_v1` used) — this experiment tests whether more/more-diverse training data changes
the outcome, not whether a different architecture does, per the brief's "start with the smallest
defensible model."

## Feasibility/data audit (before training anything)

Production `data/development` (24 pairs, 3 families: `dev_strip_anchor`, `dev_single_mat`,
`dev_dense_periodic`) is the direct, previously-diagnosed cause of `embedding_reranker_v1`'s
72-triplet overfitting. Per `reports/DATASET_AUDIT.md` section 5's bounded recommendation, generated
a targeted expansion — **16 new families x 10 pairs = 160 pairs, seed `913700`** (distinct from
production `777001` and every other experiment's seed), covering the failure modes identified in
forensics: all 6 DRAM presets individually (periodicity), 4 boundary-proximity crop modes, 3
rotation/scale-interaction combinations, and 4 noise/distortion combinations. Combined with the
existing 24 production pairs: **184 total training pairs across 19 families** (`generate_training_data.py`,
isolated entirely under `experiments/learned_candidate_generator/dev_data/` — never touches
`data/development/`).

**Leakage check** (explicit, as required): verified 184/184 distinct signatures internally, **zero**
signature overlap with `validation`/`held_out`/`challenge`, and exactly the expected 24-pair overlap
with production `data/development` (the intentionally-copied existing pairs, not a leak).
`validation`/`held_out`/`challenge`/`cross_generator` were never read by any training code.

**Internal train/early-stop split**: 4 of the 19 families held out from gradient updates
(`dev2_single_mat_compact`, `dev2_mat_boundary`, `dev2_combined_drift`, `dev2_speckle_saltpepper` —
one per major axis) for early stopping, mirroring `model/train.py`'s existing pattern but spanning
more axes. 426 training triplets, 120 early-stop triplets (vs. 72 total before — ~5.9x more).

## Training

Best epoch 11 (of 21 before early stopping), early-stop loss 0.207. Training loss still collapses
quickly (0.0037 by epoch 12) while early-stop loss plateaus (~0.21-0.26) — the same qualitative
overfitting signature as before, just at a larger data scale. The real test is downstream evaluation,
not the training curve shape.

## Evaluation: classical vs. learned vs. hybrid (frozen benchmark, n=132)

| | Acc@5px | Median | Mean | Max | >50px |
|---|---:|---:|---:|---:|---:|
| Classical (this eval, matches production) | 70.5% | 0.33px | 56.9px | 900.8px | 17.4% |
| Learned-only | **0.8%** | 311.4px | 349.5px | 977.5px | 87.9% |
| Hybrid (classical UNION learned) | 70.5% | 0.33px | 56.9px | 900.8px | 17.4% |

**Hybrid == Classical on every single metric.** Rescue/break vs. production classical: hybrid
0/0/net 0 — literally zero pairs changed prediction.

### Candidate recall

| Tolerance | Classical | Learned | Hybrid |
|---|---:|---:|---:|
| @1px | 78.0% | 0.0% | 78.0% |
| @2px | 81.1% | 0.8% | 81.1% |
| @5px | 82.6% | **3.0%** | 83.3% |
| @10px | 83.3% | 6.1% | 84.1% |

Learned top-K recall@5px: K=5 -> 2.3%, K=10 -> 2.3%, K=20 -> 3.0%, K=40 -> 3.0% (saturates almost
immediately — going from top-5 to top-40 barely helps, meaning the true location isn't merely
ranked low in the learned scoring, it's essentially **absent** from the model's notion of
similarity at almost every one of the 3,600 grid points).

## Why this failed (not just that it did)

3.0% recall is not literally random (a back-of-envelope uniform-random baseline for 20 samples over
a ~900x900px valid region landing within 5px of one specific point is on the order of 0.2%), so the
model learned *something* non-trivial — just roughly an order of magnitude short of anywhere near
useful. Two contributing factors, stated as hypotheses rather than proven mechanisms:

1. **Training/deployment mismatch**: the model was trained with hard negatives mined from the
   *classical candidate pool* (~20-50 locations/pair, the plausible-looking decoys), but deployed
   against a dense grid of ~3,600 locations spanning the whole image — a far larger space of
   potential false positives the model never learned to reject during training.
2. **Consistent with Experiment C's information-ceiling finding**: `experiments/spatial_context/`
   found no reliable classical signal separating true sites from decoys even at 3x context. This
   experiment used a small CNN, not a fundamentally different context mechanism — if the raw
   information is genuinely hard to extract at this training-data scale, a same-sized model trained
   on ~6x more data would plausibly still fall well short of solving it, which is consistent with
   (not proof of) what was actually observed.

184 pairs is a meaningful, bounded expansion (per `reports/DATASET_AUDIT.md`), not "thousands of
random pairs" — and it still wasn't enough. That is itself the honest finding, not a reason to keep
scaling data arbitrarily without a specific reason to believe more would cross the threshold.

## Verdict

**REJECT.** Learned-only is far worse than classical alone (expected, and irrelevant to production
since it would never run standalone). The hybrid — the actually-relevant comparison, since the
brief's design never replaces classical candidate generation — shows **zero measurable benefit or
harm**: identical predictions to classical on every pair. Not integrated. Do not scale training data
further without a specific, evidenced reason (per the dataset-sufficiency discipline) — the dense-grid
proposal approach at this model size and training scale is not close enough to warrant it.

## Production impact

None. `pipeline/`, `model/`, and `data/` are all untouched; every artifact here lives under
`experiments/learned_candidate_generator/`.
