# experiments/learned_reranker_v2 — REJECT (7.5x more data did not fix it; still regresses hard, all 3 seeds)

## Summary

**Verdict: REJECT — same failure mode as `embedding_reranker_v1`, not fixed by more data.**
This experiment executed the one concrete, previously-unexecuted recommendation from
`reports/ACCURACY_IMPROVEMENT_PHASE.md` ("what would be worth doing next" #2) and
`reports/DATASET_AUDIT.md` section 5: expand the embedding re-ranker's training data from 72
triplets (24 pairs / 3 families) to 537 triplets (180 pairs / 15 families) — a 7.5x increase,
matching the audit's bounded recommendation almost exactly. Despite this, **the re-ranker still
catastrophically regresses accuracy on every evaluated split, on all 3 independently-trained
seeds**: pooled accuracy@5px on the matched splits (validation/held_out/challenge/
cross_generator) drops from a 77.3% classical baseline to 38.6–40.9% — worse than the original
72-triplet `embedding_reranker_v1` in relative terms on some splits. **The data-scale hypothesis
from `DATASET_AUDIT.md` is now tested and falsified at this scale**: more training data alone
does not fix this approach. Production is untouched; `pipeline/ranking.py::rank_classical`
remains the default, unmodified.

## 1. What was executed

- **Data**: `generate_data.py` generated 180 new pairs across 15 structural families
  (`ldev_strip_anchor`, `ldev_single_mat`, `ldev_dense_periodic`, `ldev_mat_boundary`,
  `ldev_same_preset_boundary`, `ldev_multi_mat`, `ldev_rotation_drift`, `ldev_scale_drift`,
  `ldev_heavy_noise`, `ldev_linewidth_bias`, `ldev_vignette_gamma`,
  `ldev_combined_acquisition`, `ldev_barrel_charging`, `ldev_speckle_saltpepper`,
  `ldev_worst_case`) — mirroring every major structural/degradation axis validated elsewhere in
  the benchmark, per the audit's exact guidance. Seed `910001`, fully independent of
  production (`777001`), the forensics sweep (`830001`), and every other experiment's seed —
  by construction, via `generator.dataset_generator`'s per-pair
  `default_rng([seed, family_salt(split, family_name), pair_index])` scheme, there is no
  possibility of RNG-level overlap with any existing split. Written to this experiment's own
  `data/development/` — never touches production `data/`.
- **Triplet mining**: `model.dataset.TripletPatchDataset` (unmodified), pointed at the new
  data root, mined **537 triplets** (501 train / 36 early-stop) — within the audit's predicted
  450–540 range, confirming the expansion behaved as expected.
- **Training**: `train.py` — a thin copy of `model/train.py`'s exact training loop (same
  `EmbeddingNet` architecture, same Adam/triplet-margin-loss/early-stopping strategy, same
  dev-only-never-validation discipline per `model/TRAINING_PROTOCOL.md`); the only change is
  `EARLY_STOP_FAMILIES` pointing at this dataset's analogous hardest family
  (`ldev_dense_periodic` vs. the original `dev_dense_periodic`). Trained 3 seeds
  (`20260101`/`02`/`03`), matching `embedding_reranker_v1`'s own 3-seed evaluation discipline.
- **Evaluation**: `run_experiment.py` uses the **exact production code path** —
  `pipeline.localize.localize(ranking_mode="learned", model=...)` via
  `evaluation.evaluate.evaluate_all`, unmodified — no custom harness needed, since
  `ranking_mode="learned"` is already a first-class option in production `localize()`.

## 2. Training behaved normally

All 3 seeds converged smoothly with no training-loop errors: train loss dropped from ~0.17 to
~0.003–0.007 within 4–7 epochs, early stopping triggered at epoch 12–15 (patience=8) with best
checkpoints at epoch 4, 6, and 7 respectively. No sign of a training-mechanics bug — the model
is learning *something* from the triplets, just not something that transfers.

| Seed | Best epoch | Best dev early-stop loss |
|---:|---:|---:|
| 20260101 | 7 | 0.2036 |
| 20260102 | 4 | 0.1611 |
| 20260103 | 6 | 0.1430 |

## 3. Evaluation result — regresses everywhere, every seed

Evaluated on `validation`/`held_out`/`challenge`/`cross_generator` (132 pairs; `development` is
training data here, excluded from scoring exactly as `scripts/evaluate_model.py` does for the
production split).

| Split | Baseline acc@5px | Seed 101 | Seed 102 | Seed 103 |
|---|---:|---:|---:|---:|
| validation | 0.900 | 0.600 | 0.550 | 0.500 |
| held_out | 0.650 | 0.175 | 0.300 | 0.250 |
| challenge | 0.750 | 0.375 | 0.188 | 0.313 |
| cross_generator | 0.800 | 0.550 | 0.550 | 0.650 |
| **Pooled (n=132)** | **0.7727** | **0.409** | **0.386** | **0.402** |

Every split, every seed, regresses — no exceptions. Integration gate: **FAILED every criterion**
except runtime (6) and cross-seed stability (7, in the sense that all 3 seeds *agree* on
failing). This is the same "fails everything, all seeds" shape as the original
`embedding_reranker_v1` result (per-split range ~30–55%), not an improvement in kind — despite
7.5x more training data.

## 4. Why more data didn't fix it

This falsifies the specific hypothesis `reports/DATASET_AUDIT.md` section 5 proposed (72
triplets from 3 families was the direct cause of the original failure; a bounded ~450–540
triplet, ~12-15 family expansion would address it). It did not. Plausible remaining
explanations, none confirmed here and worth flagging honestly rather than guessing further:

1. **Still not enough data for a from-scratch CNN embedding**, even at 537 triplets — modern
   metric-learning setups typically use orders of magnitude more; "6-7x more" may have been the
   right *shape* of fix but the wrong *scale*, and the audit itself only ever claimed this was
   a "bounded, not huge" first step, not a guaranteed fix.
2. **Triplet correlation**: many triplets share the same underlying macro canvas / family
   (12 pairs per family, same crop_mode), so 537 triplets may carry meaningfully less than
   537 independent "lessons" — effective diversity could still resemble the original problem
   more than the raw triplet count suggests.
3. **A methodological issue independent of data volume** — e.g. `normalize_patch`'s zero-mean/
   unit-std normalization, the fixed 100×100 resize, or the triplet-margin-loss formulation
   itself may not be well-suited to distinguishing periodic near-duplicates, regardless of how
   much data it sees. This experiment cannot distinguish between explanations 1–3; it only
   confirms the specific "72→537 triplets" fix from the audit does not work.

## 5. What this means for the "learned component" direction generally

Combined with the prior `learned_candidate_generator` result (recall@5px only 3.0% even with a
dense grid — `reports/ACCURACY_IMPROVEMENT_PHASE.md` #3) and the original
`embedding_reranker_v1` rejection, this is now **the second independent learned-re-ranking
attempt to fail badly** at two different data scales (72 and 537 triplets). The evidence is
mounting that a small from-scratch CNN embedding re-ranker, trained only on this project's
synthetic generator's output, is not a productive direction without a fundamentally different
approach (pretrained backbone / different loss / far larger data) — none of which were in scope
for this bounded follow-up. Classical `rank_classical` remains, by a wide and now
twice-confirmed margin, the stronger choice for production.

## Reproduce

```
cd experiments/learned_reranker_v2
python generate_data.py     # ~180 pairs, seed 910001, writes data/development/
python train.py             # trains 3 seeds -> checkpoints/
python run_experiment.py    # evaluates all 3 seeds via production ranking_mode="learned"
```

Outputs: `data/development/` (180 pairs), `checkpoints/*.pt` + `training_history_seed*.json`,
`outputs/per_pair_results_learned_seed*.csv`, `outputs/cross_seed_summary.json`.
