# Experiment: embedding_reranker_v1

**Status: REJECTED — failed the integration gate on every accuracy/failure criterion, across all 3
required training seeds.** Not integrated into `pipeline/`. See the repository-root
`reports/V2_MODEL_EVALUATION_REPORT.md` for the full writeup (numbers, diagnosis, and what would need to change
to revisit this) — this file is a short pointer plus the raw artifacts, not a duplicate report.

## What this experiment was

The first candidate learned re-ranker for V2's classical localization baseline: a small CNN embedding
network (`model/architecture.py::EmbeddingNet`, 64,992 params) trained with triplet loss on
hard negatives mined from the classical candidate pool, intended to re-rank
`pipeline/candidate_generation.py`'s top classical candidates by embedding similarity
(`pipeline/ranking.py::rank_with_model`).

## Artifacts in this directory

- `gate_results/integration_gate_seed{101,102,103}.json` — full per-split, per-family integration-gate
  comparison against the classical baseline, one file per training seed.
- `training_histories/training_history_seed{101,102,103}.json` — per-epoch train/early-stopping loss
  for each of the 3 training runs (shows training loss collapsing to ~0 while early-stopping loss
  plateaus — the overfitting signature discussed in the root report).
- `checkpoints/embedding_net_seed{20260101,20260102,20260103}.pt` (+ matching
  `training_history_seed*.json`) — the trained checkpoints themselves, reproducible from
  `model/train.py` given the documented seeds (see `model/TRAINING_PROTOCOL.md`). Kept inside this
  experiment's own directory, not `model/checkpoints/`, since the experiment was rejected — a
  production model directory should only ever hold a model that passed the integration gate.

## One-line result

Learned re-ranking roughly halved accuracy@5px and doubled-to-tripled the catastrophic-failure
(>50px) rate on every evaluated split, consistently across all 3 seeds — training-data scale (72
triplets from only 2 structural families) is the identified cause, not a training-procedure bug.

## Production impact

None. `pipeline/ranking.py::rank_classical` remains the sole default ranking path
(`pipeline/localize.py`'s `ranking_mode` defaults to `"classical"`); nothing under `pipeline/` or
`generator/` was modified by this experiment.

## Note on dataset version

These gate results were produced against the pre-Phase-4 dataset (`driftsensev2.0.0`, before the
cross-split RNG-seeding fix documented in `reports/DATASET_AUDIT.md`). Not re-run against the
corrected `driftsensev2.1.0` dataset — the rejection margin (accuracy roughly halved, catastrophic
rate 2-3x on every split, all 3 seeds) is far too large to plausibly be explained by that leakage,
and the root cause (72 triplets from 2 of 13+ structural families) is a training-data-construction
problem independent of which benchmark version scores it.
