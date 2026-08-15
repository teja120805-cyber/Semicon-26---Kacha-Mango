# DriftSense V2 — Model Training Protocol

Written before training was run, per the brief's requirement that checkpoint selection and evaluation
protocol be fixed in advance rather than adjusted after seeing held-out numbers.

## Dataset split

Training uses the **development split only** (24 pairs: `dev_strip_anchor`, `dev_single_mat`,
`dev_dense_periodic`, 8 each). `validation`, `held_out`, `challenge`, and `cross_generator` are never
read by `model/dataset.py` or `model/train.py` — there is no code path by which they could leak into
training even by accident.

For early stopping / checkpoint selection, the development split is further divided **within itself**:

| Role | Families | Pairs | Triplets (3 negatives/pair) |
|---|---|---:|---:|
| Training | `dev_strip_anchor`, `dev_single_mat` | 16 | 48 |
| Internal early-stopping | `dev_dense_periodic` | 8 | 24 |

`dev_dense_periodic` was chosen as the internal early-stopping slice because it is the hardest
in-development-split family (25%@5px in `reports/V2_BASELINE_REPORT.md`, the worst of the three) — checkpoint
selection is answering "does this generalize to the hardest in-scope case", not "did loss go down on
what it was trained on". This carve-out is entirely within `development`; it does not touch
`validation`/`held_out`/`challenge`/`cross_generator`, which remain reserved for the integration gate.

## Number of samples

72 triplets total (24 pairs x 3 hard negatives/pair). This is a genuinely small training set — stated
plainly rather than hidden, since it directly bounds how much to trust any single-seed result (see
"honest expectations" below).

## Triplet construction (hard-negative mining)

- **Anchor**: the Reference image, resized to 100x100.
- **Positive**: a 100x100 Search-image patch centered at ground truth.
- **Negative**: a 100x100 Search-image patch centered at one of the top classical-candidate locations
  (`pipeline/candidate_generation.py`, deduplicated by location) that is more than 20px from ground
  truth — i.e., a real decoy the classical matcher itself found plausible (periodic repeats,
  same-preset boundaries), not an easier random crop. Up to 3 hard negatives per pair.
- Negatives are mined **once**, at dataset-construction time, from the classical candidate pool — not
  re-mined per epoch. This keeps the training set a fixed, fully reproducible object given a seed,
  rather than a moving target across runs.

## Augmentations

**None.** This is a deliberate decision, not an oversight: the training set is already small enough
that its size is the honest limiting factor being tested. Adding augmentation would make it harder to
tell whether a result reflects genuine generalization or an augmentation-specific artifact, and this
experiment's purpose (per `reports/V2_ARCHITECTURE_PLAN.md` section 7) is to honestly benchmark whether a
learned re-ranker helps at all on data this scarce — not to squeeze maximum performance out of a
production model.

## Architecture

`model/architecture.py::EmbeddingNet` — 4 conv+BN+ReLU blocks (1→16→32→64→64 channels, stride 2 each),
global average pool, one linear layer to a 64-d embedding, L2-normalized output. 64,992 trainable
parameters (verify with `python -m model.architecture`).

## Optimizer / training hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 8 |
| Triplet margin | 0.3 |
| Max epochs | 40 |
| Early-stopping patience | 8 epochs without improvement on the internal early-stopping loss |
| Checkpoint selection | Lowest triplet loss on the internal early-stopping slice (not final epoch) |
| Loss | `torch.nn.functional.triplet_margin_loss` |
| Device | CPU (dataset is small enough that GPU offers no meaningful speed benefit; CPU keeps results reproducible on any machine without a CUDA-determinism dependency) |

## Random seeds

Trained with **three independent seeds**: `20260101`, `20260102`, `20260103` — required by the
integration gate's criterion 7 ("improvement holds across at least 2 additional random training
seeds", `reports/V2_ARCHITECTURE_PLAN.md` section 8). Each seed controls Python's `random`, NumPy, and PyTorch
RNGs (`model/train.py::set_seed`) and is passed through to `TripletPatchDataset`, so the fixed
hard-negative mining and the train/early-stop split are the only sources shared across seeds — weight
initialization and minibatch order differ.

## Evaluation protocol

1. For each trained checkpoint, run `evaluation/evaluate.py --ranking-mode learned --model-checkpoint
   <path>` over `validation`, `held_out`, `challenge`, and `cross_generator` — the model re-ranks the
   top 12 classical candidates by embedding cosine similarity (`pipeline/ranking.py::rank_with_model`);
   it never replaces candidate generation itself.
2. Run `evaluation/benchmark.py` to compare the learned-ranking results against the frozen classical
   baseline (`outputs/reports/per_pair_results.csv`) and apply every integration-gate criterion.
3. Report the outcome — pass or fail — in `reports/V2_MODEL_EVALUATION_REPORT.md`, honestly, including if it
   fails. A failing model is filed under `experiments/` with its own report; the production pipeline's
   default ranking stays classical (`pipeline/ranking.py::rank_classical`) regardless of outcome unless
   every criterion in the gate passes.

## Honest expectations, stated before training

72 triplets is a small-data regime for a from-scratch CNN, even a ~65k-parameter one. A plausible,
even likely, outcome is that the model does not clearly beat the classical baseline, or improves one
split while regressing another — that would not be a bug in the model or the experiment; it would be
the correct, informative answer to "is a from-scratch learned re-ranker justified on a dataset this
size", which is exactly what the integration gate exists to check honestly rather than assume.
