# Experiments

Every experiment that touches `model/` or proposes a `pipeline/` change lives here as
`experiments/<experiment_name>/`, per `reports/V2_ARCHITECTURE_PLAN.md` section 12 and the mandatory
integration gate in section 9.

## Rules

- An experiment imports shared, unmodified code from `generator/`, `pipeline/`, `model/` where nothing
  changed — it never edits those packages in place to try something out.
- An experiment gets its own config and its own output directory (or references artifacts already
  written under `outputs/reports/` — see each experiment's `REPORT.md` for what's copied vs. referenced).
- **A change only merges into `pipeline/` or `model/` proper after it passes every criterion in the
  integration gate** (`evaluation/benchmark.py`), cited in the commit that merges it. Until then, it
  stays here, with an honest report — including if it failed.
- Production code (`pipeline/ranking.py::rank_classical` as the default ranking path) never changes
  automatically because an experiment ran. That is always a separate, deliberate code change.

## Current experiments

| Experiment | Status | One-line result |
|---|---|---|
| [`embedding_reranker_v1/`](embedding_reranker_v1/REPORT.md) | **Rejected** | CNN embedding re-ranker failed the integration gate on every accuracy/failure criterion, across all 3 required seeds (see `reports/V2_MODEL_EVALUATION_REPORT.md` for the full writeup) — not integrated; production ranking stays classical. |
| [`accuracy_forensics/`](../reports/ACCURACY_FORENSICS.md) | Analysis (not a pipeline candidate) | Controlled single-factor/interaction sweeps identifying boundary presence, periodicity/aliasing, and rotation/scale hypothesis-grid misalignment as the dominant failure mechanisms — not noise/drift/jitter. Report lives in `reports/` since it's diagnostic, not a candidate change. |
| [`wider_candidate_pool/`](wider_candidate_pool/REPORT.md) | **Rejected** | Widening the classical candidate pool (more peaks/hypothesis, tighter NMS) produced bit-identical predictions to baseline on every pair — structurally a no-op under pure arg-max ranking, not a failed improvement. Clarifies that fixing periodicity needs a re-ranking stage, not more candidates alone. |
| [`finer_hypothesis_grid/`](finer_hypothesis_grid/REPORT.md) | **Near-miss, not integrated** | Densifying the scale/rotation hypothesis grid (81 vs. 25) improved held_out (+5pp) and challenge (+6.2pp) with no per-family regression and acceptable runtime (3.17x, corrected), but validation tied rather than improved (already at a 90% ceiling) — fails the gate on that one criterion. The strongest result of any experiment so far; worth revisiting with a less ceiling-limited validation set. |
| [`periodicity/`](periodicity/REPORT.md) | **Rejected** (both variants) | Gradient-domain and intensity+gradient-ensemble candidate scoring were tested as alternatives to raw-intensity ZNCC for periodicity-driven candidate-generation failures. Both net-negative (gradient: -9 rescue/break; ensemble: -4, and over the runtime budget). Rules out "different classical correlation representation" as a periodicity fix. |
| [`rotation_scale/`](rotation_scale/REPORT.md) | **Rejected** | Coarse-to-fine local rotation/scale refinement (cheaper than a globally denser grid) — net +2 rescue, weaker than `finer_hypothesis_grid`'s +5 at similar (not dramatically lower) runtime cost. No clear advantage over the already-tested alternative. |

### 90%-accuracy campaign (2026-08-16) — see [`ACCURACY_90_CAMPAIGN.md`](ACCURACY_90_CAMPAIGN.md) for the consolidated writeup

Eight further ideas, tried after A2 (`scale_range_v1`)/A6 (`multiway_tiebreak_v1`) integration,
against the verified production baseline (74.36%@5px, n=156). All eight rejected; the
consolidated report explains the cross-cutting mechanism finding (raw ZNCC score, not candidate
discovery/ranking, is the real bottleneck on periodicity failures) that four of them converged on
independently.

| Experiment | Status | One-line result |
|---|---|---|
| [`cross_hypothesis_consensus_rerank/`](cross_hypothesis_consensus_rerank/REPORT.md) | **Rejected** | Re-ranking by cross-hypothesis spatial support density — clean no-op, dev sweep never beat `alpha=0` at any tested strength. |
| [`subpixel_grid_refinement/`](subpixel_grid_refinement/REPORT.md) | **Rejected** | Joint scale/rotation subpixel interpolation — real effect on 156/156 pairs (verified), but max shift 0.305px, far below the 5px tolerance; prior grid-density integrations already closed most of the gap this would recover. |
| [`hough_subpatch_voting/`](hough_subpatch_voting/REPORT.md) | **Rejected** | Re-ranking by sub-patch geometric self-consistency — true no-op, verified directly (0/156 coordinate changes even at an aggressive exploratory setting). Periodic decoys are self-similar at the sub-patch level too. |
| [`learned_reranker_v2/`](learned_reranker_v2/REPORT.md) | **Rejected** | CNN embedding re-ranker retried with 7.5x more training data (537 vs. 72 triplets) than the original rejected attempt — still catastrophic regression on every split, all 3 seeds. Falsifies the specific "more data" hypothesis from `reports/DATASET_AUDIT.md`. |
| [`keypoint_candidate_fusion/`](keypoint_candidate_fusion/REPORT.md) | **Rejected** | ORB keypoint + RANSAC candidate proposal fused into the pool — proposes a real, competing candidate on 56/156 pairs, but never once outscores the classical winner under raw ZNCC arbitration. |
| [`pyramid_periodicity_search/`](pyramid_periodicity_search/REPORT.md) | **Rejected** | Coarse (blurred) candidate proposal + local full-resolution refinement — found a candidate 0.4px from ground truth on one currently-failing pair, scored 4x lower than the (wrong) classical winner. Bit-identical to baseline on all 156 pairs. |
| [`prominence_rerank/`](prominence_rerank/REPORT.md) | **Rejected** | Re-ranking by peak prominence vs. a generic local annulus — a real, non-inert signal, but net HARMFUL when active (14-15 pairs broken per 4 rescued at full benchmark scale); auto-tunes to a no-op under disciplined dev-only tuning. |
| [`pitch_aware_prominence/`](pitch_aware_prominence/REPORT.md) | **Rejected** | Surgical follow-up to `prominence_rerank` using a per-pair MEASURED periodic pitch instead of a generic radius — passed 6/7 integration-gate criteria on the production-seed benchmark (0.7436→0.7500), but **failed an independent second-seed validation** (0.6618→0.6324, 0 rescues / 4 breaks). The campaign's closest near-miss, and a cautionary example of single-seed overfitting. |
| [`pitch_aware_prominence_v2/`](pitch_aware_prominence_v2/REPORT.md) | **Rejected** | Diagnosed and fixed the exact bonus-driven flip that broke the v1 formula on all 10 pairs it ever changed (both seeds) — clipping the re-score to a penalty-only term (`score + gamma*min(prominence,0)`) fully fixes all 4 second-seed breaks, but is a complete no-op everywhere else (0/156 pairs changed at any tested strength, up to 4x v1's peak gamma). Conclusively shows v1's apparent gain was never real periodicity signal. |
