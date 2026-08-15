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
| [`center_tiebreak_v2/`](center_tiebreak_v2/REPORT.md) | **Rejected** | A genuinely non-inert centre tie-break (`tie_score_epsilon=0.001`, empirically derived from the full 156-pair score-gap distribution — see `reports/TIE_BREAK_IMPLEMENTATION.md` for the shipped, provably-inert `1e-6` definition this was evaluated against). Rescues both documented `genuine_ambiguity` near-ties it targets (2/2) but also creates 2 new catastrophic (>50px) failures on the frozen benchmark and a new per-family regression on an independent fresh dataset — pooled accuracy@5px improves on both datasets (+1.5pp, +1.8pp) while tail metrics (mean/P90/P95/failure>50px) get worse, exactly the failure mode a pooled-only number hides. Fails gate criteria 2, 4, and 7. Not integrated; production stays at the shipped epsilon=1e-6. |
