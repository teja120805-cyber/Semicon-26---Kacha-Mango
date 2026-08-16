# experiments/keypoint_candidate_fusion — REJECT (real proposals, never win)

## Summary

**Verdict: REJECT — a genuine mechanism, but bit-identical to baseline everywhere.** ORB
keypoint matching + RANSAC similarity-transform estimation proposes an additional, off-grid
candidate location on 56/156 pairs of the frozen benchmark (36% — not rare), scored with the
exact same ZNCC metric as every other candidate for a fair comparison. But that proposal **never
once outscores the classical grid's own winner**, on any of the 56 pairs it fires on, at any of
9 tested (ratio_threshold, min_inliers) configurations. Pooled accuracy@5px: 0.7436 → 0.7436,
bit-for-bit identical predictions. Production is untouched.

## 1. The idea

`reports/ACCURACY_FORENSICS.md` identifies `candidate_generation`-stage failures (the true
location never enters the pool at all - 31/156, the single largest failure category) as
distinct from `candidate_ranking` failures (true location is in the pool but doesn't win).
Two already-rejected experiments ruled out two ways of fixing `candidate_generation` within the
existing whole-template-ZNCC-correlation framework: `wider_candidate_pool` (more peaks per
hypothesis doesn't help, because the true location isn't even a local maximum on the SAME
correlation surface) and `periodicity/` (alternative whole-template scoring functions -
gradient-domain, gradient+intensity ensemble - both made things worse). This experiment tries a
structurally different candidate SOURCE: sparse ORB keypoint correspondence (corners/blobs, not
periodic bulk texture) between the Reference and the Search image, robustly fit via RANSAC to a
similarity transform, proposing a candidate location **outside the fixed 99-hypothesis grid
entirely**. The proposal is scored by a fresh local ZNCC correlation at the transform-estimated
continuous scale/rotation (not an arbitrary confidence score), so it competes fairly under the
unmodified `rank_classical` arg-max - this experiment adds a candidate source, not a new scoring
function or ranking rule.

## 2. Implementation notes worth recording

- **Default ORB detector thresholds are unusable on this dataset.** `cv2.ORB_create()`'s
  defaults (`edgeThreshold=31`, `fastThreshold=20`) found only ~19 keypoints on a 100x100
  reference template - too few for any reliable matching. This dataset's synthetic SEM imagery
  is smooth and low-texture by design; loosening the detector (`edgeThreshold=5`,
  `patchSize=15`, `fastThreshold=5`, `nlevels=12`, `nfeatures=1000`) brought that up to
  ~970-1000 keypoints, which is what makes any matching possible at all - documented directly in
  `keypoint_gen.py` so this isn't silently rediscovered later.
- **Lowe's ratio test is itself sensitive to periodicity** - a directly relevant mechanistic
  finding. On `dev_dense_periodic_002` with default (strict) ORB thresholds, ratio<0.75 kept
  **zero** good matches out of 19 raw matches (every second-best match was nearly as good as the
  best - the same ambiguity signature whole-template ZNCC shows on periodic content). With
  loosened detector thresholds, match counts recovered (37-490 good matches across ratio
  0.6-0.95 on that pair), confirming the earlier zero-match result was a detector-sensitivity
  artifact, not evidence that periodicity defeats keypoint matching outright.
- `keypoint_gen.py::generate_keypoint_candidate` - detects ORB keypoints on a nominal-scale
  (10.0, 0deg) template and the full Search image, `BFMatcher(NORM_HAMMING)` + Lowe's ratio
  test, `cv2.estimateAffinePartial2D(..., RANSAC)` for a robust similarity transform, decomposes
  it for an implied scale/rotation (guarded against outlier decompositions - falls back to the
  nominal scale if the implied value is outside [7, 13] or rotation exceeds 8deg), builds one
  verification template at that continuous scale/rotation, and does a small local ZNCC
  correlation in a window around the transform-projected location. Returns `None` (no proposal)
  if keypoints/matches/inliers are too few - the pool then falls back to the unmodified
  classical grid alone.
- `harness.py::localize_keypoint_fusion` - structurally identical to `pipeline.localize.localize`,
  merging the (at most one) keypoint proposal into the raw candidate pool **before**
  deduplication; candidate generation's own 99 grid hypotheses, dedup, classical ranking, center
  tiebreak, and refinement are all unmodified production calls.

## 3. Evaluation

### Dev sweep (n=24, ratio_threshold ∈ {0.65, 0.75, 0.85} × min_inliers ∈ {4, 6, 10})

| ratio_threshold | min_inliers | acc@5px | mean_err_px | kp_proposed | kp_won |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 4 | 0.583 | 111.50 | 9/24 | 0/24 |
| 0.65 | 6 | 0.583 | 111.50 | 2/24 | 0/24 |
| 0.65 | 10 | 0.583 | 111.50 | 2/24 | 0/24 |
| 0.75 | 4 | 0.583 | 111.50 | 17/24 | 0/24 |
| 0.75 | 6 | 0.583 | 111.50 | 11/24 | 0/24 |
| 0.75 | 10 | 0.583 | 111.50 | 5/24 | 0/24 |
| 0.85 | 4 | 0.583 | 111.50 | 22/24 | 0/24 |
| 0.85 | 6 | 0.583 | 111.50 | 21/24 | 0/24 |
| 0.85 | 10 | 0.583 | 111.50 | 17/24 | 0/24 |

Every single configuration - even the most permissive (`ratio_threshold=0.85`, proposing on
22/24 dev pairs) - produced **exactly** baseline accuracy and mean error, because **the keypoint
candidate never won even once**, on any dev pair, at any config. The dev-only selection procedure
picked the first-sorted tied config (`ratio_threshold=0.65, min_inliers=4`) since accuracy and
mean error were identical across the whole grid.

### Full frozen benchmark at the dev-selected config

| Split | Baseline acc@5px | Candidate acc@5px | kp_proposed |
|---|---:|---:|---:|
| development | 0.583 | 0.583 | 9/24 |
| validation | 0.900 | 0.900 | 15/40 |
| held_out | 0.650 | 0.650 | 11/40 |
| challenge | 0.750 | 0.750 | 8/32 |
| cross_generator | 0.800 | 0.800 | 13/20 |
| **Pooled (n=156)** | **0.7436** | **0.7436** | **56/156** |

**Bit-identical per-pair predictions everywhere** (confirmed via the integration gate's
per-family regression check - zero families regressed or improved). The keypoint mechanism
proposed a real, off-grid candidate on 56/156 pairs (36% - a substantial fraction, not a rare
edge case) but **won on 0/156** - its local ZNCC score never exceeded the classical grid's own
winner, anywhere in the benchmark.

## 4. Why this didn't work

Unlike `cross_hypothesis_consensus_rerank` (which never had a chance to disagree - alpha=0 was
mathematically a no-op) or `hough_subpatch_voting`'s dev sweep (only flipped 0/24 at the
disciplined config), this mechanism DOES actively propose competing, off-grid, real-content
candidates on over a third of the benchmark - it is not structurally inert. It simply loses,
consistently. Two plausible reasons, consistent with each other:

1. **Scores are the deciding factor, and keypoint-derived proposals score honestly** - a
   proposal built from a handful of RANSAC inliers, projected through an estimated transform,
   and locally correlated is inherently a slightly less globally-optimized location than a
   candidate that came from exhaustively correlating the FULL template over the FULL 99-
   hypothesis grid. Even when the keypoint proposal lands near the true location (verified
   directly during development: several proposals landed 33-68px from ground truth with scores
   0.76-0.88 - closer than the classical winner in those specific cases, but still short of both
   5px accuracy AND the classical winner's own score), it rarely out-scores whatever the
   classical grid's own winner already is.
2. **This reinforces `hough_subpatch_voting`'s mechanism-level finding from a different angle**:
   when the classical ranker is wrong, it's usually wrong in favor of a genuinely
   high-scoring periodic decoy - not because no other reasonable candidate exists, but because
   ZNCC score itself is fooled. A different DISCOVERY mechanism (keypoints instead of exhaustive
   correlation) doesn't help if the final arbitration is still "highest ZNCC score wins" and the
   decoy's ZNCC score is genuinely higher.

## 5. What this doesn't rule out

This only tested pure ZNCC-score arbitration between the keypoint proposal and the classical
pool. It doesn't rule out a formulation that gives the keypoint proposal a `MULTIWAY`-tiebreak-
style structural boost (e.g., "prefer the keypoint-consistent candidate when scores are within a
tolerance," similar in spirit to the already-integrated `apply_center_tiebreak`'s multiway tier)
rather than pure score arbitration - but that would be a materially different, more speculative
design than what was tested here, and this project's own `center_tiebreak_v2` history is a
direct caution against loosening tie-break margins without very strong, narrowly-scoped evidence.

## Reproduce

```
cd experiments/keypoint_candidate_fusion
python run_experiment.py
```

Outputs: `outputs/dev_sweep_results.json`, `outputs/per_pair_results_keypoint_fusion.csv`,
`outputs/keypoint_fusion_metrics.json`, `outputs/integration_gate_result.json`.
