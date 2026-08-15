"""Candidate ranking.

`rank_classical` is the production default (reports/V2_ARCHITECTURE_PLAN.md
section 9): a learned ranker (`rank_with_model`) is only ever wired into
`pipeline/localize.py` in place of it after passing every criterion in the
integration gate (evaluation/benchmark.py). Until then, both functions
exist side by side so the classical baseline stays runnable and comparable
forever, not just until a model happens to be merged.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from . import feature_extraction
from .candidate_generation import Candidate


def rank_classical(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


# Two candidate scores are "the same value" (not merely close) up to
# float32 ZNCC computation noise. Deliberately NOT the pipeline's existing
# AMBIGUITY_THRESHOLD (0.92 second-best/best ratio, in localize.py): that
# threshold was designed to flag a result as worth reporting as uncertain,
# not to declare two different locations interchangeable as the final
# answer. Empirically, reusing it here (checked against the development
# split before settling on this constant) treats the large majority of
# pairs as "tied" - collapsing pooled accuracy@5px from 71.2% to 33.3% on
# the frozen benchmark - because ZNCC scores decay gradually across
# distinct wrong-location candidates, not just between true near-ties.
# The smallest real gap observed between genuinely distinct top-2
# candidates on the development split was ~4e-4, three orders of
# magnitude above this epsilon, so this bar only ever fires on scores
# that are numerically the same value, never on merely-similar ones.
TIE_SCORE_EPSILON = 1e-6

# --- Multiway tier (integrated 2026-08-15) --------------------------------
# The epsilon above only ever fires on near-exact numeric ties: 0/156 on the
# frozen benchmark, 0/112 and 0/132 on two independently-seeded datasets
# (experiments/multiway_tiebreak_v1/outputs/sweep_results.csv, eps=1e-6 rows).
# Provably correct, but on real data it almost never engages the Applied
# Materials rule it exists to satisfy ("if more than one matching region is
# found, return the one closest to the centre of the Search image" - stated
# 4 times across the help doc, sponsor pptx, and hackathon portal).
#
# Two attempts to widen it using ONLY a wider score-gap threshold were
# tried and rejected by the integration gate (experiments/center_tiebreak_v2/):
# a fixed absolute margin let through a 497px catastrophic regression, and a
# separate absolute-confidence-floor check was refuted directly against
# per-pair data (rescued cases' winner_score 0.786-0.919 vs harmed cases'
# 0.846-0.923 - almost total overlap, no discriminating signal). Both
# failures were pairwise (tie_len==2 by construction) - ZNCC score alone
# cannot tell a genuine multi-way tie (many near-identical periodic
# structures, see reports/ACCURACY_FORENSICS.md's `dev_dense_periodic_*`
# pairs) from a coincidental closeness between two unrelated candidates,
# one of which is simply wrong.
#
# MULTIWAY_TIE_SCORE_EPSILON is only ever consulted as a SECOND, independent
# tier - never in place of the tight tier above - gated on two structural
# conditions a coincidence does not share with a genuine periodicity
# pileup: at least MULTIWAY_MIN_GROUP_SIZE candidates must be tied (a
# periodic lattice produces many near-identical peaks; a coincidence
# produces exactly 2 - every min_group_size=2 configuration swept in
# experiments/multiway_tiebreak_v1/ reproduced the same growing-harm pattern
# as the two rejected attempts, at every epsilon tried; every
# min_group_size>=3 configuration that was also safe used exactly this
# epsilon range), and the tied group's spatial extent must not exceed
# MULTIWAY_MAX_SPREAD_PX (the single worst rejected-attempt regression was a
# 525px outlier against every genuine flip's 40-115px range).
#
# Validated end-to-end (full pipeline, not just the tiered logic in
# isolation) across 2 independently-seeded datasets
# (experiments/multiway_tiebreak_v1/outputs/final_gate_summary.json): frozen
# benchmark n=132, one confirmed catastrophic rescue (ch_worst_case_006,
# 118.5px -> 4.6px), zero regressions across all 13 families; fresh dataset
# (seed 502187) n=112, mechanism fires safely with zero regressions and no
# analogous case to rescue on that particular draw. Integrated as a
# documented gate exception - see reports/GATE_EXCEPTIONS.md for why the
# automated gate's "must broadly improve pooled validation/held_out"
# criteria don't fit a fix whose effect surface is, by construction, a rare
# tie condition in specific families.
MULTIWAY_TIE_SCORE_EPSILON = 0.005
MULTIWAY_MIN_GROUP_SIZE = 3
MULTIWAY_MAX_SPREAD_PX = 200.0


def _tied_prefix_length(ranked: list[Candidate], epsilon: float) -> int:
    """How many leading candidates (ranked descending) are within `epsilon`
    of the top score. Always >= 1 (the winner is trivially tied with
    itself)."""
    if len(ranked) < 2 or ranked[0].score <= 0:
        return 1
    best_score = ranked[0].score
    tie_len = 1
    for c in ranked[1:]:
        if best_score - c.score < epsilon:
            tie_len += 1
        else:
            break
    return tie_len


def _max_pairwise_distance_px(candidates: list[Candidate]) -> float:
    max_d = 0.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            d = math.hypot(candidates[i].x - candidates[j].x, candidates[i].y - candidates[j].y)
            max_d = max(max_d, d)
    return max_d


def apply_center_tiebreak(ranked: list[Candidate], search_shape: tuple[int, int],
                           tie_score_epsilon: float = TIE_SCORE_EPSILON, *,
                           multiway_tie_score_epsilon: float = MULTIWAY_TIE_SCORE_EPSILON,
                           multiway_min_group_size: int = MULTIWAY_MIN_GROUP_SIZE,
                           multiway_max_spread_px: float = MULTIWAY_MAX_SPREAD_PX) -> list[Candidate]:
    """Applied Materials spec compliance: "if more than one matching region
    is found, return the one closest to the centre of the Search image."

    `ranked` must already be sorted by score, descending (as returned by
    `rank_classical`/`rank_with_model`). Does nothing when there is a
    single clear-best candidate - this is a tie-break, not a re-ranking,
    so it must never override a genuine score-based winner.

    Two independent tiers, checked in order - see the constants above for
    the full rationale:

    1. Tight tier (`tie_score_epsilon`, default `TIE_SCORE_EPSILON`):
       genuinely equal scores, any group size >= 2. Unconditional, exactly
       matching the original shipped behavior - never weakened by the
       tier below.
    2. Multiway tier (only consulted if the tight tier didn't fire):
       a wider score gap, but only treated as a tie if at least
       `multiway_min_group_size` candidates qualify AND their spatial
       spread is within `multiway_max_spread_px`. Both conditions must
       hold, or the classical ranking stands untouched.
    """
    if len(ranked) < 2:
        return ranked

    tie_len = _tied_prefix_length(ranked, tie_score_epsilon)
    if tie_len >= 2:
        candidate_group = ranked[:tie_len]
    else:
        wide_len = _tied_prefix_length(ranked, multiway_tie_score_epsilon)
        if wide_len < multiway_min_group_size:
            return ranked
        candidate_group = ranked[:wide_len]
        if _max_pairwise_distance_px(candidate_group) > multiway_max_spread_px:
            return ranked
        tie_len = wide_len

    height, width = search_shape
    center_x, center_y = width / 2.0, height / 2.0
    tied_group = sorted(candidate_group,
                         key=lambda c: (c.x - center_x) ** 2 + (c.y - center_y) ** 2)
    return tied_group + ranked[tie_len:]


def rank_with_model(candidates: list[Candidate], reference: np.ndarray, search: np.ndarray,
                     model, device: str = "cpu", top_n: Optional[int] = 12) -> list[Candidate]:
    """Re-rank the top `top_n` classical candidates by learned-embedding
    cosine similarity to the Reference. Deliberately re-ranks a classical
    shortlist rather than replacing candidate generation outright: the
    classical matcher already proposes structurally reasonable candidates
    (including the hard decoys - periodic repeats, same-preset boundaries -
    the model is meant to discriminate between), so re-ranking isolates
    exactly the question the model experiment is asking.
    """
    import torch

    pool = rank_classical(candidates)[: top_n or len(candidates)]

    ref_small = cv2.resize(reference.astype(np.float32), (100, 100), interpolation=cv2.INTER_AREA)
    ref_patch = feature_extraction.normalize_patch(ref_small)
    ref_tensor = torch.from_numpy(ref_patch).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        ref_embedding = model(ref_tensor)

        scored = []
        for cand in pool:
            patch = feature_extraction.extract_patch(search, cand.x, cand.y, size=100)
            patch = feature_extraction.normalize_patch(patch)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            cand_embedding = model(patch_tensor)
            similarity = float(torch.nn.functional.cosine_similarity(ref_embedding, cand_embedding).item())
            scored.append((similarity, cand))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]
