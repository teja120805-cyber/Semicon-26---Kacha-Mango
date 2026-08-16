"""Cross-hypothesis consensus re-ranking.

Idea: pure arg-max classical ranking (`pipeline.ranking.rank_classical`)
selects the single highest ZNCC score among deduplicated candidates. But a
genuine correct match is often found as a competitive peak by MULTIPLE
scale/rotation hypotheses (the true structure tolerates a slightly-wrong
scale/rotation guess - see candidate_generation.deduplicate_by_location's
own docstring for exactly this observation), while an isolated periodic-
aliasing peak (reports/ACCURACY_FORENSICS.md Finding 1) may be more
hypothesis-specific - only certain scale/rotation combinations happen to
line up with that particular wrong repeat's spacing. This experiment
re-ranks deduplicated candidates by combining their raw ZNCC score with a
"cross-hypothesis support" signal: how much ZNCC-score-weighted backing
from the RAW (pre-dedup) candidate pool - i.e. from possibly many
different scale/rotation hypotheses - clusters near this same location.

Never modifies pipeline/ - imports it unmodified (candidate_generation.py,
Candidate) and only adds a new ranking function alongside
pipeline.ranking.rank_classical / rank_with_model, following the exact same
signature convention those two already use.
"""
from __future__ import annotations

import numpy as np

from pipeline.candidate_generation import Candidate


def compute_support(deduped: list[Candidate], raw: list[Candidate], radius_px: float) -> list[float]:
    """For each deduped candidate, sum the ZNCC scores of every raw
    candidate (from any scale/rotation hypothesis, including itself) within
    radius_px of its location - "how much independent hypothesis weight
    backs this location", not just a raw count, so a handful of
    high-confidence hypotheses agreeing outweighs many weak/noisy ones."""
    if not raw:
        return [0.0 for _ in deduped]
    raw_xy = np.array([[c.x, c.y] for c in raw])
    raw_scores = np.array([c.score for c in raw])
    support = []
    for c in deduped:
        d2 = (raw_xy[:, 0] - c.x) ** 2 + (raw_xy[:, 1] - c.y) ** 2
        mask = d2 <= radius_px ** 2
        support.append(float(raw_scores[mask].sum()))
    return support


def rank_consensus(deduped: list[Candidate], raw: list[Candidate], *,
                    alpha: float, radius_px: float = 15.0) -> list[Candidate]:
    """Re-rank deduplicated candidates by score + alpha * normalized
    support. `alpha=0.0` is exactly `rank_classical` (a built-in sanity
    check the sweep script uses). Support is normalized by the pool's own
    max support so `alpha` has a comparable scale to the ZNCC score
    (roughly 0-1) regardless of how many hypotheses/candidates a given
    pair produces."""
    if not deduped:
        return deduped
    support = compute_support(deduped, raw, radius_px)
    max_support = max(support) if support else 0.0
    norm_support = [s / max_support if max_support > 0 else 0.0 for s in support]
    combined = [c.score + alpha * ns for c, ns in zip(deduped, norm_support)]
    order = sorted(range(len(deduped)), key=lambda i: combined[i], reverse=True)
    return [deduped[i] for i in order]
